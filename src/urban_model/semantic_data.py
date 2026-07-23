from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from urban_dataset.torch_dataset import UrbanTileDataset

from .data import ExtensionDataset
from .semantic_config import SEMANTIC_NAMES, SemanticDiffusionConfig


def layers_to_semantic(layers: torch.Tensor) -> torch.Tensor:
    """Convert the twelve surface channels to CityGen-style categorical labels."""
    if layers.shape[-3] != 12:
        raise ValueError("Expected twelve city channels")
    output_shape = layers.shape[:-3] + layers.shape[-2:]
    classes = torch.zeros(output_shape, dtype=torch.long, device=layers.device)
    threshold = 0.05

    vegetation = layers[..., 7, :, :] > threshold
    water = layers[..., 0, :, :] > threshold
    building = layers[..., 8, :, :] > threshold
    road = layers[..., (1, 2, 3), :, :].amax(dim=-3) > threshold
    rail = layers[..., 11, :, :] > threshold

    # Confirmed surface transport has priority at same-level occupancy conflicts.
    # Underground and elevated transport remain in separate auxiliary arrays.
    classes = torch.where(vegetation, 1, classes)
    classes = torch.where(water, 5, classes)
    classes = torch.where(building, 2, classes)
    classes = torch.where(road, 3, classes)
    classes = torch.where(rail, 4, classes)
    return classes


def semantic_to_model_space(
    classes: torch.Tensor,
    class_count: int = len(SEMANTIC_NAMES),
) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(classes.long(), num_classes=class_count).movedim(-1, -3).float()
    return one_hot.mul(2.0).sub(1.0)


def model_space_to_semantic(values: torch.Tensor) -> torch.Tensor:
    return values.argmax(dim=-3)


def _augment_semantic(classes: torch.Tensor) -> torch.Tensor:
    turns = int(torch.randint(0, 4, ()).item())
    if turns:
        classes = torch.rot90(classes, turns, dims=(-2, -1))
    if bool(torch.rand(()) < 0.5):
        classes = torch.flip(classes, dims=(-1,))
    if bool(torch.rand(()) < 0.5):
        classes = torch.flip(classes, dims=(-2,))
    return classes


class SemanticBlockDataset(Dataset):
    """Dense square crops used to learn the unconditional local block distribution."""

    def __init__(
        self,
        config: SemanticDiffusionConfig,
        manifest: str | Path,
        *,
        augment: bool,
    ) -> None:
        self.tiles = UrbanTileDataset(manifest, include_auxiliary=False)
        self.resolution = config.resolution
        self.crop_size = config.crop_size_pixels
        self.crop_stride = config.crop_stride_pixels
        self.augment = augment
        self.crops: list[tuple[int, int, int]] = []

        if not self.tiles:
            return
        first = self.tiles[0]["x"]
        height, width = first.shape[-2:]
        if self.crop_size > min(height, width):
            raise ValueError(
                f"Crop size {self.crop_size} exceeds tile dimensions {height}x{width}"
            )
        top_values = _crop_positions(height, self.crop_size, self.crop_stride)
        left_values = _crop_positions(width, self.crop_size, self.crop_stride)
        self.crops = [
            (tile_index, top, left)
            for tile_index in range(len(self.tiles))
            for top in top_values
            for left in left_values
        ]

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, index: int) -> dict[str, Any]:
        tile_index, top, left = self.crops[index]
        tile = self.tiles[tile_index]
        layers = tile["x"][:, top : top + self.crop_size, left : left + self.crop_size]
        classes = layers_to_semantic(layers)
        if self.augment:
            classes = _augment_semantic(classes)
        x0 = semantic_to_model_space(classes)
        if x0.shape[-2:] != self.resolution:
            x0 = F.interpolate(x0.unsqueeze(0), size=self.resolution, mode="nearest")[0]
        return {
            "x0": x0,
            "tile_id": f"{tile['tile_id']}:{top}:{left}",
        }


def _crop_positions(length: int, crop_size: int, stride: int) -> list[int]:
    positions = list(range(0, length - crop_size + 1, stride))
    last = length - crop_size
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


class SemanticOutpaintingDataset(Dataset):
    """Adjacent real tiles for CityGen-style masked outpainting fine-tuning."""

    def __init__(
        self,
        config: SemanticDiffusionConfig,
        manifest: str | Path,
        *,
        augment: bool,
    ) -> None:
        self.pairs = ExtensionDataset(
            manifest,
            augment=augment,
            directions=config.directions,
            boundary_width=config.boundary_width,
            guide_length=config.guide_length,
        )
        self.resolution = config.resolution

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        classes = layers_to_semantic(pair["full_target"])
        x0 = semantic_to_model_space(classes)
        x0 = F.interpolate(x0.unsqueeze(0), size=self.resolution, mode="nearest")[0]
        known_mask = F.interpolate(
            pair["known_mask"].unsqueeze(0),
            size=self.resolution,
            mode="nearest",
        )[0]
        road_guide = F.interpolate(
            pair["boundary_guide"].unsqueeze(0),
            size=self.resolution,
            mode="nearest",
        )[0]
        return {
            "x0": x0,
            "known_mask": known_mask,
            "road_guide": road_guide,
            "tile_id": pair["tile_id"],
        }


def dataset_for_semantic_task(
    config: SemanticDiffusionConfig,
    manifest: str | Path,
    *,
    augment: bool,
) -> Dataset:
    if config.task == "block":
        return SemanticBlockDataset(config, manifest, augment=augment)
    return SemanticOutpaintingDataset(config, manifest, augment=augment)
