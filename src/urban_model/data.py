from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from urban_dataset.torch_dataset import UrbanTileDataset

from .config import LayeredDiffusionConfig

SURFACE_NAMES = (
    "terrain",
    "vegetation",
    "building",
    "road_major",
    "road_secondary",
    "road_local",
    "rail_surface",
    "water",
)
AUXILIARY_NAMES = (
    "road_underground",
    "road_elevated",
    "rail_underground",
    "rail_elevated",
    "building_height",
)
LAYER_NAMES = SURFACE_NAMES + AUXILIARY_NAMES
SURFACE_CLASS_COUNT = len(SURFACE_NAMES)
MODEL_CHANNELS = len(LAYER_NAMES)


def _crop_positions(length: int, crop_size: int, stride: int) -> list[int]:
    positions = list(range(0, length - crop_size + 1, stride))
    last = length - crop_size
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def layers_to_model_space(
    layers: torch.Tensor,
    road_vertical_masks: torch.Tensor,
    rail_vertical_masks: torch.Tensor,
) -> torch.Tensor:
    """Convert corpus arrays into one surface categorical layer plus four vertical layers."""
    if layers.shape[-3] != 12:
        raise ValueError(f"Expected 12 source channels, found {layers.shape[-3]}")
    if road_vertical_masks.shape[-3] != 4 or rail_vertical_masks.shape[-3] != 4:
        raise ValueError("Vertical transport masks must have shape [4,H,W]")

    shape = layers.shape[:-3] + layers.shape[-2:]
    classes = torch.zeros(shape, dtype=torch.long, device=layers.device)
    threshold = 0.05

    vegetation = layers[..., 7, :, :] > threshold
    water = layers[..., 0, :, :] > threshold
    building = layers[..., 8, :, :] > threshold
    major = layers[..., 1, :, :] > threshold
    secondary = layers[..., 2, :, :] > threshold
    local = layers[..., 3, :, :] > threshold
    rail = layers[..., 11, :, :] > threshold

    classes = torch.where(vegetation, 1, classes)
    classes = torch.where(water, 7, classes)
    classes = torch.where(building, 2, classes)
    classes = torch.where(local, 5, classes)
    classes = torch.where(secondary, 4, classes)
    classes = torch.where(major, 3, classes)
    classes = torch.where(rail, 6, classes)

    surface = F.one_hot(classes, num_classes=SURFACE_CLASS_COUNT).movedim(-1, -3).float()
    auxiliary = torch.stack(
        [
            road_vertical_masks[..., 1, :, :],
            road_vertical_masks[..., 2, :, :],
            rail_vertical_masks[..., 1, :, :],
            rail_vertical_masks[..., 2, :, :],
            layers[..., 9, :, :] * building.float(),
        ],
        dim=-3,
    ).float()
    values = torch.cat([surface, auxiliary], dim=-3)
    return values.mul(2.0).sub(1.0)


def model_space_to_layers(values: torch.Tensor, *, threshold: float = 0.0) -> dict[str, torch.Tensor]:
    if values.shape[-3] != MODEL_CHANNELS:
        raise ValueError(f"Expected {MODEL_CHANNELS} model channels, found {values.shape[-3]}")
    surface = values[..., :SURFACE_CLASS_COUNT, :, :].argmax(dim=-3)
    auxiliary = values[..., SURFACE_CLASS_COUNT:, :, :]
    building = surface == 2
    return {
        "surface": surface,
        "road_underground": auxiliary[..., 0, :, :] > threshold,
        "road_elevated": auxiliary[..., 1, :, :] > threshold,
        "rail_underground": auxiliary[..., 2, :, :] > threshold,
        "rail_elevated": auxiliary[..., 3, :, :] > threshold,
        "building_height": auxiliary[..., 4, :, :].add(1.0).div(2.0).clamp(0.0, 1.0)
        * building.float(),
    }


def _augment(values: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    turns = int(torch.randint(0, 4, ()).item())
    if turns:
        values = torch.rot90(values, turns, dims=(-2, -1))
        valid_mask = torch.rot90(valid_mask, turns, dims=(-2, -1))
    if bool(torch.rand(()) < 0.5):
        values = torch.flip(values, dims=(-1,))
        valid_mask = torch.flip(valid_mask, dims=(-1,))
    if bool(torch.rand(()) < 0.5):
        values = torch.flip(values, dims=(-2,))
        valid_mask = torch.flip(valid_mask, dims=(-2,))
    return values, valid_mask


class LayeredBlockDataset(Dataset):
    """Dense square crops for unconditional multilayer city generation."""

    def __init__(
        self,
        config: LayeredDiffusionConfig,
        manifest: str | Path,
        *,
        augment: bool,
    ) -> None:
        self.config = config
        self.tiles = UrbanTileDataset(manifest, include_auxiliary=True)
        self.augment = augment
        self.crops: list[tuple[int, int, int]] = []

        if not self.tiles:
            return
        first = self.tiles[0]
        height, width = first["x"].shape[-2:]
        if config.crop_size_pixels > min(height, width):
            raise ValueError(
                f"Crop size {config.crop_size_pixels} exceeds tile dimensions {height}x{width}"
            )
        tops = _crop_positions(height, config.crop_size_pixels, config.crop_stride_pixels)
        lefts = _crop_positions(width, config.crop_size_pixels, config.crop_stride_pixels)

        for tile_index in range(len(self.tiles)):
            tile = self.tiles[tile_index]
            if "road_vertical_masks" not in tile or "rail_vertical_masks" not in tile:
                raise RuntimeError(
                    "The corpus lacks vertical transport arrays. Rebuild it with the current "
                    "dataset pipeline before training the layered model."
                )
            for top in tops:
                for left in lefts:
                    crop = (
                        slice(top, top + config.crop_size_pixels),
                        slice(left, left + config.crop_size_pixels),
                    )
                    vertical_pixels = (
                        tile["road_vertical_masks"][1:3, crop[0], crop[1]].sum()
                        + tile["rail_vertical_masks"][1:3, crop[0], crop[1]].sum()
                    )
                    repeat = (
                        config.vertical_crop_repeat
                        if augment and float(vertical_pixels) > 0
                        else 1
                    )
                    self.crops.extend([(tile_index, top, left)] * repeat)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, index: int) -> dict[str, Any]:
        tile_index, top, left = self.crops[index]
        tile = self.tiles[tile_index]
        size = self.config.crop_size_pixels
        rows = slice(top, top + size)
        columns = slice(left, left + size)
        values = layers_to_model_space(
            tile["x"][:, rows, columns],
            tile["road_vertical_masks"][:, rows, columns],
            tile["rail_vertical_masks"][:, rows, columns],
        )
        valid_mask = tile["valid_data_mask"][rows, columns].float().unsqueeze(0)

        if values.shape[-2:] != self.config.resolution:
            values = F.interpolate(
                values.unsqueeze(0),
                size=self.config.resolution,
                mode="nearest",
            )[0]
            valid_mask = F.interpolate(
                valid_mask.unsqueeze(0),
                size=self.config.resolution,
                mode="nearest",
            )[0]
        if self.augment:
            values, valid_mask = _augment(values, valid_mask)

        return {
            "x0": values,
            "valid_mask": valid_mask,
            "tile_id": f"{tile['tile_id']}:{top}:{left}",
        }


def check_layered_data(
    config: LayeredDiffusionConfig,
    *,
    samples_per_split: int = 4,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "layers": list(LAYER_NAMES),
        "resolution": list(config.resolution),
    }
    for split, manifest in (
        ("train", config.train_manifest),
        ("validation", config.validation_manifest),
    ):
        if not manifest.exists():
            raise FileNotFoundError(f"Missing {split} manifest: {manifest}")
        dataset = LayeredBlockDataset(config, manifest, augment=False)
        shapes: list[list[int]] = []
        coverage = torch.zeros(MODEL_CHANNELS, dtype=torch.float64)
        pixels = 0
        for index in range(min(len(dataset), samples_per_split)):
            sample = dataset[index]
            shapes.append(list(sample["x0"].shape))
            decoded = sample["x0"].add(1.0).div(2.0)
            coverage += decoded.sum(dim=(-2, -1), dtype=torch.float64)
            pixels += int(decoded.shape[-2] * decoded.shape[-1])
        result[split] = {
            "manifest": str(manifest),
            "samples": len(dataset),
            "sample_shapes": shapes,
            "inspected_coverage": {
                name: float(value / max(pixels, 1))
                for name, value in zip(LAYER_NAMES, coverage.tolist(), strict=True)
            },
        }
    return result
