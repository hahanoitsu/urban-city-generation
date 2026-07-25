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
PROFILE_NAMES = (
    "road_surface_offset",
    "road_underground_depth",
    "road_elevated_height",
    "rail_surface_offset",
    "rail_underground_depth",
    "rail_elevated_height",
)
LAYER_NAMES = SURFACE_NAMES + AUXILIARY_NAMES + PROFILE_NAMES
SURFACE_CLASS_COUNT = len(SURFACE_NAMES)
MODEL_CHANNELS = len(LAYER_NAMES)


def _crop_positions(length: int, crop_size: int, stride: int) -> list[int]:
    positions = list(range(0, length - crop_size + 1, stride))
    last = length - crop_size
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _surface_classes(layers: torch.Tensor) -> torch.Tensor:
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
    return classes


def layers_to_model_space(
    layers: torch.Tensor,
    road_vertical_masks: torch.Tensor,
    rail_vertical_masks: torch.Tensor,
    road_vertical_profiles_m: torch.Tensor,
    rail_vertical_profiles_m: torch.Tensor,
    config: LayeredDiffusionConfig,
) -> torch.Tensor:
    if layers.shape[-3] != 12:
        raise ValueError(f"Expected 12 source channels, found {layers.shape[-3]}")
    if road_vertical_masks.shape[-3] != 4 or rail_vertical_masks.shape[-3] != 4:
        raise ValueError("Vertical transport masks must have shape [4,H,W]")
    if road_vertical_profiles_m.shape[-3] != 3 or rail_vertical_profiles_m.shape[-3] != 3:
        raise ValueError("Vertical transport profiles must have shape [3,H,W]")

    classes = _surface_classes(layers)
    surface = F.one_hot(classes, num_classes=SURFACE_CLASS_COUNT).movedim(-1, -3).float()
    masks_and_height = torch.stack(
        [
            road_vertical_masks[..., 1, :, :],
            road_vertical_masks[..., 2, :, :],
            rail_vertical_masks[..., 1, :, :],
            rail_vertical_masks[..., 2, :, :],
            layers[..., 9, :, :] * (classes == 2).float(),
        ],
        dim=-3,
    ).float()

    profiles = torch.stack(
        [
            (road_vertical_profiles_m[..., 0, :, :] / config.max_surface_offset_m).clamp(
                -1.0, 1.0
            ),
            (-road_vertical_profiles_m[..., 1, :, :] / config.max_underground_depth_m)
            .clamp(0.0, 1.0)
            .mul(2.0)
            .sub(1.0),
            (road_vertical_profiles_m[..., 2, :, :] / config.max_elevated_height_m)
            .clamp(0.0, 1.0)
            .mul(2.0)
            .sub(1.0),
            (rail_vertical_profiles_m[..., 0, :, :] / config.max_surface_offset_m).clamp(
                -1.0, 1.0
            ),
            (-rail_vertical_profiles_m[..., 1, :, :] / config.max_underground_depth_m)
            .clamp(0.0, 1.0)
            .mul(2.0)
            .sub(1.0),
            (rail_vertical_profiles_m[..., 2, :, :] / config.max_elevated_height_m)
            .clamp(0.0, 1.0)
            .mul(2.0)
            .sub(1.0),
        ],
        dim=-3,
    )
    return torch.cat(
        [surface.mul(2.0).sub(1.0), masks_and_height.mul(2.0).sub(1.0), profiles],
        dim=-3,
    )


def model_space_to_layers(
    values: torch.Tensor,
    *,
    auxiliary_threshold: float = 0.35,
    max_surface_offset_m: float = 12.0,
    max_underground_depth_m: float = 40.0,
    max_elevated_height_m: float = 30.0,
) -> dict[str, torch.Tensor]:
    if values.shape[-3] != MODEL_CHANNELS:
        raise ValueError(f"Expected {MODEL_CHANNELS} model channels, found {values.shape[-3]}")
    surface = values[..., :SURFACE_CLASS_COUNT, :, :].argmax(dim=-3)
    auxiliary = values[..., SURFACE_CLASS_COUNT : SURFACE_CLASS_COUNT + 5, :, :]
    profiles = values[..., SURFACE_CLASS_COUNT + 5 :, :, :]
    road_underground = auxiliary[..., 0, :, :] > auxiliary_threshold
    road_elevated = auxiliary[..., 1, :, :] > auxiliary_threshold
    rail_underground = auxiliary[..., 2, :, :] > auxiliary_threshold
    rail_elevated = auxiliary[..., 3, :, :] > auxiliary_threshold
    building = surface == 2
    road_surface = (surface >= 3) & (surface <= 5)
    rail_surface = surface == 6
    return {
        "surface": surface,
        "road_underground": road_underground,
        "road_elevated": road_elevated,
        "rail_underground": rail_underground,
        "rail_elevated": rail_elevated,
        "building_height": auxiliary[..., 4, :, :].add(1.0).div(2.0).clamp(0.0, 1.0)
        * building.float(),
        "road_surface_offset_m": profiles[..., 0, :, :].clamp(-1.0, 1.0)
        * max_surface_offset_m
        * road_surface.float(),
        "road_underground_depth_m": profiles[..., 1, :, :]
        .add(1.0)
        .div(2.0)
        .clamp(0.0, 1.0)
        * max_underground_depth_m
        * road_underground.float(),
        "road_elevated_height_m": profiles[..., 2, :, :]
        .add(1.0)
        .div(2.0)
        .clamp(0.0, 1.0)
        * max_elevated_height_m
        * road_elevated.float(),
        "rail_surface_offset_m": profiles[..., 3, :, :].clamp(-1.0, 1.0)
        * max_surface_offset_m
        * rail_surface.float(),
        "rail_underground_depth_m": profiles[..., 4, :, :]
        .add(1.0)
        .div(2.0)
        .clamp(0.0, 1.0)
        * max_underground_depth_m
        * rail_underground.float(),
        "rail_elevated_height_m": profiles[..., 5, :, :]
        .add(1.0)
        .div(2.0)
        .clamp(0.0, 1.0)
        * max_elevated_height_m
        * rail_elevated.float(),
    }


def _confidence_weight(values: torch.Tensor) -> torch.Tensor:
    return torch.where(
        values >= 3,
        torch.ones_like(values),
        torch.where(
            values >= 2,
            torch.full_like(values, 0.8),
            torch.where(values >= 1, torch.full_like(values, 0.5), torch.full_like(values, 0.25)),
        ),
    )


def _supervision_mask(
    values: torch.Tensor,
    layers: torch.Tensor,
    road_vertical_masks: torch.Tensor,
    rail_vertical_masks: torch.Tensor,
    road_profile_confidence: torch.Tensor,
    rail_profile_confidence: torch.Tensor,
    height_confidence: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    height, width = valid_mask.shape
    result = torch.zeros((MODEL_CHANNELS, height, width), dtype=torch.float32)
    result[:12] = valid_mask.unsqueeze(0)
    building = layers[8] > 0.05
    result[12] = valid_mask * building.float() * _confidence_weight(height_confidence.float())

    road_surface = road_vertical_masks[0] > 0.5
    rail_surface = rail_vertical_masks[0] > 0.5
    result[13] = valid_mask * road_surface.float() * _confidence_weight(
        road_profile_confidence[0]
    )
    result[14] = valid_mask * (road_vertical_masks[1] > 0.5).float() * _confidence_weight(
        road_profile_confidence[1]
    )
    result[15] = valid_mask * (road_vertical_masks[2] > 0.5).float() * _confidence_weight(
        road_profile_confidence[2]
    )
    result[16] = valid_mask * rail_surface.float() * _confidence_weight(
        rail_profile_confidence[0]
    )
    result[17] = valid_mask * (rail_vertical_masks[1] > 0.5).float() * _confidence_weight(
        rail_profile_confidence[1]
    )
    result[18] = valid_mask * (rail_vertical_masks[2] > 0.5).float() * _confidence_weight(
        rail_profile_confidence[2]
    )
    return result


def _augment(values: torch.Tensor, supervision: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    turns = int(torch.randint(0, 4, ()).item())
    if turns:
        values = torch.rot90(values, turns, dims=(-2, -1))
        supervision = torch.rot90(supervision, turns, dims=(-2, -1))
    if bool(torch.rand(()) < 0.5):
        values = torch.flip(values, dims=(-1,))
        supervision = torch.flip(supervision, dims=(-1,))
    if bool(torch.rand(()) < 0.5):
        values = torch.flip(values, dims=(-2,))
        supervision = torch.flip(supervision, dims=(-2,))
    return values, supervision


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
            required = (
                "road_vertical_masks",
                "rail_vertical_masks",
                "road_vertical_profiles_m",
                "rail_vertical_profiles_m",
                "road_vertical_profile_confidence",
                "rail_vertical_profile_confidence",
            )
            if any(name not in tile for name in required):
                raise RuntimeError(
                    "The corpus lacks continuous vertical profiles. Re-run prepare-city and "
                    "build-corpus with the current dataset pipeline before training."
                )
            for top in tops:
                for left in lefts:
                    rows = slice(top, top + config.crop_size_pixels)
                    columns = slice(left, left + config.crop_size_pixels)
                    vertical_pixels = (
                        tile["road_vertical_masks"][1:3, rows, columns].sum()
                        + tile["rail_vertical_masks"][1:3, rows, columns].sum()
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
        layers = tile["x"][:, rows, columns]
        road_masks = tile["road_vertical_masks"][:, rows, columns]
        rail_masks = tile["rail_vertical_masks"][:, rows, columns]
        road_profiles = tile["road_vertical_profiles_m"][:, rows, columns]
        rail_profiles = tile["rail_vertical_profiles_m"][:, rows, columns]
        road_confidence = tile["road_vertical_profile_confidence"][:, rows, columns]
        rail_confidence = tile["rail_vertical_profile_confidence"][:, rows, columns]
        height_confidence = tile["height_confidence"][rows, columns]
        valid_mask = tile["valid_data_mask"][rows, columns].float()

        values = layers_to_model_space(
            layers,
            road_masks,
            rail_masks,
            road_profiles,
            rail_profiles,
            self.config,
        )
        supervision = _supervision_mask(
            values,
            layers,
            road_masks,
            rail_masks,
            road_confidence,
            rail_confidence,
            height_confidence,
            valid_mask,
        )

        if values.shape[-2:] != self.config.resolution:
            values = F.interpolate(
                values.unsqueeze(0), size=self.config.resolution, mode="nearest"
            )[0]
            supervision = F.interpolate(
                supervision.unsqueeze(0), size=self.config.resolution, mode="nearest"
            )[0]
        if self.augment:
            values, supervision = _augment(values, supervision)

        return {
            "x0": values,
            # The training loop historically calls this valid_mask. It is now a
            # per-channel confidence-aware supervision mask.
            "valid_mask": supervision,
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
        supervision = torch.zeros(MODEL_CHANNELS, dtype=torch.float64)
        pixels = 0
        for index in range(min(len(dataset), samples_per_split)):
            sample = dataset[index]
            shapes.append(list(sample["x0"].shape))
            decoded = sample["x0"].add(1.0).div(2.0)
            coverage += decoded.sum(dim=(-2, -1), dtype=torch.float64)
            supervision += sample["valid_mask"].sum(dim=(-2, -1), dtype=torch.float64)
            pixels += int(decoded.shape[-2] * decoded.shape[-1])
        result[split] = {
            "manifest": str(manifest),
            "samples": len(dataset),
            "sample_shapes": shapes,
            "inspected_coverage": {
                name: float(value / max(pixels, 1))
                for name, value in zip(LAYER_NAMES, coverage.tolist(), strict=True)
            },
            "supervised_fraction": {
                name: float(value / max(pixels, 1))
                for name, value in zip(LAYER_NAMES, supervision.tolist(), strict=True)
            },
        }
    return result
