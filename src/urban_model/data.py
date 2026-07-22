from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from urban_dataset.torch_dataset import UrbanTileDataset

ROAD_SOURCE_CHANNELS = (1, 2, 3)
LANDUSE_SOURCE_CHANNELS = (4, 5, 6, 7, 10)
BINARY_SOURCE_CHANNELS = (0, 8, 11)
HEIGHT_SOURCE_CHANNEL = 9
DIRECTIONS = ("east", "west", "north", "south")


class RandomOrientation:
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        turns = int(torch.randint(0, 4, ()).item())
        keys = [
            "x",
            "height_confidence",
            "landuse_known_mask",
            "road_centerlines",
            "valid_data_mask",
        ]
        if turns:
            for key in keys:
                sample[key] = torch.rot90(sample[key], turns, dims=(-2, -1))
        if bool(torch.rand(()) < 0.5):
            for key in keys:
                sample[key] = torch.flip(sample[key], dims=(-1,))
        return sample


def _targets(layers: torch.Tensor) -> dict[str, torch.Tensor]:
    road_values = layers[list(ROAD_SOURCE_CHANNELS)]
    road_target = road_values.argmax(dim=0).long() + 1
    road_target[road_values.amax(dim=0) <= 0.05] = 0

    landuse_values = layers[list(LANDUSE_SOURCE_CHANNELS)]
    landuse_target = landuse_values.argmax(dim=0).long() + 1
    landuse_target[landuse_values.amax(dim=0) <= 0.05] = 0

    return {
        "road_target": road_target,
        "landuse_target": landuse_target,
        "binary_target": layers[list(BINARY_SOURCE_CHANNELS)],
        "height_target": layers[HEIGHT_SOURCE_CHANNEL : HEIGHT_SOURCE_CHANNEL + 1],
    }


class ReconstructionDataset(Dataset):
    def __init__(self, manifest: str | Path, *, augment: bool = False) -> None:
        transform = RandomOrientation() if augment else None
        self.tiles = UrbanTileDataset(manifest, include_auxiliary=True, transform=transform)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.tiles[index]
        layers = sample["x"]
        if layers.shape[0] != 12:
            raise ValueError(f"Expected 12 channels, found {layers.shape[0]}")

        return {
            "input": layers,
            **_targets(layers),
            "centerline_target": sample["road_centerlines"],
            "height_confidence": sample["height_confidence"],
            "landuse_known_mask": sample["landuse_known_mask"],
            "valid_mask": sample["valid_data_mask"],
            "tile_id": sample["tile_id"],
        }


def _grid_key(row: dict[str, Any], column: int, grid_row: int) -> tuple[str, str, int, int]:
    return (
        str(row.get("city_id", "")),
        str(row.get("area_id", "")),
        column,
        grid_row,
    )


def _neighbour(column: int, row: int, direction: str) -> tuple[int, int]:
    if direction == "east":
        return column + 1, row
    if direction == "west":
        return column - 1, row
    if direction == "north":
        return column, row + 1
    if direction == "south":
        return column, row - 1
    raise ValueError(f"Unknown extension direction: {direction}")


def _turns_to_east(direction: str) -> int:
    return {"east": 0, "west": 2, "north": 3, "south": 1}[direction]


def _rotate_sample(sample: dict[str, Any], turns: int) -> dict[str, Any]:
    if not turns:
        return sample
    result = dict(sample)
    for key in [
        "x",
        "height_confidence",
        "landuse_known_mask",
        "road_centerlines",
        "valid_data_mask",
    ]:
        result[key] = torch.rot90(sample[key], turns, dims=(-2, -1))
    return result


def make_boundary_guide(
    centerlines: torch.Tensor,
    *,
    boundary_width: int,
    guide_length: int,
) -> torch.Tensor:
    if centerlines.ndim != 3 or centerlines.shape[0] != 3:
        raise ValueError("Road centre-lines must have shape [3, H, W]")
    height, width = centerlines.shape[-2:]
    boundary_width = max(1, min(int(boundary_width), width))
    guide_length = max(1, min(int(guide_length), width))

    guide = torch.zeros((3, height, width * 2), dtype=centerlines.dtype)
    guide[:, :, width - boundary_width : width] = centerlines[:, :, -boundary_width:]
    crossings = centerlines[:, :, -boundary_width:].amax(dim=-1) > 0.5
    guide[:, :, width : width + guide_length] = crossings.unsqueeze(-1).to(guide.dtype)
    return guide


class ExtensionDataset(Dataset):
    """Pairs neighbouring tiles and rotates every example into an eastward extension."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        augment: bool = False,
        directions: Iterable[str] = DIRECTIONS,
        boundary_width: int = 3,
        guide_length: int = 12,
    ) -> None:
        self.tiles = UrbanTileDataset(manifest, include_auxiliary=True)
        self.augment = augment
        self.boundary_width = int(boundary_width)
        self.guide_length = int(guide_length)

        selected_directions = tuple(str(value).lower() for value in directions)
        if not selected_directions or any(value not in DIRECTIONS for value in selected_directions):
            raise ValueError(f"directions must be chosen from {DIRECTIONS}")

        locations: dict[tuple[str, str, int, int], int] = {}
        for index, row in enumerate(self.tiles.rows):
            column = int(row["column"])
            grid_row = int(row["row"])
            locations[_grid_key(row, column, grid_row)] = index

        pairs: list[tuple[int, int, str]] = []
        for seed_index, row in enumerate(self.tiles.rows):
            column = int(row["column"])
            grid_row = int(row["row"])
            for direction in selected_directions:
                neighbour_column, neighbour_row = _neighbour(column, grid_row, direction)
                target_index = locations.get(_grid_key(row, neighbour_column, neighbour_row))
                if target_index is not None:
                    pairs.append((seed_index, target_index, direction))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        seed_index, target_index, direction = self.pairs[index]
        turns = _turns_to_east(direction)
        seed = _rotate_sample(self.tiles[seed_index], turns)
        target = _rotate_sample(self.tiles[target_index], turns)

        seed_layers = seed["x"]
        target_layers = target["x"]
        if seed_layers.shape != target_layers.shape or seed_layers.shape[0] != 12:
            raise ValueError("Extension pairs require matching twelve-channel tiles")

        height, width = seed_layers.shape[-2:]
        full_layers = torch.cat([seed_layers, target_layers], dim=-1)
        known_mask = torch.zeros((1, height, width * 2), dtype=seed_layers.dtype)
        known_mask[:, :, :width] = 1
        known_layers = full_layers * known_mask
        guide = make_boundary_guide(
            seed["road_centerlines"],
            boundary_width=self.boundary_width,
            guide_length=self.guide_length,
        )

        height_confidence = torch.cat(
            [seed["height_confidence"], target["height_confidence"]], dim=-1
        )
        landuse_known = torch.cat(
            [seed["landuse_known_mask"], target["landuse_known_mask"]], dim=-1
        )
        centerline_target = torch.cat(
            [seed["road_centerlines"], target["road_centerlines"]], dim=-1
        )
        valid_mask = torch.zeros((height, width * 2), dtype=seed_layers.dtype)
        valid_mask[:, width:] = target["valid_data_mask"]

        if self.augment and bool(torch.rand(()) < 0.5):
            full_layers = torch.flip(full_layers, dims=(-2,))
            known_mask = torch.flip(known_mask, dims=(-2,))
            known_layers = torch.flip(known_layers, dims=(-2,))
            guide = torch.flip(guide, dims=(-2,))
            height_confidence = torch.flip(height_confidence, dims=(-2,))
            landuse_known = torch.flip(landuse_known, dims=(-2,))
            centerline_target = torch.flip(centerline_target, dims=(-2,))
            valid_mask = torch.flip(valid_mask, dims=(-2,))

        return {
            "input": torch.cat([known_layers, known_mask, guide], dim=0),
            **_targets(full_layers),
            "centerline_target": centerline_target,
            "height_confidence": height_confidence,
            "landuse_known_mask": landuse_known,
            "valid_mask": valid_mask,
            "known_layers": known_layers,
            "known_mask": known_mask,
            "boundary_guide": guide,
            "guide_length": self.guide_length,
            "full_target": full_layers,
            "tile_id": f"{seed['tile_id']}->{target['tile_id']}",
            "seed_tile_id": seed["tile_id"],
            "target_tile_id": target["tile_id"],
            "direction": direction,
        }


def dataset_for_task(
    task: str,
    manifest: str | Path,
    *,
    augment: bool,
    directions: Iterable[str] = DIRECTIONS,
    boundary_width: int = 3,
    guide_length: int = 12,
) -> Dataset:
    if task == "reconstruction":
        return ReconstructionDataset(manifest, augment=augment)
    if task == "extension":
        return ExtensionDataset(
            manifest,
            augment=augment,
            directions=directions,
            boundary_width=boundary_width,
            guide_length=guide_length,
        )
    raise ValueError(f"Unknown training task: {task}")
