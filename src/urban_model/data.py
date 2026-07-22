from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from urban_dataset.torch_dataset import UrbanTileDataset

ROAD_SOURCE_CHANNELS = (1, 2, 3)
LANDUSE_SOURCE_CHANNELS = (4, 5, 6, 7, 10)
BINARY_SOURCE_CHANNELS = (0, 8, 11)
HEIGHT_SOURCE_CHANNEL = 9


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

        road_values = layers[list(ROAD_SOURCE_CHANNELS)]
        road_target = road_values.argmax(dim=0).long() + 1
        road_target[road_values.amax(dim=0) <= 0.05] = 0

        landuse_values = layers[list(LANDUSE_SOURCE_CHANNELS)]
        landuse_target = landuse_values.argmax(dim=0).long() + 1
        landuse_target[landuse_values.amax(dim=0) <= 0.05] = 0

        return {
            "input": layers,
            "road_target": road_target,
            "landuse_target": landuse_target,
            "binary_target": layers[list(BINARY_SOURCE_CHANNELS)],
            "height_target": layers[HEIGHT_SOURCE_CHANNEL : HEIGHT_SOURCE_CHANNEL + 1],
            "centerline_target": sample["road_centerlines"],
            "height_confidence": sample["height_confidence"],
            "landuse_known_mask": sample["landuse_known_mask"],
            "valid_mask": sample["valid_data_mask"],
            "tile_id": sample["tile_id"],
        }
