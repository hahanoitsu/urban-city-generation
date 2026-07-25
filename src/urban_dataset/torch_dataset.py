from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


class UrbanTileDataset:
    """PyTorch-compatible loader for tensors and their supervision masks."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        include_auxiliary: bool = True,
        append_height_known_mask: bool = False,
        transform: Callable | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.include_auxiliary = include_auxiliary
        self.append_height_known_mask = append_height_known_mask
        self.transform = transform
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        return path.resolve()

    def __getitem__(self, index: int):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the project with the 'ml' extra to use this class") from exc
        row = self.rows[index]
        with np.load(self._resolve(row["sample_path"]), allow_pickle=False) as archive:
            layers = archive["layers"].astype(np.float32)
            known = archive["height_known_mask"].astype(np.float32)
            if self.append_height_known_mask:
                layers = np.concatenate([layers, known[None, ...]], axis=0)
            result = {
                "x": torch.from_numpy(layers),
                "tile_id": row["tile_id"],
                "metadata": row,
            }
            if self.include_auxiliary:
                required = {
                    "height_confidence": np.int64,
                    "landuse_known_mask": np.float32,
                    "road_centerlines": np.float32,
                    "valid_data_mask": np.float32,
                }
                for name, dtype in required.items():
                    result[name] = torch.from_numpy(archive[name].astype(dtype))
                optional = {
                    "road_vertical_masks": np.float32,
                    "rail_vertical_masks": np.float32,
                    "road_vertical_profiles_m": np.float32,
                    "rail_vertical_profiles_m": np.float32,
                    "road_vertical_profile_confidence": np.float32,
                    "rail_vertical_profile_confidence": np.float32,
                    "surface_transport_reservation": np.float32,
                    "buildable_surface_mask": np.float32,
                    "buildability_known_mask": np.float32,
                }
                for name, dtype in optional.items():
                    if name in archive:
                        result[name] = torch.from_numpy(archive[name].astype(dtype))
        if self.transform is not None:
            result = self.transform(result)
        return result
