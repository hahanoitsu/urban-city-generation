from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .codec import FIELDS, CommandCodecConfig, encode_program
from .schema import style_vector


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _transform_coordinate(x: int, y: int, maximum: int, transform: int) -> tuple[int, int]:
    if transform >= 4:
        x = maximum - x
        transform -= 4
    for _ in range(transform):
        x, y = maximum - y, x
    return x, y


def transform_program(program: dict[str, Any], transform: int) -> dict[str, Any]:
    if not 0 <= transform < 8:
        raise ValueError("transform must be between 0 and 7")
    copied = json.loads(json.dumps(program))
    bins = int(copied.get("program_config", {}).get("coordinate_bins", 256))
    maximum = bins - 1
    for command in copied.get("commands", []):
        if command.get("op") not in {"root", "add"}:
            continue
        command["x_bin"], command["y_bin"] = _transform_coordinate(
            int(command["x_bin"]), int(command["y_bin"]), maximum, transform
        )
    return copied


class GraphProgramDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        codec: CommandCodecConfig,
        *,
        style_mean: list[float],
        style_std: list[float],
        augment: bool = False,
    ) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        self.root = self.index_path.parent
        self.rows = _read_jsonl(self.index_path)
        if not self.rows:
            raise ValueError(f"Program index is empty: {self.index_path}")
        self.codec = codec
        self.style_mean = torch.tensor(style_mean, dtype=torch.float32)
        self.style_std = torch.tensor(style_std, dtype=torch.float32)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        program = json.loads((self.root / row["program_path"]).read_text(encoding="utf-8"))
        if self.augment:
            program = transform_program(program, random.randrange(8))
        encoded = encode_program(program, self.codec)
        style = torch.tensor(style_vector(program.get("style", {})), dtype=torch.float32)
        style = (style - self.style_mean) / self.style_std
        result = {field: torch.tensor(encoded[field], dtype=torch.long) for field in FIELDS}
        result["style"] = style
        result["tile_id"] = str(row.get("tile_id", index))
        return result


def collate_graph_programs(samples: list[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(int(sample["op"].shape[0]) for sample in samples)
    batch: dict[str, Any] = {}
    for field in FIELDS:
        values = torch.zeros((len(samples), maximum), dtype=torch.long)
        for row, sample in enumerate(samples):
            length = int(sample[field].shape[0])
            values[row, :length] = sample[field]
        batch[field] = values
    batch["style"] = torch.stack([sample["style"] for sample in samples])
    batch["tile_ids"] = [sample["tile_id"] for sample in samples]
    return batch
