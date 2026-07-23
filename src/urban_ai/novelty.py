from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _transform(x: int, y: int, maximum: int, transform: int) -> tuple[int, int]:
    if transform >= 4:
        x = maximum - x
        transform -= 4
    for _ in range(transform):
        x, y = maximum - y, x
    return x, y


def edge_signature(program: dict[str, Any], transform: int = 0) -> set[tuple[Any, ...]]:
    bins = int(program.get("program_config", {}).get("coordinate_bins", 256))
    maximum = bins - 1
    positions: list[tuple[int, int]] = []
    signatures: set[tuple[Any, ...]] = set()
    for command in program.get("commands", []):
        op = command.get("op")
        if op in {"root", "add"}:
            positions.append(
                _transform(
                    int(command["x_bin"]),
                    int(command["y_bin"]),
                    maximum,
                    transform,
                )
            )
        if op == "add":
            left, right = int(command["parent"]), int(command["node"])
        elif op == "connect":
            left, right = int(command["from"]), int(command["to"])
        else:
            continue
        endpoints = tuple(sorted((positions[left], positions[right])))
        signatures.add(
            (
                endpoints,
                command.get("transport_mode"),
                command.get("class"),
                command.get("vertical_mode"),
                int(command.get("width_bin", 0)),
            )
        )
    return signatures


def edge_jaccard(left: set[tuple[Any, ...]], right: set[tuple[Any, ...]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def nearest_training_overlap(
    generated: dict[str, Any],
    program_root: str | Path,
) -> dict[str, Any]:
    program_root = Path(program_root).expanduser().resolve()
    index_path = program_root / "train.jsonl"
    generated_signatures = [edge_signature(generated, transform) for transform in range(8)]
    best = {"tile_id": None, "edge_jaccard": 0.0}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        program = json.loads(
            (program_root / row["program_path"]).read_text(encoding="utf-8")
        )
        reference = edge_signature(program)
        score = max(edge_jaccard(candidate, reference) for candidate in generated_signatures)
        if score > best["edge_jaccard"]:
            best = {"tile_id": row.get("tile_id"), "edge_jaccard": score}
    best["exact_copy_warning"] = bool(best["edge_jaccard"] >= 0.95)
    return best
