from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .utils import stable_int, write_json


def _load_rows(dataset_root: Path, manifest_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index_path in sorted(dataset_root.glob("*/index.csv")):
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sample = index_path.parent / "tiles" / row["tile_id"] / "layers.npz"
                metadata = index_path.parent / "tiles" / row["tile_id"] / "metadata.json"
                row["dataset_dir"] = Path(os.path.relpath(index_path.parent, manifest_root)).as_posix()
                row["sample_path"] = Path(os.path.relpath(sample, manifest_root)).as_posix()
                row["metadata_path"] = Path(os.path.relpath(metadata, manifest_root)).as_posix()
                rows.append(row)
    return rows


def _target_group_counts(group_count: int, ratios: dict[str, float]) -> dict[str, int]:
    splits = ["train", "validation", "test"]
    raw = {split: max(0.0, float(ratios.get(split, 0.0))) * group_count for split in splits}
    counts = {split: math.floor(raw[split]) for split in splits}
    remainder = group_count - sum(counts.values())
    for split in sorted(
        splits,
        key=lambda name: (raw[name] - counts[name], ratios.get(name, 0)),
        reverse=True,
    ):
        if remainder <= 0:
            break
        counts[split] += 1
        remainder -= 1

    positive = [split for split in splits if float(ratios.get(split, 0.0)) > 0]
    if group_count >= len(positive):
        for split in positive:
            if counts[split] == 0:
                donor = max(
                    (candidate for candidate in splits if counts[candidate] > 1),
                    key=lambda candidate: counts[candidate],
                    default=None,
                )
                if donor is not None:
                    counts[donor] -= 1
                    counts[split] += 1
    return counts


def build_manifests_from_values(
    dataset_root: str | Path,
    manifest_root: str | Path,
    *,
    seed: int = 5132,
    split_ratios: dict[str, float] | None = None,
    validation_cities: set[str] | None = None,
    test_cities: set[str] | None = None,
    spatial_group_tiles: int = 4,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    manifest_root = Path(manifest_root).expanduser().resolve()
    manifest_root.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(dataset_root, manifest_root)
    if not rows:
        raise FileNotFoundError(f"No city index.csv files found below {dataset_root}")

    ratios = split_ratios or {"train": 0.8, "validation": 0.1, "test": 0.1}
    ratio_sum = sum(float(ratios.get(name, 0.0)) for name in ["train", "validation", "test"])
    if not 0.999 <= ratio_sum <= 1.001:
        raise ValueError("split_ratios must sum to 1")
    validation_cities = validation_cities or set()
    test_cities = test_cities or set()
    group_tiles = max(1, int(spatial_group_tiles))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        city = row["city_id"]
        if city in validation_cities:
            group = f"forced:validation:{city}"
        elif city in test_cities:
            group = f"forced:test:{city}"
        else:
            group_x = math.floor(int(row["column"]) / group_tiles)
            group_y = math.floor(int(row["row"]) / group_tiles)
            group = f"spatial:{city}:{group_x}:{group_y}"
        groups[group].append(row)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    free_groups: list[tuple[str, list[dict[str, Any]]]] = []
    for group, members in groups.items():
        if group.startswith("forced:validation:"):
            target = "validation"
        elif group.startswith("forced:test:"):
            target = "test"
        else:
            free_groups.append((group, members))
            continue
        for row in members:
            copy = dict(row)
            copy["split"] = target
            copy["spatial_group"] = group
            split_rows[target].append(copy)

    free_groups.sort(key=lambda item: stable_int(item[0], seed))
    group_targets = _target_group_counts(len(free_groups), ratios)
    assignment_order: list[str] = []
    remaining = dict(group_targets)
    while sum(remaining.values()) > 0:
        for split in ["train", "validation", "test"]:
            if remaining[split] > 0:
                assignment_order.append(split)
                remaining[split] -= 1

    for (group, members), split in zip(free_groups, assignment_order, strict=True):
        for row in members:
            copy = dict(row)
            copy["split"] = split
            copy["spatial_group"] = group
            split_rows[split].append(copy)

    for split, split_data in split_rows.items():
        path = manifest_root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in sorted(split_data, key=lambda value: value["tile_id"]):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    warnings: list[str] = []
    for split in ["validation", "test"]:
        if float(ratios.get(split, 0)) > 0 and not split_rows[split]:
            warnings.append(
                f"{split} is empty because the dataset has too few independent spatial groups; "
                "reduce spatial_group_tiles or add more cities."
            )
    summary: dict[str, Any] = {split: len(values) for split, values in split_rows.items()}
    summary["total"] = sum(summary.values())
    summary["spatial_groups"] = len(groups)
    summary["free_group_targets"] = group_targets
    summary["sample_paths_are_relative_to_manifest_directory"] = True
    summary["warnings"] = warnings
    write_json(manifest_root / "manifest_summary.json", summary)
    return summary


def build_manifests(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    dataset_root = Path(config.get("dataset_root", "data/processed"))
    manifest_root = Path(config.get("manifest_root", "data/manifests"))
    if not dataset_root.is_absolute():
        dataset_root = (config_path.parent.parent / dataset_root).resolve()
    if not manifest_root.is_absolute():
        manifest_root = (config_path.parent.parent / manifest_root).resolve()

    return build_manifests_from_values(
        dataset_root,
        manifest_root,
        seed=int(config.get("seed", 5132)),
        split_ratios=config.get("split_ratios", {"train": 0.8, "validation": 0.1, "test": 0.1}),
        validation_cities=set(config.get("city_splits", {}).get("validation", [])),
        test_cities=set(config.get("city_splits", {}).get("test", [])),
        spatial_group_tiles=int(config.get("spatial_group_tiles", 4)),
    )
