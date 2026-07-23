from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .semantic_config import SEMANTIC_NAMES, SemanticDiffusionConfig
from .semantic_data import SemanticBlockDataset, model_space_to_semantic

_TOPOLOGY_CLASSES = ("road", "rail")
_NEIGHBOURS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _distribution(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _component_sizes(mask: np.ndarray) -> tuple[list[int], int]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    edge_components = 0

    for row, column in np.argwhere(mask):
        row = int(row)
        column = int(column)
        if visited[row, column]:
            continue
        queue = deque([(row, column)])
        visited[row, column] = True
        size = 0
        touches_edge = False
        while queue:
            current_row, current_column = queue.pop()
            size += 1
            touches_edge = touches_edge or current_row in {0, height - 1}
            touches_edge = touches_edge or current_column in {0, width - 1}
            for row_offset, column_offset in _NEIGHBOURS:
                neighbour_row = current_row + row_offset
                neighbour_column = current_column + column_offset
                if (
                    0 <= neighbour_row < height
                    and 0 <= neighbour_column < width
                    and mask[neighbour_row, neighbour_column]
                    and not visited[neighbour_row, neighbour_column]
                ):
                    visited[neighbour_row, neighbour_column] = True
                    queue.append((neighbour_row, neighbour_column))
        sizes.append(size)
        edge_components += int(touches_edge)
    return sizes, edge_components


def _run_lengths(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    for line in (*mask, *mask.T):
        padded = np.pad(line.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        runs.extend((ends - starts).tolist())
    return runs


def mask_topology_metrics(mask: np.ndarray) -> dict[str, float | int]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("Topology masks must be two-dimensional")

    foreground = int(binary.sum())
    sizes, edge_components = _component_sizes(binary)
    components = len(sizes)
    largest = max(sizes, default=0)
    tiny_pixels = sum(size for size in sizes if size <= 4)
    runs = _run_lengths(binary)

    return {
        "coverage": float(binary.mean()),
        "foreground_pixels": foreground,
        "component_count": components,
        "components_per_1000_pixels": float(components * 1000 / max(foreground, 1)),
        "largest_component_fraction": float(largest / max(foreground, 1)),
        "small_component_pixel_fraction": float(tiny_pixels / max(foreground, 1)),
        "mean_component_size": float(foreground / max(components, 1)),
        "edge_touching_component_fraction": float(edge_components / max(components, 1)),
        "mean_axis_run_length": float(np.mean(runs)) if runs else 0.0,
    }


class _Summary:
    def __init__(self) -> None:
        self.sample_count = 0
        self.coverage: dict[str, list[float]] = {name: [] for name in SEMANTIC_NAMES}
        self.topology: dict[str, dict[str, list[float]]] = {
            name: defaultdict(list) for name in _TOPOLOGY_CLASSES
        }

    def add(self, classes: np.ndarray) -> None:
        array = np.asarray(classes)
        if array.ndim != 2:
            raise ValueError("Semantic samples must be two-dimensional")
        if array.size and (array.min() < 0 or array.max() >= len(SEMANTIC_NAMES)):
            raise ValueError("Semantic sample contains an unknown class id")

        self.sample_count += 1
        for class_id, name in enumerate(SEMANTIC_NAMES):
            mask = array == class_id
            self.coverage[name].append(float(mask.mean()))
            if name in self.topology:
                for metric, value in mask_topology_metrics(mask).items():
                    self.topology[name][metric].append(float(value))

    def result(self) -> dict[str, Any]:
        return {
            "samples": self.sample_count,
            "coverage": {
                name: _distribution(values) for name, values in self.coverage.items()
            },
            "topology": {
                name: {
                    metric: _distribution(values)
                    for metric, values in metrics.items()
                }
                for name, metrics in self.topology.items()
            },
        }


def summarize_semantic_samples(samples: np.ndarray) -> dict[str, Any]:
    array = np.asarray(samples)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError("Expected semantic samples with shape [N, H, W]")
    summary = _Summary()
    for sample in array:
        summary.add(sample)
    return summary.result()


def _sample_indexes(length: int, limit: int | None) -> list[int]:
    if limit is None or limit <= 0 or limit >= length:
        return list(range(length))
    return sorted(
        {
            int(round(value))
            for value in np.linspace(0, length - 1, num=limit, dtype=np.float64)
        }
    )


def _real_summary(
    config: SemanticDiffusionConfig,
    *,
    split: str,
    max_samples: int | None,
) -> dict[str, Any]:
    if config.task != "block":
        raise ValueError("Topology evaluation of real crops requires a block config")
    manifest = {
        "train": config.train_manifest,
        "validation": config.validation_manifest,
    }.get(split)
    if manifest is None:
        raise ValueError("split must be train or validation")

    dataset = SemanticBlockDataset(config, manifest, augment=False)
    overall = _Summary()
    by_area: dict[str, _Summary] = defaultdict(_Summary)
    for index in _sample_indexes(len(dataset), max_samples):
        tile_index, _top, _left = dataset.crops[index]
        row = dataset.tiles.rows[tile_index]
        area_id = str(row.get("area_id") or "unknown")
        classes = model_space_to_semantic(dataset[index]["x0"]).cpu().numpy()
        overall.add(classes)
        by_area[area_id].add(classes)
    return {
        "split": split,
        "manifest": str(manifest),
        "available_samples": len(dataset),
        "evaluated_samples": overall.sample_count,
        "overall": overall.result(),
        "by_area": {
            area_id: summary.result()
            for area_id, summary in sorted(by_area.items())
        },
    }


def _generated_summary(paths: Iterable[str | Path]) -> dict[str, Any]:
    overall = _Summary()
    by_file: dict[str, Any] = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            path = path / "semantic-blocks.npy"
        if not path.exists():
            raise FileNotFoundError(f"Generated semantic file does not exist: {path}")
        samples = np.load(path, allow_pickle=False)
        summary = summarize_semantic_samples(samples)
        by_file[str(path)] = summary
        if samples.ndim == 2:
            samples = samples[None, ...]
        for sample in samples:
            overall.add(sample)
    return {
        "files": len(by_file),
        "overall": overall.result(),
        "by_file": by_file,
    }


def evaluate_semantic_topology(
    config: SemanticDiffusionConfig,
    generated: Iterable[str | Path],
    output: str | Path,
    *,
    split: str = "train",
    max_real_samples: int | None = 1000,
) -> dict[str, Any]:
    generated_paths = list(generated)
    if not generated_paths:
        raise ValueError("At least one generated semantic array is required")

    result = {
        "semantic_classes": list(SEMANTIC_NAMES),
        "connectivity": 8,
        "real": _real_summary(
            config,
            split=split,
            max_samples=max_real_samples,
        ),
        "generated": _generated_summary(generated_paths),
        "notes": {
            "component_count": "Eight-connected foreground components.",
            "largest_component_fraction": (
                "Fraction of class pixels contained in the largest component."
            ),
            "small_component_pixel_fraction": (
                "Fraction of class pixels in components containing four pixels or fewer."
            ),
            "mean_axis_run_length": (
                "Mean horizontal and vertical foreground run length in pixels."
            ),
        },
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output"] = str(destination)
    return result
