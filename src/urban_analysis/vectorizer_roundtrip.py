from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from urban_model.config import load_layered_diffusion_config
from urban_model.data import LayeredBlockDataset, model_space_to_layers
from urban_model.vectorize import (
    _connected_components,
    _require_image_tools,
    generated_layers_to_city_state,
)

from .generated_city_audit import _bounds, _graph_metrics, _transport_graph


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _skeleton_metrics(values) -> dict[str, float | int]:
    decoded = model_space_to_layers(values)
    surface = decoded["surface"].detach().cpu().numpy()
    road = (surface >= 3) & (surface <= 5)

    binary_closing, _distance, _label, disk, skeletonize = _require_image_tools()
    cleaned = binary_closing(road.astype(bool), structure=disk(1))
    skeleton = skeletonize(cleaned)
    active = {tuple(value) for value in np.argwhere(skeleton)}
    components = _connected_components(active)

    lengths = [len(component) for component in components]
    total = sum(lengths)
    height, width = road.shape

    def touches_boundary(component):
        return any(
            row <= 1 or row >= height - 2 or column <= 1 or column >= width - 2
            for row, column in component
        )

    interior = [
        length
        for component, length in zip(components, lengths, strict=True)
        if not touches_boundary(component)
    ]

    return {
        "skeleton_components": len(components),
        "skeleton_pixels": total,
        "skeleton_largest_fraction": max(lengths, default=0) / total if total else 0.0,
        "skeleton_interior_component_fraction": sum(interior) / total if total else 0.0,
    }


def _compiled_metrics(values, config, seed: int) -> dict[str, float | int]:
    state = generated_layers_to_city_state(
        values,
        bounds_m=(0.0, 0.0, 1024.0, 1024.0),
        max_height_m=config.max_height_m,
        max_surface_offset_m=config.max_surface_offset_m,
        max_underground_depth_m=config.max_underground_depth_m,
        max_elevated_height_m=config.max_elevated_height_m,
        road_max_grade=config.road_max_grade,
        rail_max_grade=config.rail_max_grade,
        auxiliary_threshold=config.auxiliary_threshold,
        minimum_component_pixels=config.minimum_vector_component_pixels,
        seed=seed,
    )
    graph = _transport_graph(state, "road", vertical="surface")
    metrics = _graph_metrics(graph, _bounds(state))
    return {
        "compiled_components": int(metrics["components"]),
        "compiled_total_length_m": float(metrics["total_length_m"]),
        "compiled_largest_fraction": float(metrics["largest_length_fraction"]),
        "compiled_interior_component_fraction": float(
            metrics["interior_component_length_fraction"]
        ),
        "compiled_dead_ends": int(metrics["interior_dead_ends"]),
    }


def run(config_path: str | Path, output: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists() and overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    config = load_layered_diffusion_config(config_path)
    dataset = LayeredBlockDataset(config, config.validation_manifest, augment=False)
    if not dataset:
        raise ValueError("Validation dataset is empty")

    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        values = sample["x0"]
        reference = _skeleton_metrics(values)
        compiled = _compiled_metrics(values, config, seed=20_000 + index)

        row = {
            "tile_id": str(sample["tile_id"]).split(":", 1)[0],
            **reference,
            **compiled,
        }
        row["component_inflation"] = (
            float(row["compiled_components"]) / max(float(row["skeleton_components"]), 1.0)
        )
        row["largest_fraction_retention"] = (
            float(row["compiled_largest_fraction"])
            / max(float(row["skeleton_largest_fraction"]), 1e-9)
        )
        row["interior_fraction_added"] = (
            float(row["compiled_interior_component_fraction"])
            - float(row["skeleton_interior_component_fraction"])
        )
        rows.append(row)

    with (output / "tiles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numeric = [key for key in rows[0] if key != "tile_id"]
    result = {
        "analysis_version": 1,
        "config": str(config_path),
        "validation_manifest": str(config.validation_manifest),
        "tiles": len(rows),
        "metrics": {
            key: _summary([float(row[key]) for row in rows])
            for key in numeric
        },
        "gate": {
            "purpose": (
                "Measure topology loss caused by raster-to-vector compilation on clean real "
                "raster inputs. Skeleton and compiled graph are derived from the same road mask."
            ),
            "good_direction": (
                "component_inflation approaches 1, largest_fraction_retention approaches 1, "
                "and interior_fraction_added approaches 0 without joining separate roads."
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit clean-raster vectorizer topology fidelity")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args.config, args.output, overwrite=args.overwrite)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
