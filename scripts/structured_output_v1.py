from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from urban_analysis.surface_roundtrip import audit_classes, write_state
from urban_model.config import LayeredDiffusionConfig
from urban_model.data import LayeredBlockDataset, SURFACE_CLASS_COUNT
from urban_model.morphology_control import (
    CONTROLS,
    build_model,
    normalise,
    sample,
)
from urban_model.surface_distribution_v2 import PALETTE, _classes


def _summary(rows: list[dict]) -> dict:
    result = {}
    if not rows:
        return result
    for key in rows[0]:
        if key in {"source", "sample_id", "scenario", "seed"}:
            continue
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float))
            and math.isfinite(float(row[key]))
        ]
        if values:
            result[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p10": float(np.quantile(values, 0.10)),
                "p90": float(np.quantile(values, 0.90)),
            }
    return result


def _compiled_map(masks: dict[str, np.ndarray]) -> np.ndarray:
    result = np.zeros(next(iter(masks.values())).shape, dtype=np.uint8)
    result[masks["vegetation"]] = 1
    result[masks["building"]] = 2
    result[masks["road_local"]] = 5
    result[masks["road_secondary"]] = 4
    result[masks["road_major"]] = 3
    result[masks["rail"]] = 6
    result[masks["water"]] = 7
    return result


def _save_pair(source: np.ndarray, compiled: np.ndarray, path: Path, label: str) -> None:
    left = Image.fromarray(PALETTE[source.astype(np.int64)])
    right = Image.fromarray(PALETTE[compiled.astype(np.int64)])
    width, height = left.size
    canvas = Image.new("RGB", (width * 2, height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), f"{label} - generated raster", fill="black")
    draw.text((width + 5, 5), "compiled vectors rerasterized", fill="black")
    canvas.paste(left, (0, 24))
    canvas.paste(right, (width, 24))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def _scenario_values(stats: dict) -> tuple[list[str], np.ndarray]:
    median = np.asarray([stats[name]["median"] for name in CONTROLS], dtype=np.float32)
    index = {name: i for i, name in enumerate(CONTROLS)}

    scenarios = {}

    def add(name: str, **changes: tuple[str, str]) -> None:
        values = median.copy()
        for control, level in changes.items():
            values[index[control]] = stats[control][level]
        scenarios[name] = values

    add("median")
    add("green_high", green_coverage="p90")
    add("building_high", building_coverage="p90")
    add("roads_high", road_length_km_per_km2="p90")
    add("water_high", water_coverage="p90")
    add(
        "green_building_high",
        green_coverage="p90",
        building_coverage="p90",
    )
    add(
        "dense_roads",
        building_coverage="p90",
        road_length_km_per_km2="p90",
        road_major_share="p90",
    )

    names = list(scenarios)
    values = np.stack([scenarios[name] for name in names])
    return names, values


def run(checkpoint_path: Path, output: Path, *, device_name: str, steps: int) -> dict:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = LayeredDiffusionConfig(**checkpoint["config"])
    stats = checkpoint["control_stats"]

    rows = []

    validation = LayeredBlockDataset(
        config,
        config.validation_manifest,
        augment=False,
    )
    for index in range(len(validation)):
        item = validation[index]
        classes = item["x0"][:SURFACE_CLASS_COUNT].argmax(dim=0).numpy().astype(np.uint8)
        _state, metrics, _masks = audit_classes(classes)
        rows.append(
            {
                "source": "real",
                "sample_id": str(item["tile_id"]),
                "scenario": "",
                "seed": -1,
                **metrics,
            }
        )

    device = torch.device(device_name)
    model = build_model(config).to(device)
    ema = checkpoint["ema"]
    model.load_state_dict(ema.get("shadow", ema))
    model.eval()

    scenario_names, raw_values = _scenario_values(stats)
    conditions = normalise(raw_values, stats)

    generated_root = output / "generated"
    for seed in (101, 202, 303):
        values = sample(
            model,
            config,
            conditions,
            seed=seed,
            steps=steps,
            device=device,
            same_noise=True,
        )
        class_maps = _classes(values)

        for scenario, classes in zip(scenario_names, class_maps, strict=True):
            state, metrics, masks = audit_classes(classes, seed=seed)
            state["generation"]["scenario"] = scenario

            sample_id = f"{scenario}-{seed}"
            sample_dir = generated_root / sample_id
            write_state(sample_dir / "city.json", state)
            _save_pair(
                classes,
                _compiled_map(masks),
                sample_dir / "roundtrip.png",
                sample_id,
            )

            rows.append(
                {
                    "source": "generated",
                    "sample_id": sample_id,
                    "scenario": scenario,
                    "seed": seed,
                    **metrics,
                }
            )

        del values
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fields = list(rows[0])
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    real = [row for row in rows if row["source"] == "real"]
    generated = [row for row in rows if row["source"] == "generated"]
    real_summary = _summary(real)
    generated_summary = _summary(generated)

    comparison = {}
    for key in generated_summary:
        if key not in real_summary:
            continue
        g = generated_summary[key]["median"]
        r = real_summary[key]["median"]
        comparison[key] = {
            "generated_median": g,
            "real_median": r,
            "difference": g - r,
            "ratio": g / r if abs(r) > 1e-12 else None,
        }

    result = {
        "experiment": "structured-output-v1",
        "checkpoint": str(checkpoint_path),
        "validation_tiles": len(real),
        "generated_samples": len(generated),
        "inference_steps": steps,
        "scenarios": scenario_names,
        "real": real_summary,
        "generated": generated_summary,
        "comparison": comparison,
        "notes": {
            "scope": "surface structure only",
            "height": "not generated in this experiment",
            "vertical_transport": "not generated in this experiment",
            "road_cleanup": "one-pixel binary closing before skeletonization",
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()

    run(
        args.checkpoint.expanduser().resolve(),
        args.output.expanduser().resolve(),
        device_name=args.device,
        steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
