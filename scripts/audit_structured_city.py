from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from urban_model.structured_city_data import SceneTensorConfig, scene_counts


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {}
    return {
        "min": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = SceneTensorConfig()
    rows = [
        json.loads(line)
        for line in (args.data / "targets.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    values = {name: [] for name in ("nodes", "edges", "buildings", "areas", "ports")}
    overflow = {name: 0 for name in ("nodes", "edges", "buildings", "areas", "ports")}
    fit_all = 0
    height_valid = 0
    height_total = 0
    width_valid = 0
    width_total = 0

    for index, row in enumerate(rows, start=1):
        with gzip.open(args.data / row["sample_path"], "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        counts = scene_counts(payload, config)
        fits = True
        limits = {
            "nodes": config.node_slots,
            "edges": config.edge_slots,
            "buildings": config.building_slots,
            "areas": config.area_slots,
            "ports": config.maximum_ports,
        }
        for name in values:
            values[name].append(counts[name])
            if counts[name] > limits[name]:
                overflow[name] += 1
                fits = False
        if fits:
            fit_all += 1
        for building in payload["target"].get("buildings", []):
            height_total += 1
            if bool(building.get("height_valid", building.get("height_m") is not None)):
                height_valid += 1
        for road in payload["target"].get("roads", []):
            width_total += 1
            if bool(road.get("width_valid", road.get("width_m") is not None)):
                width_valid += 1
        if index == 1 or index % 250 == 0 or index == len(rows):
            print(f"{index}/{len(rows)}", flush=True)

    summary = {
        "samples": len(rows),
        "counts": {name: stats(items) for name, items in values.items()},
        "building_height_valid_fraction": height_valid / max(height_total, 1),
        "road_width_valid_fraction": width_valid / max(width_total, 1),
        "samples_fitting_all_current_slots": fit_all,
        "samples_rejected_by_current_slots": len(rows) - fit_all,
        "overflow_samples": overflow,
        "current_slots": {
            "nodes": config.node_slots,
            "edges": config.edge_slots,
            "buildings": config.building_slots,
            "areas": config.area_slots,
            "ports": config.maximum_ports,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
