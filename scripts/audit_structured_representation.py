from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import unary_union

from urban_model.structured_city_data import (
    SceneTensorConfig,
    _iter_polygons,
    _polygon_records,
    _resample_line,
    _resample_ring,
    _transport_graph,
)


def quantiles(values):
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def polygon_fidelity(polygon, points):
    reconstructed = Polygon(points)
    if not reconstructed.is_valid:
        reconstructed = reconstructed.buffer(0)
    if reconstructed.is_empty:
        return 0.0, float("inf")
    union = polygon.union(reconstructed).area
    intersection = polygon.intersection(reconstructed).area
    iou = intersection / union if union > 1e-8 else 1.0
    return float(iou), float(polygon.hausdorff_distance(reconstructed))


def transport_fidelity(payload, config):
    graph = _transport_graph(payload, config.target_size_m)
    values = {}
    for mode, key in (("road", "roads"), ("rail", "rail")):
        raw_lines = [
            LineString(record["geometry_local_m"])
            for record in payload["target"].get(key, [])
            if len(record.get("geometry_local_m", [])) >= 2
        ]
        encoded_lines = [
            LineString(_resample_line(edge["geometry_local_m"], config.edge_shape_points))
            for edge in graph["edges"]
            if edge["transport_mode"] == mode
        ]
        if not raw_lines or not encoded_lines:
            continue
        raw = unary_union(raw_lines)
        encoded = unary_union(encoded_lines)
        values[mode] = {
            "hausdorff_m": float(raw.hausdorff_distance(encoded)),
            "length_ratio": float(encoded.length / max(raw.length, 1e-8)),
        }
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport-samples", type=int, default=400)
    parser.add_argument("--polygon-samples", type=int, default=12000)
    args = parser.parse_args()

    config = SceneTensorConfig(
        node_slots=448,
        edge_slots=512,
        building_slots=832,
        area_slots=160,
        maximum_ports=96,
    )
    rows = [
        json.loads(line)
        for line in (args.data / "targets.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    building_iou = []
    building_hausdorff = []
    area_iou = []
    area_hausdorff = []
    building_holes = 0
    building_polygons = 0
    area_holes = 0
    area_polygons = 0
    invalid_buildings = 0
    invalid_areas = 0
    height_sources = Counter()
    building_types = Counter()
    width_sources = Counter()
    transport = {
        "road_hausdorff_m": [],
        "road_length_ratio": [],
        "rail_hausdorff_m": [],
        "rail_length_ratio": [],
    }

    stride = max(1, len(rows) // max(args.transport_samples, 1))
    transport_indexes = set(range(0, len(rows), stride))
    building_budget = args.polygon_samples
    area_budget = args.polygon_samples

    for index, row in enumerate(rows):
        with gzip.open(args.data / row["sample_path"], "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        if index in transport_indexes and len(transport["road_hausdorff_m"]) < args.transport_samples:
            values = transport_fidelity(payload, config)
            for mode in ("road", "rail"):
                if mode not in values:
                    continue
                transport[f"{mode}_hausdorff_m"].append(values[mode]["hausdorff_m"])
                transport[f"{mode}_length_ratio"].append(values[mode]["length_ratio"])

        for record in payload["target"].get("roads", []):
            width_sources[str(record.get("width_source", "missing"))] += 1

        for record in payload["target"].get("buildings", []):
            height_sources[str(record.get("height_source", "unknown"))] += 1
            building_types[str(record.get("building_type", "unknown"))] += 1
            geometry_payload = record.get("footprint_local_m")
            if not geometry_payload:
                continue
            geometry = shape(geometry_payload)
            for polygon in _iter_polygons(geometry):
                building_polygons += 1
                building_holes += len(polygon.interiors)
                if not polygon.is_valid:
                    invalid_buildings += 1
                if building_budget <= 0:
                    continue
                points = _resample_ring(polygon, config.building_points)
                iou, hausdorff = polygon_fidelity(polygon, points)
                building_iou.append(iou)
                building_hausdorff.append(hausdorff)
                building_budget -= 1

        for _kind, polygon in _polygon_records(payload):
            area_polygons += 1
            area_holes += len(polygon.interiors)
            if not polygon.is_valid:
                invalid_areas += 1
            if area_budget <= 0:
                continue
            points = _resample_ring(polygon, config.area_points)
            iou, hausdorff = polygon_fidelity(polygon, points)
            area_iou.append(iou)
            area_hausdorff.append(hausdorff)
            area_budget -= 1

        if index == 0 or (index + 1) % 250 == 0 or index + 1 == len(rows):
            print(f"{index + 1}/{len(rows)}", flush=True)

    summary = {
        "samples": len(rows),
        "transport": {
            "road_hausdorff_m": quantiles(transport["road_hausdorff_m"]),
            "road_length_ratio": quantiles(transport["road_length_ratio"]),
            "rail_hausdorff_m": quantiles(transport["rail_hausdorff_m"]),
            "rail_length_ratio": quantiles(transport["rail_length_ratio"]),
        },
        "buildings": {
            "polygons": building_polygons,
            "holes": building_holes,
            "polygons_with_invalid_source_geometry": invalid_buildings,
            "sampled_iou": quantiles(building_iou),
            "sampled_hausdorff_m": quantiles(building_hausdorff),
            "height_sources": dict(height_sources.most_common()),
            "types": dict(building_types.most_common(30)),
        },
        "areas": {
            "polygons": area_polygons,
            "holes": area_holes,
            "polygons_with_invalid_source_geometry": invalid_areas,
            "sampled_iou": quantiles(area_iou),
            "sampled_hausdorff_m": quantiles(area_hausdorff),
        },
        "road_width_sources": dict(width_sources.most_common()),
        "tensor_points": {
            "transport_edge": config.edge_shape_points,
            "building": config.building_points,
            "area": config.area_points,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
