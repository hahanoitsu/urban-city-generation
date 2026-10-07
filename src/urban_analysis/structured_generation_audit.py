from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from shapely.geometry import Polygon


def polygon_stats(points, size, tolerance=0.5):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    valid = False
    if len(points) >= 3:
        shape = Polygon(points)
        valid = bool(shape.is_valid and shape.area > 1e-3)
    if len(points) == 0:
        return valid, False, 0.0
    outside = ((points < -tolerance) | (points > size + tolerance)).any(axis=1)
    return valid, bool(outside.any()), float(outside.mean())


def polygon_group(items, field, size):
    valid = 0
    outside = 0
    vertex_outside = []
    for item in items:
        ok, out, fraction = polygon_stats(item[field], size)
        valid += int(ok)
        outside += int(out)
        vertex_outside.append(fraction)
    return {
        "count": len(items),
        "valid": valid,
        "outside": outside,
        "vertex_outside_sum": float(sum(vertex_outside)),
    }


def transport_stats(nodes, edges):
    graph = nx.Graph()
    positions = {}
    for node in nodes:
        graph.add_node(node["id"])
        positions[node["id"]] = np.asarray(node["xyz_m"][:2], dtype=float)
    tortuosity = []
    endpoint_gap = []
    for edge in edges:
        graph.add_edge(edge["from"], edge["to"])
        line = np.asarray([point[:2] for point in edge["spline_xyz_m"]], dtype=float)
        if len(line) < 2:
            continue
        chord = float(np.linalg.norm(line[-1] - line[0]))
        length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
        if chord > 1e-6:
            tortuosity.append(length / chord)
        if edge["from"] in positions and edge["to"] in positions:
            endpoint_gap.append(float(np.linalg.norm(line[0] - positions[edge["from"]])))
            endpoint_gap.append(float(np.linalg.norm(line[-1] - positions[edge["to"]])))
    isolated = sum(1 for node in graph.nodes if graph.degree(node) == 0)
    components = list(nx.connected_components(graph))
    largest = max((len(item) for item in components), default=0)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": len(edges),
        "isolated_nodes": isolated,
        "components": len(components),
        "largest_component_nodes": largest,
        "tortuosity": tortuosity,
        "endpoint_gap_m": endpoint_gap,
    }


def audit_scene(scene, size=512.0):
    buildings = polygon_group(scene.get("buildings", []), "footprint_xy_m", size)
    areas = polygon_group(scene.get("areas", []), "polygon_xy_m", size)
    transport = transport_stats(scene.get("nodes", []), scene.get("edges", []))
    heights = [item["height_m"] for item in scene.get("buildings", [])]
    return {
        "buildings": buildings,
        "areas": areas,
        "transport": transport,
        "negative_heights": int(sum(1 for value in heights if value < 0)),
    }


def fraction(part, whole):
    return float(part) / whole if whole else 0.0


def summarise(rows):
    result = {}
    for name in ("buildings", "areas"):
        count = sum(row[name]["count"] for row in rows)
        result[name] = {
            "count": count,
            "valid_fraction": fraction(sum(row[name]["valid"] for row in rows), count),
            "any_vertex_outside_fraction": fraction(sum(row[name]["outside"] for row in rows), count),
            "vertex_outside_fraction": fraction(sum(row[name]["vertex_outside_sum"] for row in rows), count),
        }
    heights = sum(row["buildings"]["count"] for row in rows)
    result["buildings"]["negative_height_fraction"] = fraction(
        sum(row["negative_heights"] for row in rows), heights
    )
    nodes = sum(row["transport"]["nodes"] for row in rows)
    tortuosity = [value for row in rows for value in row["transport"]["tortuosity"]]
    gaps = [value for row in rows for value in row["transport"]["endpoint_gap_m"]]
    result["transport"] = {
        "nodes": nodes,
        "edges": sum(row["transport"]["edges"] for row in rows),
        "isolated_node_fraction": fraction(sum(row["transport"]["isolated_nodes"] for row in rows), nodes),
        "mean_components_per_tile": float(np.mean([row["transport"]["components"] for row in rows])),
        "largest_component_node_fraction": fraction(
            sum(row["transport"]["largest_component_nodes"] for row in rows), nodes
        ),
        "median_tortuosity": float(np.median(tortuosity)) if tortuosity else 0.0,
        "median_endpoint_gap_m": float(np.median(gaps)) if gaps else 0.0,
    }
    return result


def match_buildings(scene, target):
    generated = [np.asarray(item["footprint_xy_m"], dtype=float) for item in scene.get("buildings", [])]
    real = [np.asarray(item["footprint_xy_m"], dtype=float) for item in target.get("buildings", [])]
    if not generated or not real:
        return []
    from scipy.optimize import linear_sum_assignment

    left = np.asarray([points.mean(axis=0) for points in generated])
    right = np.asarray([points.mean(axis=0) for points in real])
    cost = np.linalg.norm(left[:, None] - right[None], axis=2)
    rows, columns = linear_sum_assignment(cost)
    result = []
    for row, column in zip(rows, columns):
        guess = generated[row]
        truth = real[column]
        radius = float(np.sqrt(((truth - truth.mean(axis=0)) ** 2).sum(axis=1).mean()))
        overlap = 0.0
        shape = Polygon(truth)
        if len(guess) >= 3 and shape.is_valid and shape.area > 0:
            other = Polygon(guess).buffer(0)
            overlap = float(shape.intersection(other).area / max(shape.union(other).area, 1e-9))
        result.append(
            {
                "centroid_error_m": float(cost[row, column]),
                "radius_m": radius,
                "iou": overlap,
            }
        )
    return result


def match_summary(pairs):
    if not pairs:
        return {}
    centroid = [item["centroid_error_m"] for item in pairs]
    radius = [item["radius_m"] for item in pairs]
    overlap = [item["iou"] for item in pairs]
    return {
        "matched": len(pairs),
        "median_centroid_error_m": float(np.median(centroid)),
        "median_radius_m": float(np.median(radius)),
        "median_iou": float(np.median(overlap)),
        "iou_over_half_fraction": float(np.mean([value > 0.5 for value in overlap])),
    }


def load_scenes(folder):
    paths = sorted(path for path in Path(folder).glob("*.json") if path.name != "summary.json")
    return [(path.name, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--size", type=float, default=512.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--targets", type=Path)
    args = parser.parse_args()

    scenes = load_scenes(args.generations)
    if not scenes:
        raise SystemExit("no json files found")
    rows = [audit_scene(scene, args.size) for _name, scene in scenes]
    summary = summarise(rows)
    if args.targets:
        pairs = []
        for name, scene in scenes:
            path = args.targets / name
            if path.exists():
                pairs.extend(match_buildings(scene, json.loads(path.read_text(encoding="utf-8"))))
        summary["matched_buildings"] = match_summary(pairs)
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
