from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point, MultiPoint

from .connectivity import _city_json, _read_rows
from .generated_city_audit import (
    _bounds,
    _graph_metrics,
    _near_boundary,
    _position,
    _transport_graph,
)


def _percentile(values: list[float], q: float) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    return float(np.quantile(finite, q)) if finite.size else 0.0


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows), "metrics": {}}
    if not rows:
        return result
    ignored = {"source", "sample_id", "path"}
    for key in sorted(rows[0]):
        if key in ignored:
            continue
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
        ]
        if not values:
            continue
        result["metrics"][key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": _percentile(values, 0.10),
            "p90": _percentile(values, 0.90),
        }
    return result


def _surface_edges(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        edge
        for edge in state.get("transport_graph", {}).get("edges", [])
        if edge.get("transport_mode") == "road"
        and edge.get("vertical_mode") == "surface"
        and len(edge.get("geometry_local_m", [])) >= 2
    ]


def _line(edge: dict[str, Any]) -> LineString:
    return LineString(
        [
            (float(point[0]), float(point[1]))
            for point in edge.get("geometry_local_m", [])
        ]
    )


def _crossing_metrics(edges: list[dict[str, Any]]) -> dict[str, float | int]:
    if len(edges) < 2:
        return {"unnoded_crossings": 0, "overlap_pair_count": 0, "overlap_length_m": 0.0}

    lines = [_line(edge) for edge in edges]
    tree = STRtree(lines)
    crossings = 0
    overlaps = 0
    overlap_length = 0.0

    for index, line in enumerate(lines):
        try:
            candidates = tree.query(line)
            candidate_indexes = [int(value) for value in candidates]
        except (TypeError, ValueError):
            candidate_indexes = [
                other
                for other, candidate in enumerate(lines)
                if line.bounds[0] <= candidate.bounds[2]
                and line.bounds[2] >= candidate.bounds[0]
                and line.bounds[1] <= candidate.bounds[3]
                and line.bounds[3] >= candidate.bounds[1]
            ]

        first = edges[index]
        first_nodes = {str(first.get("from_node")), str(first.get("to_node"))}
        for other in candidate_indexes:
            if other <= index:
                continue
            second = edges[other]
            second_nodes = {str(second.get("from_node")), str(second.get("to_node"))}
            if first_nodes & second_nodes:
                continue

            intersection = line.intersection(lines[other])
            if intersection.is_empty:
                continue
            if isinstance(intersection, Point):
                crossings += 1
            elif isinstance(intersection, MultiPoint):
                crossings += len(intersection.geoms)
            elif intersection.length > 1e-6:
                overlaps += 1
                overlap_length += float(intersection.length)

    return {
        "unnoded_crossings": int(crossings),
        "overlap_pair_count": int(overlaps),
        "overlap_length_m": float(overlap_length),
    }


def _hierarchy_metrics(graph, edges: list[dict[str, Any]]) -> dict[str, float]:
    class_length: dict[str, float] = defaultdict(float)
    incident: dict[str, set[str]] = defaultdict(set)

    for edge in edges:
        edge_class = str(edge.get("class") or "local")
        length = max(0.0, float(edge.get("length_m", 0.0)))
        class_length[edge_class] += length
        incident[str(edge.get("from_node"))].add(edge_class)
        incident[str(edge.get("to_node"))].add(edge_class)

    road_length = sum(class_length.values())
    local_total = class_length["local"]
    secondary_total = class_length["secondary"]

    local_touch = 0.0
    secondary_touch = 0.0
    for edge in edges:
        edge_class = str(edge.get("class") or "local")
        length = max(0.0, float(edge.get("length_m", 0.0)))
        neighbourhood = (
            incident[str(edge.get("from_node"))]
            | incident[str(edge.get("to_node"))]
        )
        if edge_class == "local" and neighbourhood & {"major", "secondary"}:
            local_touch += length
        if edge_class == "secondary" and "major" in neighbourhood:
            secondary_touch += length

    return {
        "major_length_share": class_length["major"] / road_length if road_length else 0.0,
        "secondary_length_share": (
            class_length["secondary"] / road_length if road_length else 0.0
        ),
        "local_length_share": class_length["local"] / road_length if road_length else 0.0,
        "local_direct_higher_contact_fraction": (
            local_touch / local_total if local_total else 1.0
        ),
        "secondary_direct_major_contact_fraction": (
            secondary_touch / secondary_total if secondary_total else 1.0
        ),
    }


def audit_state(
    path: Path,
    source: str,
    *,
    largest_component_only: bool = False,
) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    bounds = _bounds(state)
    graph = _transport_graph(state, "road", vertical="surface")
    edges = _surface_edges(state)

    if largest_component_only and graph.number_of_edges() > 0:
        components = list(nx.connected_components(graph))

        def component_length(nodes):
            return sum(
                float(data.get("length_m", 0.0))
                for left, right, data in graph.edges(nodes, data=True)
                if left in nodes and right in nodes
            )

        keep = max(components, key=lambda nodes: (component_length(nodes), len(nodes)))
        graph = graph.subgraph(keep).copy()
        edges = [
            edge
            for edge in edges
            if str(edge.get("from_node")) in keep and str(edge.get("to_node")) in keep
        ]

    graph_metrics = _graph_metrics(graph, bounds)

    lengths = [
        max(0.0, float(edge.get("length_m", _line(edge).length)))
        for edge in edges
    ]
    total_length = sum(lengths)
    junctions = sum(graph.degree(node) >= 3 for node in graph.nodes)
    dead_ends = int(graph_metrics["interior_dead_ends"])
    boundary_endpoints = sum(
        graph.degree(node) == 1
        and _near_boundary(*_position(data)[:2], bounds)
        for node, data in graph.nodes(data=True)
    )

    crossings = _crossing_metrics(edges)
    hierarchy = _hierarchy_metrics(graph, edges)

    positions = [_position(data) for _node, data in graph.nodes(data=True)]
    if positions:
        xs = [value[0] for value in positions]
        ys = [value[1] for value in positions]
        width = max(bounds[2] - bounds[0], 1e-9)
        height = max(bounds[3] - bounds[1], 1e-9)
        span_x = (max(xs) - min(xs)) / width
        span_y = (max(ys) - min(ys)) / height
        hull = MultiPoint([(value[0], value[1]) for value in positions]).convex_hull
        hull_fraction = float(hull.area / (width * height)) if hasattr(hull, "area") else 0.0
    else:
        span_x = 0.0
        span_y = 0.0
        hull_fraction = 0.0

    return {
        "source": source,
        "sample_id": str(state.get("tile", {}).get("tile_id") or path.parent.name),
        "path": str(path),
        "road_components": int(graph_metrics["components"]),
        "road_length_km": float(total_length / 1000.0),
        "largest_length_fraction": float(graph_metrics["largest_length_fraction"]),
        "interior_component_length_fraction": float(
            graph_metrics["interior_component_length_fraction"]
        ),
        "junctions_per_km": float(junctions / (total_length / 1000.0))
        if total_length
        else 0.0,
        "interior_dead_ends_per_km": float(dead_ends / (total_length / 1000.0))
        if total_length
        else 0.0,
        "boundary_endpoints": int(boundary_endpoints),
        "network_span_x_fraction": float(span_x),
        "network_span_y_fraction": float(span_y),
        "network_hull_area_fraction": float(hull_fraction),
        "edge_length_median_m": float(np.median(lengths)) if lengths else 0.0,
        "edge_length_p90_m": _percentile(lengths, 0.90),
        **hierarchy,
        **crossings,
    }


def _generated_paths(roots: Iterable[Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("city.json"):
            found[str(path.resolve())] = path.resolve()
    return [found[key] for key in sorted(found)]


def _real_paths(manifests: Iterable[Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for row, manifest in _read_rows(manifests):
        path = _city_json(row, manifest)
        found[str(path.resolve())] = path.resolve()
    return [found[key] for key in sorted(found)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(
    generated_roots: Iterable[str | Path],
    real_manifests: Iterable[str | Path],
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output).expanduser().resolve()
    if output.exists() and overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    generated_paths = _generated_paths(
        [Path(value).expanduser().resolve() for value in generated_roots]
    )
    real_paths = _real_paths(
        [Path(value).expanduser().resolve() for value in real_manifests]
    )
    if not generated_paths:
        raise ValueError("No generated city.json files found")
    if not real_paths:
        raise ValueError("No real city.json files found")

    generated = [audit_state(path, "generated") for path in generated_paths]
    real_full = [audit_state(path, "real_full") for path in real_paths]
    real = [
        audit_state(path, "real_main", largest_component_only=True)
        for path in real_paths
    ]
    _write_csv(output / "tiles.csv", [*generated, *real, *real_full])

    generated_summary = _summary(generated)
    real_summary = _summary(real)
    real_full_summary = _summary(real_full)
    comparison: dict[str, Any] = {}
    for metric, values in generated_summary["metrics"].items():
        if metric not in real_summary["metrics"]:
            continue
        g = values["median"]
        r = real_summary["metrics"][metric]["median"]
        comparison[metric] = {
            "generated_median": g,
            "real_median": r,
            "difference": g - r,
            "ratio": g / r if abs(r) > 1e-12 else None,
        }

    result = {
        "analysis_version": 1,
        "generated": generated_summary,
        "real": real_summary,
        "real_full": real_full_summary,
        "comparison": comparison,
        "notes": {
            "scope": "surface-road structural geometry only",
            "comparison_reference": (
                "Headline comparison uses the largest connected real surface-road component, "
                "matching the structural-v2 training target. real_full is also reported."
            ),
            "unnoded_crossing": (
                "Same-level road geometries intersect although the two edges share no graph node."
            ),
            "direct_hierarchy_contact": (
                "Length-weighted fraction whose edge touches a node incident to the next higher road class."
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare explicit generated road graphs with corrected real road graphs"
    )
    parser.add_argument("--generated", action="append", required=True, type=Path)
    parser.add_argument("--real-manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit(
            args.generated,
            args.real_manifest,
            args.output,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
