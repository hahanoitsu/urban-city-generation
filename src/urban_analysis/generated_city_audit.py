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
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union

from .connectivity import _city_json, _read_rows


def _percentile(values: list[float], q: float) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return None
    return float(np.quantile(finite, q))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows), "metrics": {}}
    if not rows:
        return result
    keys = sorted(
        key
        for key in rows[0]
        if key not in {"source", "sample_id", "path"}
        and all(isinstance(row.get(key), (int, float)) or row.get(key) is None for row in rows)
    )
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]
        if not values:
            continue
        result["metrics"][key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": _percentile(values, 0.10),
            "p90": _percentile(values, 0.90),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def _bounds(state: dict[str, Any]) -> tuple[float, float, float, float]:
    values = state.get("coordinate_system", {}).get(
        "local_bounds", [0.0, 0.0, 1024.0, 1024.0]
    )
    if len(values) != 4:
        return 0.0, 0.0, 1024.0, 1024.0
    return tuple(float(value) for value in values)


def _near_boundary(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    tolerance: float = 6.0,
) -> bool:
    minx, miny, maxx, maxy = bounds
    return (
        abs(x - minx) <= tolerance
        or abs(x - maxx) <= tolerance
        or abs(y - miny) <= tolerance
        or abs(y - maxy) <= tolerance
    )


def _position(node: dict[str, Any]) -> tuple[float, float, float]:
    values = node.get("position_local_m")
    if not values or len(values) < 2:
        return float("nan"), float("nan"), float("nan")
    z = 0.0 if len(values) < 3 or values[2] is None else float(values[2])
    return float(values[0]), float(values[1]), z


def _transport_graph(
    state: dict[str, Any],
    mode: str,
    *,
    vertical: str | None = None,
) -> nx.Graph:
    graph = nx.Graph()
    nodes = {
        str(node["id"]): node
        for node in state.get("transport_graph", {}).get("nodes", [])
        if node.get("transport_mode") == mode
    }
    for node_id, node in nodes.items():
        graph.add_node(node_id, **node)

    for edge in state.get("transport_graph", {}).get("edges", []):
        if edge.get("transport_mode") != mode:
            continue
        if vertical is not None and edge.get("vertical_mode") != vertical:
            continue
        left = str(edge["from_node"])
        right = str(edge["to_node"])
        if left not in nodes or right not in nodes:
            continue
        graph.add_edge(
            left,
            right,
            length_m=float(edge.get("length_m", 0.0)),
            edge_class=str(edge.get("class") or ""),
            vertical_mode=str(edge.get("vertical_mode") or ""),
            width_m=float(edge.get("width_m", 0.0) or 0.0),
            geometry_local_m=edge.get("geometry_local_m", []),
        )

    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def _add_local_vertical_transitions(
    graph: nx.Graph,
    tolerance_m: float = 1.5,
) -> nx.Graph:
    result = graph.copy()
    nodes = list(result.nodes(data=True))
    for index, (left_id, left) in enumerate(nodes):
        if result.degree(left_id) > 2:
            continue
        lx, ly, _ = _position(left)
        if not np.isfinite([lx, ly]).all():
            continue
        left_mode = str(left.get("vertical_mode") or "")
        for right_id, right in nodes[index + 1 :]:
            if result.degree(right_id) > 2:
                continue
            right_mode = str(right.get("vertical_mode") or "")
            pair = frozenset((left_mode, right_mode))
            if pair not in {
                frozenset(("surface", "elevated")),
                frozenset(("surface", "underground")),
            }:
                continue
            rx, ry, _ = _position(right)
            if not np.isfinite([rx, ry]).all():
                continue
            if math.hypot(lx - rx, ly - ry) <= tolerance_m:
                result.add_edge(
                    left_id,
                    right_id,
                    length_m=0.0,
                    edge_class="transition",
                    vertical_mode="transition",
                    width_m=0.0,
                    geometry_local_m=[],
                )
    return result


def _component_lengths(graph: nx.Graph) -> tuple[list[set[str]], list[float]]:
    components = [set(values) for values in nx.connected_components(graph)]
    lengths: list[float] = []
    for component in components:
        total = sum(
            float(data.get("length_m", 0.0))
            for left, right, data in graph.edges(component, data=True)
            if left in component and right in component
        )
        lengths.append(float(total))
    return components, lengths


def _graph_metrics(
    graph: nx.Graph,
    bounds: tuple[float, float, float, float],
) -> dict[str, float | int]:
    if graph.number_of_edges() == 0:
        return {
            "components": 0,
            "total_length_m": 0.0,
            "largest_length_fraction": 0.0,
            "node_pair_reachability": 0.0,
            "boundary_exits": 0,
            "interior_dead_ends": 0,
            "interior_components": 0,
            "interior_component_length_fraction": 0.0,
        }

    components, lengths = _component_lengths(graph)
    total = float(sum(lengths))
    boundary_nodes = {
        node_id
        for node_id, node in graph.nodes(data=True)
        if _near_boundary(*_position(node)[:2], bounds)
    }
    interior_indexes = [
        index
        for index, component in enumerate(components)
        if not (component & boundary_nodes)
    ]
    interior_length = sum(lengths[index] for index in interior_indexes)
    count = graph.number_of_nodes()
    possible = count * (count - 1)
    reachable = sum(len(component) * (len(component) - 1) for component in components)

    return {
        "components": len(components),
        "total_length_m": total,
        "largest_length_fraction": max(lengths, default=0.0) / total if total else 0.0,
        "node_pair_reachability": reachable / possible if possible else 1.0,
        "boundary_exits": sum(graph.degree(node) == 1 for node in boundary_nodes),
        "interior_dead_ends": sum(
            graph.degree(node) == 1 and node not in boundary_nodes for node in graph.nodes
        ),
        "interior_components": len(interior_indexes),
        "interior_component_length_fraction": interior_length / total if total else 0.0,
    }


def _road_lines(state: dict[str, Any], vertical: str = "surface") -> list[LineString]:
    lines: list[LineString] = []
    for edge in state.get("transport_graph", {}).get("edges", []):
        if edge.get("transport_mode") != "road" or edge.get("vertical_mode") != vertical:
            continue
        coords = edge.get("geometry_local_m", [])
        if len(coords) < 2:
            continue
        line = LineString([(float(value[0]), float(value[1])) for value in coords])
        if not line.is_empty and line.length > 0:
            lines.append(line)
    return lines


def _buildings(state: dict[str, Any]) -> list[tuple[Any, float]]:
    result: list[tuple[Any, float]] = []
    for item in state.get("building_solids", []):
        raw = item.get("footprint_local_m")
        if not raw:
            continue
        try:
            polygon = shape(raw)
        except Exception:
            continue
        if polygon.is_empty or polygon.area <= 0:
            continue
        result.append((polygon, float(item.get("height_m", 0.0) or 0.0)))
    return result


def _building_metrics(
    state: dict[str, Any],
    surface_graph: nx.Graph,
    bounds: tuple[float, float, float, float],
) -> dict[str, float | int]:
    buildings = _buildings(state)
    lines = _road_lines(state)
    road_union = unary_union(lines) if lines else None

    areas = [float(polygon.area) for polygon, _ in buildings]
    heights = [height for _, height in buildings]
    distances = (
        [float(polygon.distance(road_union)) for polygon, _ in buildings]
        if road_union is not None and not road_union.is_empty
        else [float("inf")] * len(buildings)
    )

    tile_area = max((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]), 1.0)

    dead_end_distances: list[float] = []
    if buildings:
        building_union = unary_union([polygon for polygon, _ in buildings])
        for node_id, node in surface_graph.nodes(data=True):
            if surface_graph.degree(node_id) != 1:
                continue
            x, y, _ = _position(node)
            if _near_boundary(x, y, bounds):
                continue
            dead_end_distances.append(float(Point(x, y).distance(building_union)))

    finite_distances = [value for value in distances if np.isfinite(value)]
    return {
        "building_count": len(buildings),
        "building_coverage": sum(areas) / tile_area if areas else 0.0,
        "building_area_median_m2": float(np.median(areas)) if areas else 0.0,
        "building_area_p90_m2": _percentile(areas, 0.90) or 0.0,
        "building_area_max_m2": max(areas, default=0.0),
        "building_height_median_m": float(np.median(heights)) if heights else 0.0,
        "building_height_p90_m": _percentile(heights, 0.90) or 0.0,
        "building_road_distance_median_m": (
            float(np.median(finite_distances)) if finite_distances else float("nan")
        ),
        "building_road_distance_p90_m": (
            _percentile(finite_distances, 0.90) if finite_distances else float("nan")
        ),
        "buildings_within_20m_road_fraction": (
            sum(value <= 20.0 for value in distances) / len(distances) if distances else 0.0
        ),
        "buildings_within_40m_road_fraction": (
            sum(value <= 40.0 for value in distances) / len(distances) if distances else 0.0
        ),
        "buildings_beyond_80m_road_fraction": (
            sum(value > 80.0 for value in distances) / len(distances) if distances else 0.0
        ),
        "interior_dead_ends_near_building_fraction": (
            sum(value <= 30.0 for value in dead_end_distances) / len(dead_end_distances)
            if dead_end_distances
            else 1.0
        ),
        "interior_dead_ends_over_60m_from_building": sum(
            value > 60.0 for value in dead_end_distances
        ),
    }


def _hierarchy_metrics(
    graph: nx.Graph,
    state: dict[str, Any],
) -> dict[str, float]:
    if graph.number_of_edges() == 0:
        return {
            "local_length_connected_to_higher_fraction": 0.0,
            "road_component_length_serving_buildings_fraction": 0.0,
        }

    buildings = _buildings(state)
    building_union = unary_union([polygon for polygon, _ in buildings]) if buildings else None
    components, lengths = _component_lengths(graph)
    component_index = {
        node: index for index, component in enumerate(components) for node in component
    }

    local_total = 0.0
    local_good = 0.0
    component_classes: dict[int, set[str]] = defaultdict(set)
    for left, _right, data in graph.edges(data=True):
        index = component_index[left]
        edge_class = str(data.get("edge_class") or "")
        component_classes[index].add(edge_class)
        if edge_class == "local":
            local_total += float(data.get("length_m", 0.0))

    for left, _right, data in graph.edges(data=True):
        if str(data.get("edge_class") or "") != "local":
            continue
        index = component_index[left]
        if component_classes[index] & {"major", "secondary"}:
            local_good += float(data.get("length_m", 0.0))

    served_length = 0.0
    total_length = float(sum(lengths))
    if building_union is not None and not building_union.is_empty:
        for index, component in enumerate(components):
            served = False
            for left, right, data in graph.edges(component, data=True):
                if left not in component or right not in component:
                    continue
                coords = data.get("geometry_local_m", [])
                if len(coords) < 2:
                    continue
                line = LineString([(float(value[0]), float(value[1])) for value in coords])
                if line.distance(building_union) <= 40.0:
                    served = True
                    break
            if served:
                served_length += lengths[index]

    return {
        "local_length_connected_to_higher_fraction": (
            local_good / local_total if local_total else 1.0
        ),
        "road_component_length_serving_buildings_fraction": (
            served_length / total_length if total_length else 0.0
        ),
    }


def audit_state(path: Path, source: str) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    bounds = _bounds(state)

    road_raw = _transport_graph(state, "road")
    road_assisted = _add_local_vertical_transitions(road_raw)
    road_surface = _transport_graph(state, "road", vertical="surface")
    rail_raw = _transport_graph(state, "rail")
    rail_assisted = _add_local_vertical_transitions(rail_raw)

    road = _graph_metrics(road_raw, bounds)
    road_help = _graph_metrics(road_assisted, bounds)
    surface = _graph_metrics(road_surface, bounds)
    rail = _graph_metrics(rail_raw, bounds)
    rail_help = _graph_metrics(rail_assisted, bounds)
    building = _building_metrics(state, road_surface, bounds)
    hierarchy = _hierarchy_metrics(road_surface, state)

    maximum_grade = max(
        [
            float(edge.get("maximum_grade", 0.0) or 0.0)
            for edge in state.get("transport_graph", {}).get("edges", [])
        ],
        default=0.0,
    )
    transitions = sum(
        str(node.get("vertical_mode") or "") == "transition"
        for node in state.get("transport_graph", {}).get("nodes", [])
    )

    return {
        "source": source,
        "sample_id": str(state.get("tile", {}).get("tile_id") or path.parent.name),
        "path": str(path),
        "road_components": road["components"],
        "road_total_length_m": road["total_length_m"],
        "road_largest_length_fraction": road["largest_length_fraction"],
        "road_node_pair_reachability": road["node_pair_reachability"],
        "road_interior_dead_ends": road["interior_dead_ends"],
        "road_interior_component_length_fraction": road[
            "interior_component_length_fraction"
        ],
        "road_assisted_components": road_help["components"],
        "road_assisted_largest_length_fraction": road_help["largest_length_fraction"],
        "road_assisted_interior_component_length_fraction": road_help[
            "interior_component_length_fraction"
        ],
        "surface_road_components": surface["components"],
        "surface_road_largest_length_fraction": surface["largest_length_fraction"],
        "rail_components": rail["components"],
        "rail_total_length_m": rail["total_length_m"],
        "rail_largest_length_fraction": rail["largest_length_fraction"],
        "rail_assisted_components": rail_help["components"],
        "rail_assisted_largest_length_fraction": rail_help["largest_length_fraction"],
        "transition_nodes": transitions,
        "maximum_generated_grade": maximum_grade,
        **building,
        **hierarchy,
    }


def _generated_paths(roots: Iterable[Path]) -> list[Path]:
    result: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("city.json"):
            result[str(path.resolve())] = path.resolve()
    return [result[key] for key in sorted(result)]


def _real_paths(manifests: Iterable[Path]) -> list[Path]:
    result: dict[str, Path] = {}
    for row, manifest in _read_rows(manifests):
        path = _city_json(row, manifest)
        result[str(path.resolve())] = path.resolve()
    return [result[key] for key in sorted(result)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_representatives(
    generated: list[dict[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    if not generated:
        return []
    selectors = [
        ("best_road_connectivity", "road_assisted_largest_length_fraction", max),
        ("worst_road_connectivity", "road_assisted_largest_length_fraction", min),
        ("best_building_access", "buildings_within_20m_road_fraction", max),
        ("worst_building_access", "buildings_within_20m_road_fraction", min),
        (
            "most_interior_disconnect",
            "road_assisted_interior_component_length_fraction",
            max,
        ),
        ("largest_building_blob", "building_area_max_m2", max),
    ]
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    root = output / "representatives"
    root.mkdir(parents=True, exist_ok=True)

    for label, metric, operation in selectors:
        row = operation(generated, key=lambda value: float(value.get(metric, 0.0)))
        key = str(row["path"])
        if key in seen:
            continue
        seen.add(key)
        source_dir = Path(key).parent
        destination = root / f"{len(chosen) + 1:02d}_{label}_{row['sample_id']}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("preview.png", "city.json", "city.obj", "city.mtl"):
            candidate = source_dir / name
            if candidate.exists():
                shutil.copy2(candidate, destination / name)
        chosen.append(
            {
                "label": label,
                "sample_id": row["sample_id"],
                "metric": metric,
                "value": row.get(metric),
                "source": str(source_dir),
            }
        )
    return chosen


def audit(
    generated_roots: Iterable[str | Path],
    real_manifests: Iterable[str | Path],
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

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
    real = [audit_state(path, "real") for path in real_paths]
    rows = [*generated, *real]
    _write_csv(output_path / "tiles.csv", rows)

    generated_summary = _summary(generated)
    real_summary = _summary(real)
    comparison: dict[str, Any] = {}
    for metric, values in generated_summary["metrics"].items():
        if metric not in real_summary["metrics"]:
            continue
        generated_median = values["median"]
        real_median = real_summary["metrics"][metric]["median"]
        comparison[metric] = {
            "generated_median": generated_median,
            "real_median": real_median,
            "ratio": (
                generated_median / real_median if abs(real_median) > 1e-12 else None
            ),
            "difference": generated_median - real_median,
        }

    representatives = _copy_representatives(generated, output_path)
    result = {
        "analysis_version": 1,
        "generated": generated_summary,
        "real": real_summary,
        "comparison": comparison,
        "representatives": representatives,
        "notes": {
            "building_access": (
                "Distance from building footprint to nearest surface-road centreline."
            ),
            "interior_disconnect": (
                "Road-component length fraction in components that do not touch the tile boundary."
            ),
            "hierarchy": (
                "Fraction of local-road length sharing a connected surface component with a major or secondary road."
            ),
            "vertical_assistance": (
                "Evaluation-only joins of surface/elevated or surface/underground endpoints within 1.5 m XY."
            ),
        },
    }
    (output_path / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare generated city-state functionality with real tiles"
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
