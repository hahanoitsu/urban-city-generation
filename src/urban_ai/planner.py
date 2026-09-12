from __future__ import annotations

import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from .config import load_config, resolve_path
from .prepare import _largest_surface_road_payload, read_jsonl, write_json
from .scene import compile_generated_city, export_generated_city_obj, render_generated_city
from .schema import city_style


ROAD_WIDTHS_M = {"major": 18.0, "secondary": 12.0, "local": 7.0}


def _near_boundary(x: float, y: float, bounds: list[float], tolerance: float = 1.0) -> bool:
    return (
        abs(x - bounds[0]) <= tolerance
        or abs(x - bounds[2]) <= tolerance
        or abs(y - bounds[1]) <= tolerance
        or abs(y - bounds[3]) <= tolerance
    )


def _profile(payload: dict[str, Any]) -> dict[str, Any]:
    main = _largest_surface_road_payload(payload)
    graph = main.get("transport_graph", {})
    nodes = {str(node["id"]): node for node in graph.get("nodes", [])}
    edges = list(graph.get("edges", []))
    bounds = [
        float(value)
        for value in main.get("coordinate_system", {}).get(
            "local_bounds", [0.0, 0.0, 1024.0, 1024.0]
        )
    ]

    length_m = sum(max(0.0, float(edge.get("length_m", 0.0))) for edge in edges)
    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[str(edge.get("from_node"))] += 1
        degree[str(edge.get("to_node"))] += 1

    boundary_endpoints = 0
    interior_dead_ends = 0
    junctions = 0
    for node_id, node in nodes.items():
        value = degree[node_id]
        position = node.get("position_local_m", [0.0, 0.0, 0.0])
        boundary = _near_boundary(float(position[0]), float(position[1]), bounds)
        if value == 1 and boundary:
            boundary_endpoints += 1
        elif value == 1:
            interior_dead_ends += 1
        if value >= 3:
            junctions += 1

    km = max(length_m / 1000.0, 1e-9)
    style = city_style(main)
    return {
        "style": style,
        "road_length_m": length_m,
        "boundary_endpoints": boundary_endpoints,
        "junctions_per_km": junctions / km,
        "interior_dead_ends_per_km": interior_dead_ends / km,
    }


def _load_profiles(manifest: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in read_jsonl(manifest):
        sample_path = (manifest.parent / str(row["sample_path"])).resolve()
        state_path = sample_path.parent / "city.json"
        if not state_path.exists():
            continue
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        profile = _profile(payload)
        if profile["road_length_m"] <= 500.0:
            continue
        profile["tile_id"] = str(row.get("tile_id") or sample_path.parent.name)
        result.append(profile)
    if not result:
        raise RuntimeError(f"No usable road profiles found in {manifest}")
    return result


def _perimeter_point(distance: float, bounds: list[float]) -> tuple[float, float]:
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    perimeter = 2.0 * (width + height)
    value = distance % perimeter
    if value < width:
        return minx + value, miny
    value -= width
    if value < height:
        return maxx, miny + value
    value -= height
    if value < width:
        return maxx - value, maxy
    value -= width
    return minx, maxy - value


def _boundary_points(
    count: int,
    bounds: list[float],
    rng: random.Random,
) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    minx, miny, maxx, maxy = bounds
    perimeter = 2.0 * ((maxx - minx) + (maxy - miny))
    offset = rng.random()
    values = [
        ((index + offset + rng.uniform(-0.22, 0.22)) / count) * perimeter
        for index in range(count)
    ]
    return [_perimeter_point(value, bounds) for value in values]


def _interior_points(
    count: int,
    bounds: list[float],
    rng: random.Random,
    *,
    margin_m: float,
) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    minx, miny, maxx, maxy = bounds
    minx += margin_m
    miny += margin_m
    maxx -= margin_m
    maxy -= margin_m
    side = max(1, int(math.ceil(math.sqrt(count))))
    cell_w = (maxx - minx) / side
    cell_h = (maxy - miny) / side
    cells = [(row, column) for row in range(side) for column in range(side)]
    rng.shuffle(cells)

    points: list[tuple[float, float]] = []
    for row, column in cells[:count]:
        x = minx + (column + 0.5 + rng.uniform(-0.34, 0.34)) * cell_w
        y = miny + (row + 0.5 + rng.uniform(-0.34, 0.34)) * cell_h
        points.append((x, y))
    return points


def _orientation_penalty(angle: float, base: float, strength: float) -> float:
    return 1.0 + max(0.0, strength) * math.sin(2.0 * (angle - base)) ** 2


def _candidate_graph(
    interior: list[tuple[float, float]],
    boundary: list[tuple[float, float]],
    rng: random.Random,
    *,
    orientation_strength: float,
) -> nx.Graph:
    points = interior + boundary
    if len(points) < 3:
        raise ValueError("At least three planner points are required")
    array = np.asarray(points, dtype=float)
    triangulation = Delaunay(array)
    candidate = nx.Graph()
    for index, point in enumerate(points):
        candidate.add_node(
            index,
            x=float(point[0]),
            y=float(point[1]),
            boundary=index >= len(interior),
        )

    pairs: set[tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        values = [int(value) for value in simplex]
        for left, right in ((values[0], values[1]), (values[1], values[2]), (values[2], values[0])):
            if left == right:
                continue
            pairs.add((min(left, right), max(left, right)))

    base_angle = rng.uniform(0.0, math.pi / 2.0)
    for left, right in sorted(pairs):
        if candidate.nodes[left]["boundary"] and candidate.nodes[right]["boundary"]:
            continue
        x1, y1 = points[left]
        x2, y2 = points[right]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.atan2(y2 - y1, x2 - x1)
        weight = (
            length
            * _orientation_penalty(angle, base_angle, orientation_strength)
            * rng.uniform(0.92, 1.08)
        )
        candidate.add_edge(left, right, length_m=length, weight=weight)
    return candidate


def _dead_ends(graph: nx.Graph, boundary_nodes: set[int]) -> int:
    return sum(graph.degree(node) == 1 and node not in boundary_nodes for node in graph.nodes)


def _junctions(graph: nx.Graph) -> int:
    return sum(graph.degree(node) >= 3 for node in graph.nodes)


def _total_length(graph: nx.Graph) -> float:
    return sum(float(data["length_m"]) for *_ends, data in graph.edges(data=True))


def _build_topology(
    profile: dict[str, Any],
    bounds: list[float],
    rng: random.Random,
    *,
    orientation_strength: float,
    margin_m: float,
    maximum_attempts: int,
) -> nx.Graph:
    target_length = max(1500.0, float(profile["road_length_m"]))
    boundary_count = max(0, min(24, int(round(float(profile["boundary_endpoints"])))))
    target_dead_ends = max(
        0,
        int(round(float(profile["interior_dead_ends_per_km"]) * target_length / 1000.0)),
    )
    target_junctions = max(
        2,
        int(round(float(profile["junctions_per_km"]) * target_length / 1000.0)),
    )

    # Euclidean MST length in a fixed-area tile scales roughly with sqrt(N).
    # Sparse real tiles can legitimately have only ~1.5-2 km of main-road
    # network, so a hard floor of 24 interior points makes their target
    # impossible: the spanning tree alone is already several kilometres.
    count = int(np.clip((target_length / 900.0) ** 2, 4, 180))
    last_error: Exception | None = None
    best_graph: nx.Graph | None = None
    best_score = math.inf
    best_diagnostic: dict[str, float | int] | None = None

    for attempt in range(maximum_attempts):
        interior = _interior_points(count, bounds, rng, margin_m=margin_m)
        boundary = _boundary_points(boundary_count, bounds, rng)
        try:
            candidate = _candidate_graph(
                interior,
                boundary,
                rng,
                orientation_strength=orientation_strength,
            )
            interior_nodes = list(range(len(interior)))
            interior_graph = candidate.subgraph(interior_nodes).copy()
            if not nx.is_connected(interior_graph):
                raise RuntimeError("Interior Delaunay graph is disconnected")

            selected = nx.minimum_spanning_tree(interior_graph, weight="weight")
            boundary_nodes = set(range(len(interior), len(interior) + len(boundary)))
            for node in sorted(boundary_nodes):
                options = [
                    (float(data["weight"]), neighbour)
                    for neighbour, data in candidate[node].items()
                    if neighbour in interior_nodes
                ]
                if not options:
                    raise RuntimeError("Boundary port has no interior Delaunay edge")
                _weight, neighbour = min(options)
                selected.add_node(node, **candidate.nodes[node])
                data = candidate.get_edge_data(node, neighbour)
                selected.add_edge(node, neighbour, **data)

            for node, data in candidate.nodes(data=True):
                if node in selected:
                    selected.nodes[node].update(data)

            extras = [
                (left, right, data)
                for left, right, data in candidate.edges(data=True)
                if left in interior_nodes
                and right in interior_nodes
                and not selected.has_edge(left, right)
            ]

            while extras:
                current_length = _total_length(selected)
                current_dead = _dead_ends(selected, boundary_nodes)
                current_junctions = _junctions(selected)
                enough = (
                    current_length >= target_length * 0.94
                    and current_dead <= target_dead_ends
                    and current_junctions >= target_junctions * 0.80
                )
                if enough or current_length >= target_length * 1.18:
                    break

                scored = []
                for left, right, data in extras:
                    degree_bonus = (
                        (2 if selected.degree(left) == 1 else 1 if selected.degree(left) == 2 else 0)
                        + (2 if selected.degree(right) == 1 else 1 if selected.degree(right) == 2 else 0)
                    )
                    score = float(data["weight"]) / (1.0 + 0.55 * degree_bonus)
                    scored.append((score, rng.random(), left, right, data))
                _score, _tie, left, right, data = min(scored)
                selected.add_edge(left, right, **data)
                extras = [
                    item
                    for item in extras
                    if not (item[0] == left and item[1] == right)
                ]

            actual_length = _total_length(selected)
            ratio = actual_length / target_length
            actual_dead = _dead_ends(selected, boundary_nodes)
            actual_junctions = _junctions(selected)

            length_error = abs(math.log(max(ratio, 1e-9)))
            dead_error = abs(actual_dead - target_dead_ends) / max(target_dead_ends, 4)
            junction_error = (
                abs(actual_junctions - target_junctions) / max(target_junctions, 4)
            )
            score = length_error + 0.18 * dead_error + 0.18 * junction_error

            if score < best_score:
                best_score = score
                best_graph = selected.copy()
                best_diagnostic = {
                    "attempt": attempt + 1,
                    "target_length_m": target_length,
                    "actual_length_m": actual_length,
                    "length_ratio": ratio,
                    "target_dead_ends": target_dead_ends,
                    "actual_dead_ends": actual_dead,
                    "target_junctions": target_junctions,
                    "actual_junctions": actual_junctions,
                    "interior_points": count,
                }

            print(
                "planner attempt "
                f"{attempt + 1}/{maximum_attempts}: "
                f"points={count} target={target_length:.0f}m "
                f"actual={actual_length:.0f}m ratio={ratio:.3f} "
                f"dead={actual_dead}/{target_dead_ends} "
                f"junctions={actual_junctions}/{target_junctions}",
                flush=True,
            )

            if 0.70 <= ratio <= 1.30:
                return selected

            # Adjust the number of interior sites gently. The square-law comes
            # from the sqrt(N) MST scaling, but cap each adjustment so a single
            # noisy attempt cannot bounce between extremes.
            desired = count / max(ratio, 0.20) ** 2
            lower = max(4, int(math.floor(count * 0.55)))
            upper = min(180, int(math.ceil(count * 1.80)))
            count = int(np.clip(round(desired), lower, upper))
        except Exception as exc:
            last_error = exc
            count = max(4, min(180, count + rng.choice((-4, 4))))
            print(
                f"planner attempt {attempt + 1}/{maximum_attempts}: "
                f"construction error: {exc}",
                flush=True,
            )

    # Do not discard a valid connected planar graph just because its scalar
    # statistics miss the preferred 0.70-1.30 window after the search budget.
    # The downstream structural audit is the authority on whether the sample
    # is scientifically acceptable.
    if best_graph is not None and best_diagnostic is not None:
        ratio = float(best_diagnostic["length_ratio"])
        if 0.50 <= ratio <= 1.80:
            print(
                "planner using best valid topology after search budget: "
                + json.dumps(best_diagnostic, sort_keys=True),
                flush=True,
            )
            return best_graph

        detail = json.dumps(best_diagnostic, sort_keys=True)
        raise RuntimeError(
            "Could not construct planner topology within a defensible road-length "
            f"range. Best attempt: {detail}"
        )

    if last_error is not None:
        raise RuntimeError(f"Could not construct planner topology: {last_error}") from last_error
    raise RuntimeError("Could not construct planner topology: no valid attempts were produced")


def _class_edges(
    graph: nx.Graph,
    style: dict[str, float],
) -> dict[tuple[int, int], str]:
    centrality = nx.edge_betweenness_centrality(graph, normalized=True, weight="length_m")
    values = []
    for left, right, data in graph.edges(data=True):
        key = (min(left, right), max(left, right))
        values.append(
            (
                float(centrality.get((left, right), centrality.get((right, left), 0.0))),
                float(data["length_m"]),
                key,
            )
        )
    values.sort(reverse=True)

    total = sum(length for _score, length, _key in values) or 1.0
    major_target = float(np.clip(style.get("major_fraction", 0.2), 0.05, 0.65)) * total
    secondary_target = float(np.clip(style.get("secondary_fraction", 0.18), 0.04, 0.55)) * total

    result: dict[tuple[int, int], str] = {}
    major = 0.0
    secondary = 0.0
    for _score, length, key in values:
        if major < major_target:
            result[key] = "major"
            major += length
        elif secondary < secondary_target:
            result[key] = "secondary"
            secondary += length
        else:
            result[key] = "local"
    return result


def _state_from_topology(
    graph: nx.Graph,
    profile: dict[str, Any],
    bounds: list[float],
    *,
    seed: int,
) -> dict[str, Any]:
    style = dict(profile["style"])
    classes = _class_edges(graph, style)
    maximum_segment = {"major": 180.0, "secondary": 120.0, "local": 80.0}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    positions: dict[str, tuple[float, float, float]] = {}

    def add_node(x: float, y: float) -> str:
        identifier = f"node_{len(nodes):05d}"
        positions[identifier] = (float(x), float(y), 0.0)
        nodes.append(
            {
                "id": identifier,
                "position_local_m": [float(x), float(y), 0.0],
                "transport_mode": "road",
                "vertical_mode": "surface",
            }
        )
        return identifier

    topology_node_ids = {
        node: add_node(float(data["x"]), float(data["y"]))
        for node, data in graph.nodes(data=True)
    }

    for left, right, data in graph.edges(data=True):
        key = (min(left, right), max(left, right))
        road_class = classes[key]
        x1, y1 = float(graph.nodes[left]["x"]), float(graph.nodes[left]["y"])
        x2, y2 = float(graph.nodes[right]["x"]), float(graph.nodes[right]["y"])
        length = math.hypot(x2 - x1, y2 - y1)
        pieces = max(1, int(math.ceil(length / maximum_segment[road_class])))

        chain = [topology_node_ids[left]]
        for index in range(1, pieces):
            ratio = index / pieces
            chain.append(add_node(x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio))
        chain.append(topology_node_ids[right])

        for first, second in zip(chain[:-1], chain[1:], strict=True):
            p1 = positions[first]
            p2 = positions[second]
            segment_length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            edges.append(
                {
                    "id": f"edge_{len(edges):05d}",
                    "from_node": first,
                    "to_node": second,
                    "transport_mode": "road",
                    "class": road_class,
                    "vertical_mode": "surface",
                    "width_m": ROAD_WIDTHS_M[road_class],
                    "length_m": float(segment_length),
                    "minimum_z_m": 0.0,
                    "maximum_z_m": 0.0,
                    "maximum_grade": 0.0,
                    "z_source": "planner_surface",
                    "geometry_local_m": [list(p1), list(p2)],
                }
            )

    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge["from_node"]] += 1
        degree[edge["to_node"]] += 1
    for node in nodes:
        node["degree"] = degree[node["id"]]
        node["node_type"] = (
            "endpoint"
            if node["degree"] <= 1
            else "intersection"
            if node["degree"] >= 3
            else "continuation"
        )

    return {
        "format": "urban-generated-city",
        "version": "0.1.0",
        "coordinate_system": {
            "units": "metres",
            "axis_convention": "x-east, y-north, z-up",
            "local_bounds": [float(value) for value in bounds],
        },
        "generation": {
            "kind": "hierarchical_stochastic_planner",
            "seed": int(seed),
            "style": style,
            "profile": {
                "road_length_m": float(profile["road_length_m"]),
                "boundary_endpoints": int(profile["boundary_endpoints"]),
                "junctions_per_km": float(profile["junctions_per_km"]),
                "interior_dead_ends_per_km": float(
                    profile["interior_dead_ends_per_km"]
                ),
            },
        },
        "transport_graph": {"nodes": nodes, "edges": edges},
        "statistics": {
            "nodes": len(nodes),
            "edges": len(edges),
            "components": 1,
            "surface_edges": len(edges),
            "underground_edges": 0,
            "elevated_edges": 0,
        },
    }


def plan_from_config(
    config_file: str | Path,
    output_root: str | Path,
    *,
    count: int | None = None,
    seed: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_path, config = load_config(config_file)
    data = config.get("data", {})
    planner = config.get("planner", {})
    manifest = resolve_path(
        config_path,
        data.get("profile_manifest", "data/manifests/corpus-v2/train.jsonl"),
    )
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Planner output is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    profiles = _load_profiles(manifest)
    count = int(count or planner.get("count", 24))
    seed = int(seed if seed is not None else planner.get("seed", 9301))
    bounds = [float(value) for value in planner.get("bounds_m", [0, 0, 1024, 1024])]
    orientation_strength = float(planner.get("orientation_strength", 0.35))
    margin_m = float(planner.get("interior_margin_m", 45.0))
    maximum_attempts = int(planner.get("maximum_attempts", 10))

    results: list[dict[str, Any]] = []
    for index in range(count):
        sample_seed = seed + index
        rng = random.Random(sample_seed)
        profile = profiles[rng.randrange(len(profiles))]
        topology = _build_topology(
            profile,
            bounds,
            rng,
            orientation_strength=orientation_strength,
            margin_m=margin_m,
            maximum_attempts=maximum_attempts,
        )
        state = _state_from_topology(topology, profile, bounds, seed=sample_seed)
        city = compile_generated_city(
            state,
            seed=sample_seed,
            minimum_block_area_m2=float(planner.get("minimum_block_area_m2", 400.0)),
            target_parcel_area_m2=float(planner.get("target_parcel_area_m2", 1200.0)),
            minimum_parcel_area_m2=float(planner.get("minimum_parcel_area_m2", 160.0)),
        )

        sample_dir = output_root / f"sample-{index + 1:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_json(sample_dir / "city.json", city)
        preview = render_generated_city(city, sample_dir / "preview.png")
        obj = export_generated_city_obj(city, sample_dir / "city.obj")
        results.append(
            {
                "index": index,
                "seed": sample_seed,
                "profile_tile": profile["tile_id"],
                "profile": {
                    "road_length_m": profile["road_length_m"],
                    "boundary_endpoints": profile["boundary_endpoints"],
                    "junctions_per_km": profile["junctions_per_km"],
                    "interior_dead_ends_per_km": profile[
                        "interior_dead_ends_per_km"
                    ],
                    "style": profile["style"],
                },
                "city": str(sample_dir / "city.json"),
                "preview": preview["preview"],
                "obj": obj["obj"],
                "statistics": city.get("statistics", {}),
            }
        )
        print(
            f"sample {index + 1}/{count}: profile={profile['tile_id']} "
            f"nodes={city.get('statistics', {}).get('nodes', 0)} "
            f"edges={city.get('statistics', {}).get('edges', 0)}",
            flush=True,
        )

    summary = {
        "format": "hierarchical-stochastic-planner-run",
        "config": str(config_path),
        "profile_manifest": str(manifest),
        "profile_count": len(profiles),
        "output_root": str(output_root),
        "seed": seed,
        "samples": results,
    }
    write_json(output_root / "summary.json", summary)
    return summary
