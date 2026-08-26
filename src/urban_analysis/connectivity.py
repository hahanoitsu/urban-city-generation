from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString


TRANSITION_PAIRS = {
    frozenset(("surface", "elevated")),
    frozenset(("surface", "underground")),
}


def _read_rows(manifests: Iterable[Path]) -> list[tuple[dict[str, Any], Path]]:
    rows: dict[str, tuple[dict[str, Any], Path]] = {}
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows[str(row["tile_id"])] = (row, manifest)
    return [rows[key] for key in sorted(rows)]


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _city_json(row: dict[str, Any], manifest: Path) -> Path:
    for field in ("metadata_path", "sample_path"):
        if field not in row:
            continue
        path = _resolve(manifest.parent, str(row[field]))
        candidate = path.parent / "city.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No city.json beside tile {row.get('tile_id')}")


def _namespace(tile_id: str, node_id: str) -> str:
    return f"{tile_id}:{node_id}"


def _edge_length(graph: nx.Graph, nodes: set[str]) -> float:
    total = 0.0
    for left, right, data in graph.edges(nodes, data=True):
        if left not in nodes or right not in nodes:
            continue
        if data.get("kind") == "transport":
            total += float(data.get("length_m", 0.0))
    return total


def _metrics(graph: nx.Graph, open_boundary_nodes: set[str] | None = None) -> dict[str, float | int]:
    open_boundary_nodes = open_boundary_nodes or set()
    active = graph.copy()
    isolated = list(nx.isolates(active))
    active.remove_nodes_from(isolated)
    if active.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "transport_edges": 0,
            "components": 0,
            "largest_length_fraction": 0.0,
            "node_pair_reachability": 0.0,
            "internal_dead_ends": 0,
            "total_length_m": 0.0,
        }

    components = [set(values) for values in nx.connected_components(active)]
    lengths = [_edge_length(active, values) for values in components]
    total_length = sum(lengths)
    largest = max(lengths, default=0.0)

    node_count = active.number_of_nodes()
    possible_pairs = node_count * (node_count - 1)
    reachable_pairs = sum(len(values) * (len(values) - 1) for values in components)

    internal_dead_ends = sum(
        active.degree(node) == 1 and node not in open_boundary_nodes
        for node in active.nodes
    )

    return {
        "nodes": node_count,
        "transport_edges": sum(
            data.get("kind") == "transport" for _, _, data in active.edges(data=True)
        ),
        "components": len(components),
        "largest_length_fraction": largest / total_length if total_length else 0.0,
        "node_pair_reachability": reachable_pairs / possible_pairs if possible_pairs else 1.0,
        "internal_dead_ends": int(internal_dead_ends),
        "total_length_m": float(total_length),
    }


def _local_metrics(graph: nx.Graph) -> dict[str, float | int]:
    active = graph.copy()
    isolated = list(nx.isolates(active))
    active.remove_nodes_from(isolated)
    if active.number_of_nodes() == 0:
        result = _metrics(active)
        result.update(
            {
                "boundary_touching_components": 0,
                "interior_components": 0,
                "interior_component_length_fraction": 0.0,
            }
        )
        return result

    components = [set(values) for values in nx.connected_components(active)]
    lengths = [_edge_length(active, values) for values in components]
    total_length = sum(lengths)
    boundary = {
        node for node, data in active.nodes(data=True) if data.get("boundary_port_key")
    }
    boundary_components = [values for values in components if values & boundary]
    interior_components = [values for values in components if not values & boundary]
    interior_length = sum(_edge_length(active, values) for values in interior_components)

    result = _metrics(active, boundary)
    result.update(
        {
            "boundary_touching_components": len(boundary_components),
            "interior_components": len(interior_components),
            "interior_component_length_fraction": (
                interior_length / total_length if total_length else 0.0
            ),
        }
    )
    return result


def _tile_graph(state: dict[str, Any], transport_mode: str) -> nx.Graph:
    tile_id = str(state["tile"]["tile_id"])
    source = state["transport_graph"]
    graph = nx.Graph()

    for node in source.get("nodes", []):
        if node.get("transport_mode") != transport_mode:
            continue
        key = _namespace(tile_id, str(node["id"]))
        graph.add_node(key, tile_id=tile_id, **node)

    for edge in source.get("edges", []):
        if edge.get("transport_mode") != transport_mode:
            continue
        left = _namespace(tile_id, str(edge["from_node"]))
        right = _namespace(tile_id, str(edge["to_node"]))
        if left not in graph or right not in graph:
            continue
        graph.add_edge(
            left,
            right,
            kind="transport",
            tile_id=tile_id,
            length_m=float(edge.get("length_m", 0.0)),
            vertical_mode=edge.get("vertical_mode"),
            geometry_local_m=edge.get("geometry_local_m", []),
        )
    return graph


def _transition_candidates(
    graph: nx.Graph,
    tolerance_m: float,
) -> list[dict[str, Any]]:
    nodes = list(graph.nodes(data=True))
    if len(nodes) < 2:
        return []

    positions = np.asarray(
        [data.get("position_projected_m", [np.nan, np.nan])[:2] for _, data in nodes],
        dtype=float,
    )
    valid = np.isfinite(positions).all(axis=1)
    if not valid.any():
        return []

    valid_indexes = np.flatnonzero(valid)
    tree = cKDTree(positions[valid])
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for local_index, neighbours in enumerate(tree.query_ball_tree(tree, tolerance_m)):
        first_index = int(valid_indexes[local_index])
        first_key, first = nodes[first_index]
        for neighbour in neighbours:
            second_index = int(valid_indexes[neighbour])
            if second_index <= first_index:
                continue
            second_key, second = nodes[second_index]
            if first.get("tile_id") != second.get("tile_id"):
                continue
            pair = frozenset((str(first.get("vertical_mode")), str(second.get("vertical_mode"))))
            if pair not in TRANSITION_PAIRS:
                continue
            if int(first.get("degree", graph.degree(first_key))) > 2:
                continue
            if int(second.get("degree", graph.degree(second_key))) > 2:
                continue
            key = tuple(sorted((first_key, second_key)))
            if key in seen:
                continue
            seen.add(key)
            distance = float(np.linalg.norm(positions[first_index] - positions[second_index]))
            candidates.append(
                {
                    "from_node": first_key,
                    "to_node": second_key,
                    "tile_id": first.get("tile_id"),
                    "from_mode": first.get("vertical_mode"),
                    "to_mode": second.get("vertical_mode"),
                    "distance_m": distance,
                }
            )
    return candidates


def _add_transitions(graph: nx.Graph, candidates: list[dict[str, Any]]) -> nx.Graph:
    result = graph.copy()
    for candidate in candidates:
        result.add_edge(
            candidate["from_node"],
            candidate["to_node"],
            kind="transition_candidate",
            length_m=0.0,
            tile_id=candidate["tile_id"],
        )
    return result


def _stitch(graph: nx.Graph) -> tuple[nx.Graph, set[str], int, int]:
    result = graph.copy()
    ports: dict[str, list[str]] = defaultdict(list)
    for node, data in result.nodes(data=True):
        port = data.get("boundary_port_key")
        if port:
            ports[str(port)].append(node)

    open_nodes: set[str] = set()
    stitched_ports = 0
    for values in ports.values():
        tile_ids = {result.nodes[node].get("tile_id") for node in values}
        if len(values) >= 2 and len(tile_ids) >= 2:
            stitched_ports += 1
            anchor = values[0]
            for node in values[1:]:
                result.add_edge(anchor, node, kind="tile_stitch", length_m=0.0)
        else:
            open_nodes.update(values)
    return result, open_nodes, stitched_ports, len(ports) - stitched_ports


def _xy_crossings(state: dict[str, Any], transport_mode: str) -> int:
    graph = state.get("transport_graph", {})
    edges = [
        edge
        for edge in graph.get("edges", [])
        if edge.get("transport_mode") == transport_mode
        and edge.get("vertical_mode") in {"surface", "elevated", "underground"}
        and len(edge.get("geometry_local_m", [])) >= 2
    ]
    if not edges:
        return 0

    lines = [LineString([(p[0], p[1]) for p in edge["geometry_local_m"]]) for edge in edges]
    tree = STRtree(lines)
    count = 0
    for i, line in enumerate(lines):
        mode = edges[i].get("vertical_mode")
        for j in tree.query(line):
            j = int(j)
            if j <= i or edges[j].get("vertical_mode") == mode:
                continue
            intersection = line.intersection(lines[j])
            if intersection.is_empty:
                continue
            # Endpoint-to-endpoint contacts are plausible transitions and are counted
            # separately. This counter is for XY crossings that must not imply a join.
            endpoints_i = [line.coords[0], line.coords[-1]]
            endpoints_j = [lines[j].coords[0], lines[j].coords[-1]]
            endpoint_contact = any(
                LineString([a, b]).length <= 0.05 for a in endpoints_i for b in endpoints_j
            )
            if not endpoint_contact:
                count += 1
    return count


def audit_states(
    states: list[dict[str, Any]],
    *,
    transition_tolerance_m: float = 0.75,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tile_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"tiles": len(states), "transport": {}}

    for transport_mode in ("road", "rail"):
        global_graph = nx.Graph()
        tile_graphs: dict[str, nx.Graph] = {}
        crossings = 0

        for state in states:
            tile_id = str(state["tile"]["tile_id"])
            graph = _tile_graph(state, transport_mode)
            tile_graphs[tile_id] = graph
            global_graph = nx.compose(global_graph, graph)
            crossings += _xy_crossings(state, transport_mode)

        stitched, open_nodes, stitched_ports, open_ports = _stitch(global_graph)
        candidates = _transition_candidates(stitched, transition_tolerance_m)
        assisted = _add_transitions(stitched, candidates)
        transition_rows.extend(
            {"transport_mode": transport_mode, **candidate} for candidate in candidates
        )

        summary["transport"][transport_mode] = {
            "strict": _metrics(global_graph),
            "stitched": _metrics(stitched, open_nodes),
            "transition_assisted": _metrics(assisted, open_nodes),
            "boundary_ports_stitched": stitched_ports,
            "boundary_ports_open": open_ports,
            "transition_candidates": len(candidates),
            "cross_mode_xy_crossings_not_joined": crossings,
        }

        candidates_by_tile: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            candidates_by_tile[str(candidate["tile_id"])].append(candidate)

        for tile_id, graph in tile_graphs.items():
            strict = _local_metrics(graph)
            assisted_local = _local_metrics(
                _add_transitions(graph, candidates_by_tile.get(tile_id, []))
            )
            row = next((value for value in tile_rows if value["tile_id"] == tile_id), None)
            if row is None:
                row = {"tile_id": tile_id}
                tile_rows.append(row)
            for prefix, values in ((f"{transport_mode}_strict", strict), (f"{transport_mode}_assisted", assisted_local)):
                for key, value in values.items():
                    row[f"{prefix}_{key}"] = value
            row[f"{transport_mode}_transition_candidates"] = len(
                candidates_by_tile.get(tile_id, [])
            )

    tile_rows.sort(key=lambda value: value["tile_id"])
    return summary, tile_rows, transition_rows


def audit_manifests(
    manifests: Iterable[str | Path],
    output: str | Path,
    *,
    transition_tolerance_m: float = 0.75,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_paths = [Path(value).expanduser().resolve() for value in manifests]
    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_path}")
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    states: list[dict[str, Any]] = []
    missing: list[str] = []
    for row, manifest in _read_rows(manifest_paths):
        try:
            path = _city_json(row, manifest)
        except FileNotFoundError:
            missing.append(str(row.get("tile_id")))
            continue
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if "transport_graph" not in state:
            missing.append(str(row.get("tile_id")))
            continue
        states.append(state)

    if not states:
        raise FileNotFoundError("No tile city.json files with transport_graph were found")

    summary, tile_rows, transition_rows = audit_states(
        states,
        transition_tolerance_m=transition_tolerance_m,
    )
    summary.update(
        {
            "analysis_version": 1,
            "manifests": [str(path) for path in manifest_paths],
            "tiles_requested": len(_read_rows(manifest_paths)),
            "tiles_loaded": len(states),
            "tiles_missing_city_json": missing,
            "transition_tolerance_m": transition_tolerance_m,
        }
    )

    with (output_path / "tiles.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted(set().union(*(row.keys() for row in tile_rows)))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tile_rows)

    with (output_path / "transition_candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "transport_mode",
            "tile_id",
            "from_node",
            "to_node",
            "from_mode",
            "to_mode",
            "distance_m",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(transition_rows)

    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit real vector transport connectivity")
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transition-tolerance-m", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = audit_manifests(
        args.manifest,
        args.output,
        transition_tolerance_m=args.transition_tolerance_m,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
