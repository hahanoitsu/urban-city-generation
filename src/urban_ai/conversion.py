from __future__ import annotations

import math
from collections import deque
from typing import Any

import networkx as nx
from shapely.geometry import LineString

from .schema import (
    GRAPH_PROGRAM_VERSION,
    ProgramConfig,
    city_style,
    normalise_class,
    normalise_mode,
    normalise_vertical,
    quantise,
    quantise_layer,
    quantise_width,
)
from .validation import validate_program


def _edge_priority(data: dict[str, Any]) -> tuple[int, int, float, int, str]:
    mode = normalise_mode(data.get("transport_mode"))
    edge_class = normalise_class(data.get("class"), mode)
    class_order = {
        "major": 0,
        "secondary": 1,
        "local": 2,
        "rail": 0,
        "subway": 1,
        "light_rail": 2,
        "tram": 3,
    }
    vertical_order = {"surface": 0, "elevated": 1, "underground": 2, "unknown": 3}
    return (
        0 if mode == "road" else 1,
        class_order.get(edge_class, 9),
        -float(data.get("length_m", 0.0)),
        vertical_order.get(normalise_vertical(data.get("vertical_mode")), 9),
        str(data.get("source_edge_id", "")),
    )


def _expanded_graph(payload: dict[str, Any], config: ProgramConfig) -> nx.MultiGraph:
    source_graph = payload.get("transport_graph", {})
    source_nodes = {
        str(node.get("id")): node
        for node in source_graph.get("nodes", [])
        if node.get("id") is not None
    }
    graph = nx.MultiGraph()
    for node_id, node in source_nodes.items():
        position = node.get("position_local_m", [0.0, 0.0, None])
        graph.add_node(node_id, x=float(position[0]), y=float(position[1]), source_node=True)

    for edge_index, edge in enumerate(source_graph.get("edges", [])):
        coordinates = edge.get("geometry_local_m") or []
        if len(coordinates) < 2:
            continue
        line = LineString([(float(point[0]), float(point[1])) for point in coordinates])
        if line.length <= 1e-6:
            continue
        simplified = line.simplify(config.simplify_tolerance_m, preserve_topology=False)
        points = list(simplified.coords)
        if len(points) < 2:
            points = list(line.coords)
        start_id = str(edge.get("from_node"))
        end_id = str(edge.get("to_node"))
        if start_id not in graph:
            graph.add_node(start_id, x=float(points[0][0]), y=float(points[0][1]))
        if end_id not in graph:
            graph.add_node(end_id, x=float(points[-1][0]), y=float(points[-1][1]))

        path_ids = [start_id]
        for point_index, point in enumerate(points[1:-1], start=1):
            point_id = f"shape:{edge.get('id', edge_index)}:{point_index}"
            graph.add_node(point_id, x=float(point[0]), y=float(point[1]), source_node=False)
            path_ids.append(point_id)
        path_ids.append(end_id)

        mode = normalise_mode(edge.get("transport_mode"))
        base_data = {
            "transport_mode": mode,
            "class": normalise_class(edge.get("class"), mode),
            "vertical_mode": normalise_vertical(edge.get("vertical_mode")),
            "layer_order": edge.get("layer_order"),
            "width_m": float(edge.get("width_m", 5.0)),
            "source_edge_id": str(edge.get("id", edge_index)),
        }
        for segment_index, (left, right) in enumerate(zip(path_ids, path_ids[1:])):
            x1, y1 = graph.nodes[left]["x"], graph.nodes[left]["y"]
            x2, y2 = graph.nodes[right]["x"], graph.nodes[right]["y"]
            length = math.hypot(x2 - x1, y2 - y1)
            if left == right or length <= 1e-6:
                continue
            graph.add_edge(
                left,
                right,
                key=f"{base_data['source_edge_id']}:{segment_index}",
                **base_data,
                length_m=length,
            )
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def _root_node(graph: nx.MultiGraph, component: set[str], bounds: list[float]) -> str:
    center_x = (bounds[0] + bounds[2]) / 2.0
    center_y = (bounds[1] + bounds[3]) / 2.0

    def score(node_id: str) -> tuple[float, float, float, str]:
        incident = [data for *_rest, data in graph.edges(node_id, data=True, keys=True)]
        class_weight = sum(
            {"major": 3.0, "secondary": 2.0, "local": 1.0}.get(data.get("class"), 1.5)
            for data in incident
        )
        node = graph.nodes[node_id]
        distance = math.hypot(node["x"] - center_x, node["y"] - center_y)
        return (class_weight, float(graph.degree(node_id)), -distance, node_id)

    return max(component, key=score)


def _component_priority(graph: nx.MultiGraph, component: set[str]) -> tuple[int, float, str]:
    edges = [data for *_rest, data in graph.subgraph(component).edges(data=True, keys=True)]
    has_road = any(edge.get("transport_mode") == "road" for edge in edges)
    length = sum(float(edge.get("length_m", 0.0)) for edge in edges)
    return (0 if has_road else 1, -length, min(component))


def _edge_attributes(data: dict[str, Any], config: ProgramConfig) -> dict[str, Any]:
    mode = normalise_mode(data.get("transport_mode"))
    return {
        "transport_mode": mode,
        "class": normalise_class(data.get("class"), mode),
        "width_bin": quantise_width(float(data.get("width_m", 5.0)), config),
        "vertical_mode": normalise_vertical(data.get("vertical_mode")),
        "layer_bin": quantise_layer(data.get("layer_order"), config),
    }


def city_state_to_program(
    payload: dict[str, Any],
    config: ProgramConfig | None = None,
) -> dict[str, Any]:
    config = config or ProgramConfig()
    bounds = [
        float(value)
        for value in payload.get("coordinate_system", {}).get(
            "local_bounds", [0.0, 0.0, 1024.0, 1024.0]
        )
    ]
    graph = _expanded_graph(payload, config)
    commands: list[dict[str, Any]] = []
    program_index: dict[str, int] = {}
    used_edges: set[tuple[str, str, str]] = set()
    components = [set(component) for component in nx.connected_components(graph)]
    components.sort(key=lambda component: _component_priority(graph, component))

    for component in components:
        root = _root_node(graph, component, bounds)
        root_index = len(program_index)
        program_index[root] = root_index
        root_node = graph.nodes[root]
        incident = [data for *_rest, data in graph.edges(root, data=True, keys=True)]
        signature = _edge_attributes(min(incident, key=_edge_priority), config)
        commands.append(
            {
                "op": "root",
                "node": root_index,
                "x_bin": quantise(root_node["x"], bounds[0], bounds[2], config.coordinate_bins),
                "y_bin": quantise(root_node["y"], bounds[1], bounds[3], config.coordinate_bins),
                "transport_mode": signature["transport_mode"],
                "vertical_mode": signature["vertical_mode"],
                "layer_bin": signature["layer_bin"],
            }
        )

        queue: deque[str] = deque([root])
        visited = {root}
        while queue:
            current = queue.popleft()
            neighbours: list[tuple[tuple[Any, ...], str, str, dict[str, Any]]] = []
            for left, right, key, data in graph.edges(current, data=True, keys=True):
                neighbour = right if left == current else left
                neighbours.append((_edge_priority(data), str(neighbour), str(key), data))
            neighbours.sort(key=lambda item: (item[0], item[1], item[2]))
            for _priority, neighbour, edge_key, data in neighbours:
                canonical = tuple(sorted((str(current), str(neighbour)))) + (edge_key,)
                if canonical in used_edges or neighbour in visited:
                    continue
                used_edges.add(canonical)
                visited.add(neighbour)
                queue.append(neighbour)
                node_index = len(program_index)
                program_index[neighbour] = node_index
                node = graph.nodes[neighbour]
                commands.append(
                    {
                        "op": "add",
                        "node": node_index,
                        "parent": program_index[current],
                        "x_bin": quantise(node["x"], bounds[0], bounds[2], config.coordinate_bins),
                        "y_bin": quantise(node["y"], bounds[1], bounds[3], config.coordinate_bins),
                        **_edge_attributes(data, config),
                    }
                )

        closure_edges: list[tuple[int, int, tuple[Any, ...], str, dict[str, Any]]] = []
        for left, right, key, data in graph.subgraph(component).edges(data=True, keys=True):
            canonical = tuple(sorted((str(left), str(right)))) + (str(key),)
            if canonical in used_edges:
                continue
            closure_edges.append(
                (
                    program_index[str(left)],
                    program_index[str(right)],
                    _edge_priority(data),
                    str(key),
                    data,
                )
            )
        closure_edges.sort(
            key=lambda item: (min(item[0], item[1]), max(item[0], item[1]), item[2], item[3])
        )
        for left_index, right_index, _priority, _key, data in closure_edges:
            if left_index == right_index:
                continue
            commands.append(
                {
                    "op": "connect",
                    "from": left_index,
                    "to": right_index,
                    **_edge_attributes(data, config),
                }
            )

    source = payload.get("tile", {})
    program = {
        "format": "urban-graph-program",
        "version": GRAPH_PROGRAM_VERSION,
        "bounds_m": bounds,
        "source": {
            "city_id": source.get("city_id"),
            "area_id": source.get("area_id"),
            "tile_id": source.get("tile_id"),
        },
        "program_config": config.to_dict(),
        "style": city_style(payload),
        "commands": commands,
    }
    validate_program(program, config)
    return program
