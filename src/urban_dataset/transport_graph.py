from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def split_polyline(points: list[list[float]], parts: int) -> list[list[list[float]]]:
    xy = np.asarray([point[:2] for point in points], dtype=np.float64)
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if distances[-1] <= 1e-8:
        raise ValueError("Cannot split a zero-length transport edge")

    def at(distance):
        index = min(int(np.searchsorted(distances, distance, side="right")) - 1, len(xy) - 2)
        index = max(index, 0)
        fraction = (distance - distances[index]) / max(
            distances[index + 1] - distances[index], 1e-12
        )
        left, right = points[index], points[index + 1]
        return [
            float(a + (b - a) * fraction) if a is not None and b is not None else None
            for a, b in zip(left, right, strict=True)
        ]

    result = []
    bounds = np.linspace(0.0, distances[-1], parts + 1)
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        inside = [list(points[i]) for i in range(1, len(points) - 1) if start < distances[i] < end]
        result.append([at(start), *inside, at(end)])
    return result


def simple_transport_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Subdivide loops and parallel edges without removing paths or geometry."""
    nodes = {str(node["id"]): dict(node) for node in graph["nodes"]}
    edges = []
    seen = set()
    added = 0
    for edge in graph["edges"]:
        start, end = str(edge["from_node"]), str(edge["to_node"])
        pair = tuple(sorted((start, end)))
        parts = 3 if start == end else 2 if pair in seen else 1
        seen.add(pair)
        if parts == 1:
            edges.append(dict(edge))
            continue

        pieces = split_polyline(edge["geometry_local_m"], parts)
        ids = [start]
        for index, piece in enumerate(pieces[:-1]):
            node_id = f"shape:{edge['id']}:{index}"
            if node_id in nodes:
                raise ValueError(f"Duplicate shape node: {node_id}")
            nodes[node_id] = {
                "id": node_id,
                "transport_mode": edge["transport_mode"],
                "vertical_mode": edge["vertical_mode"],
                "layer_order": edge.get("layer_order"),
                "position_local_m": piece[-1],
                "boundary_port_key": None,
                "node_type": "shape_anchor",
                "degree": 2,
                "source_edge_id": edge["id"],
            }
            ids.append(node_id)
            added += 1
        ids.append(end)
        for index, piece in enumerate(pieces):
            xy = np.asarray([point[:2] for point in piece])
            edges.append(
                {
                    **edge,
                    "id": f"{edge['id']}:part:{index}",
                    "from_node": ids[index],
                    "to_node": ids[index + 1],
                    "geometry_local_m": piece,
                    "length_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                    "source_edge_id": edge["id"],
                }
            )

    degree = Counter(str(edge[key]) for edge in edges for key in ("from_node", "to_node"))
    for node_id, node in nodes.items():
        node["degree"] = degree[node_id]
    return {
        **graph,
        "nodes": list(nodes.values()),
        "edges": edges,
        "statistics": {
            **graph.get("statistics", {}),
            "nodes": len(nodes),
            "edges": len(edges),
            "shape_anchors": added,
        },
    }
