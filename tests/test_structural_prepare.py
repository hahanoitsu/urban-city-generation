from __future__ import annotations

from urban_ai.prepare import _largest_surface_road_payload


def _node(name, x, y, mode="road", vertical="surface"):
    return {
        "id": name,
        "transport_mode": mode,
        "vertical_mode": vertical,
        "position_local_m": [float(x), float(y), 0.0],
    }


def _edge(name, left, right, length, edge_class="local", mode="road", vertical="surface"):
    return {
        "id": name,
        "from_node": left,
        "to_node": right,
        "transport_mode": mode,
        "vertical_mode": vertical,
        "class": edge_class,
        "length_m": float(length),
        "width_m": 7.0,
        "geometry_local_m": [[0.0, 0.0, 0.0], [float(length), 0.0, 0.0]],
    }


def test_largest_surface_road_payload_keeps_only_longest_connected_component():
    payload = {
        "transport_graph": {
            "nodes": [
                _node("a", 0, 0),
                _node("b", 100, 0),
                _node("c", 200, 0),
                _node("x", 0, 50),
                _node("y", 20, 50),
                _node("r1", 0, 80, mode="rail"),
                _node("r2", 50, 80, mode="rail"),
            ],
            "edges": [
                _edge("ab", "a", "b", 100, "major"),
                _edge("bc", "b", "c", 100, "secondary"),
                _edge("xy", "x", "y", 20),
                _edge("rail", "r1", "r2", 50, mode="rail"),
            ],
        }
    }

    result = _largest_surface_road_payload(payload)
    graph = result["transport_graph"]

    assert {node["id"] for node in graph["nodes"]} == {"a", "b", "c"}
    assert {edge["id"] for edge in graph["edges"]} == {"ab", "bc"}
    assert {node["id"]: node["degree"] for node in graph["nodes"]} == {
        "a": 1,
        "b": 2,
        "c": 1,
    }


def test_largest_surface_road_payload_rejects_elevated_and_rail_edges():
    payload = {
        "transport_graph": {
            "nodes": [
                _node("a", 0, 0),
                _node("b", 100, 0),
                _node("e1", 0, 20, vertical="elevated"),
                _node("e2", 100, 20, vertical="elevated"),
            ],
            "edges": [
                _edge("ab", "a", "b", 100),
                _edge("e", "e1", "e2", 500, vertical="elevated"),
            ],
        }
    }

    result = _largest_surface_road_payload(payload)
    assert {edge["id"] for edge in result["transport_graph"]["edges"]} == {"ab"}
