import pytest

from urban_analysis.connectivity import audit_states


def _node(
    node_id,
    x,
    y,
    *,
    mode="surface",
    transport="road",
    boundary=None,
    degree=1,
):
    return {
        "id": node_id,
        "transport_mode": transport,
        "vertical_mode": mode,
        "layer_order": None,
        "position_projected_m": [float(x), float(y), 0.0],
        "position_local_m": [float(x), float(y), 0.0],
        "boundary_port_key": boundary,
        "node_type": "boundary_port" if boundary else "endpoint",
        "degree": degree,
    }


def _edge(edge_id, left, right, points, *, mode="surface", transport="road"):
    return {
        "id": edge_id,
        "from_node": left,
        "to_node": right,
        "transport_mode": transport,
        "vertical_mode": mode,
        "length_m": 100.0,
        "geometry_local_m": [[float(x), float(y), 0.0] for x, y in points],
    }


def _state(tile_id, nodes, edges):
    return {
        "tile": {"tile_id": tile_id, "city_id": "test", "area_id": "test"},
        "transport_graph": {"nodes": nodes, "edges": edges},
    }


def test_adjacent_tile_boundary_ports_are_stitched():
    first = _state(
        "a",
        [
            _node("a0", 0, 0),
            _node("a1", 100, 0, boundary="shared"),
        ],
        [_edge("ae", "a0", "a1", [(0, 0), (100, 0)])],
    )
    second = _state(
        "b",
        [
            _node("b0", 100, 0, boundary="shared"),
            _node("b1", 200, 0),
        ],
        [_edge("be", "b0", "b1", [(0, 0), (100, 0)])],
    )

    summary, _, _ = audit_states([first, second])
    road = summary["transport"]["road"]

    assert road["strict"]["components"] == 2
    assert road["stitched"]["components"] == 1
    assert road["boundary_ports_stitched"] == 1
    assert road["stitched"]["largest_length_fraction"] == pytest.approx(1.0)


def test_cross_mode_xy_crossing_is_not_a_connection():
    state = _state(
        "cross",
        [
            _node("s0", 0, 50, mode="surface"),
            _node("s1", 100, 50, mode="surface"),
            _node("e0", 50, 0, mode="elevated"),
            _node("e1", 50, 100, mode="elevated"),
        ],
        [
            _edge("surface", "s0", "s1", [(0, 50), (100, 50)], mode="surface"),
            _edge("elevated", "e0", "e1", [(50, 0), (50, 100)], mode="elevated"),
        ],
    )

    summary, _, transitions = audit_states([state])
    road = summary["transport"]["road"]

    assert road["strict"]["components"] == 2
    assert road["transition_assisted"]["components"] == 2
    assert road["cross_mode_xy_crossings_not_joined"] == 1
    assert transitions == []


def test_shared_endpoint_becomes_transition_candidate():
    state = _state(
        "ramp",
        [
            _node("s0", 0, 50, mode="surface"),
            _node("s1", 50, 50, mode="surface"),
            _node("e0", 50, 50, mode="elevated"),
            _node("e1", 100, 50, mode="elevated"),
        ],
        [
            _edge("surface", "s0", "s1", [(0, 50), (50, 50)], mode="surface"),
            _edge("elevated", "e0", "e1", [(50, 50), (100, 50)], mode="elevated"),
        ],
    )

    summary, _, transitions = audit_states([state])
    road = summary["transport"]["road"]

    assert road["strict"]["components"] == 2
    assert road["transition_assisted"]["components"] == 1
    assert road["transition_candidates"] == 1
    assert len(transitions) == 1
    assert {transitions[0]["from_mode"], transitions[0]["to_mode"]} == {
        "surface",
        "elevated",
    }
