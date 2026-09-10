from __future__ import annotations

from urban_analysis.structural_generator_audit import _crossing_metrics


def _edge(left, right, a, b):
    return {
        "from_node": left,
        "to_node": right,
        "geometry_local_m": [[a[0], a[1], 0.0], [b[0], b[1], 0.0]],
    }


def test_unnoded_surface_crossing_is_detected():
    edges = [
        _edge("a", "b", (0, 5), (10, 5)),
        _edge("c", "d", (5, 0), (5, 10)),
    ]
    result = _crossing_metrics(edges)
    assert result["unnoded_crossings"] == 1


def test_shared_intersection_node_is_not_unnoded_crossing():
    edges = [
        _edge("a", "x", (0, 5), (5, 5)),
        _edge("x", "b", (5, 5), (10, 5)),
        _edge("c", "x", (5, 0), (5, 5)),
        _edge("x", "d", (5, 5), (5, 10)),
    ]
    result = _crossing_metrics(edges)
    assert result["unnoded_crossings"] == 0
