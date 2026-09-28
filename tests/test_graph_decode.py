from types import SimpleNamespace

import numpy as np
import torch

from urban_model.graph_decode import learned_degree_edges


def output_for(degrees, scores, modes=None):
    count = len(degrees)
    degree_logits = torch.full((1, count, max(max(degrees) + 1, 4)), -20.0)
    for index, degree in enumerate(degrees):
        degree_logits[0, index, degree] = 20.0
    mode_logits = torch.full((1, count, 2), -10.0)
    for index, mode in enumerate(modes or [0] * count):
        mode_logits[0, index, mode] = 10.0
    return {
        "node_degree": degree_logits,
        "node_mode": mode_logits,
        "edge_exists": torch.tensor(scores, dtype=torch.float32)[None],
    }


def test_degree_evidence_prevents_a_clique_from_using_every_edge():
    scores = np.full((8, 8), -20.0)
    for a in range(4):
        for b in range(a + 1, 4):
            scores[a, b] = 10
        scores[a, a + 4] = 8
    expected = [2, 2, 2, 2, 1, 1, 1, 1]
    pairs, info = learned_degree_edges(output_for(expected, scores), 8)
    assert info["optimal"]
    assert info["selected_degrees"] == expected
    assert all((i, i + 4) in pairs for i in range(4))
    assert len(pairs) == 6


def test_real_triangle_is_allowed_when_both_heads_support_it():
    pairs, info = learned_degree_edges(output_for([2, 2, 2], np.full((3, 3), 10.0)), 3)
    assert set(pairs) == {(0, 1), (0, 2), (1, 2)}
    assert info["selected_degrees"] == [2, 2, 2]


def test_large_edge_scores_cannot_override_predicted_degree_caps():
    output = output_for([2] * 12, np.full((12, 12), 50.0))
    output["node_degree"] = torch.full((1, 12, 9), -5.0)
    output["node_degree"][:, :, 2] = 5.0
    pairs, info = learned_degree_edges(output, 12)
    assert len(pairs) == 12
    assert info["selected_degrees"] == [2] * 12


def test_edges_are_not_forced_when_edge_evidence_rejects_them():
    pairs, info = learned_degree_edges(output_for([1, 1], np.full((2, 2), -200.0)), 2)
    assert pairs == []
    assert info["selected_degrees"] == [0, 0]


def test_road_and_rail_nodes_remain_separate():
    scores = np.full((4, 4), 50.0)
    pairs, _ = learned_degree_edges(output_for([1, 1, 1, 1], scores, [0, 0, 1, 1]), 4)
    assert set(pairs) == {(0, 1), (2, 3)}


def test_timeout_fallback_does_not_exceed_predicted_degrees(monkeypatch):
    monkeypatch.setattr(
        "urban_model.graph_decode.milp",
        lambda *args, **kwargs: SimpleNamespace(x=None, success=False, message="test timeout"),
    )
    pairs, info = learned_degree_edges(output_for([1, 1, 1, 1], np.full((4, 4), 10.0)), 4)
    assert len(pairs) == 2
    assert info["selected_degrees"] == [1, 1, 1, 1]
    assert info["status"].startswith("fallback")
