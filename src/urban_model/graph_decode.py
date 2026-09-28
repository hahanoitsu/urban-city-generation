from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


def learned_degree_edges(output, node_count, time_limit=5.0):
    if node_count < 2:
        return [], {"method": "learned_degree", "status": "empty", "optimal": True}

    logits = output["edge_exists"][0, :node_count, :node_count].detach().float().cpu().numpy()
    degree_logp = (
        output["node_degree"][0, :node_count].detach().float().log_softmax(-1).cpu().numpy()
    )
    modes = output["node_mode"][0, :node_count].detach().argmax(-1).cpu().numpy()
    if not np.isfinite(logits).all() or not np.isfinite(degree_logp).all():
        raise ValueError("Non-finite edge or degree predictions")
    left, right = np.triu_indices(node_count, 1)
    same_mode = modes[left] == modes[right]
    left, right = left[same_mode], right[same_mode]
    edge_variables = len(left)
    degree_states = degree_logp.shape[1]
    caps = degree_logp.argmax(axis=1)
    expected_degree = (np.exp(degree_logp) * np.arange(degree_states)).sum(axis=1)
    expected_edges = max(float(expected_degree.sum() / 2), 1.0)
    all_pairs = node_count * (node_count - 1) / 2
    # Training used a per-sample BCE weight. Estimate it from predicted degrees.
    positive_weight = np.clip((all_pairs - expected_edges) / expected_edges, 1.0, 80.0)
    scores = logits[left, right].astype(np.float64) - np.log(positive_weight)

    # Each node selects one degree state. Its incident edges must match that state.
    variable_count = edge_variables + node_count * degree_states
    edge_index = np.arange(edge_variables)
    degree_index = np.arange(edge_variables, variable_count)
    node_index = np.repeat(np.arange(node_count), degree_states)
    degree_value = np.tile(np.arange(degree_states), node_count)
    upper = np.r_[np.ones(edge_variables), degree_value <= caps[node_index]]
    rows = np.r_[left, right, node_index, node_count + node_index]
    columns = np.r_[edge_index, edge_index, degree_index, degree_index]
    values = np.r_[np.ones(edge_variables * 2), -degree_value, np.ones(len(degree_index))]
    matrix = coo_array((values, (rows, columns)), shape=(node_count * 2, variable_count)).tocsc()
    target = np.r_[np.zeros(node_count), np.ones(node_count)]
    cost = np.r_[-scores, -degree_logp.ravel().astype(np.float64)]
    result = milp(
        cost,
        integrality=np.ones(variable_count),
        bounds=Bounds(0, upper),
        constraints=LinearConstraint(matrix, target, target),
        options={"time_limit": time_limit, "mip_rel_gap": 0.01},
    )
    status = str(result.message)
    valid = result.x is not None
    if valid:
        rounded = np.rint(result.x)
        valid = bool(
            np.all((rounded >= 0) & (rounded <= upper))
            and np.allclose(result.x, rounded, atol=1e-5)
            and np.allclose(matrix @ rounded, target, atol=1e-5)
        )
    if valid:
        selected = np.flatnonzero(rounded[:edge_variables])
    else:
        # A timeout without a feasible solution must not trigger forced edge filling.
        degrees = np.zeros(node_count, dtype=int)
        selected = []
        for index in np.argsort(-scores, kind="stable"):
            a, b = left[index], right[index]
            if scores[index] > 0 and degrees[a] < caps[a] and degrees[b] < caps[b]:
                selected.append(index)
                degrees[a] += 1
                degrees[b] += 1
        status = f"fallback: {status}"

    pairs = [(int(left[index]), int(right[index])) for index in selected]
    degrees = np.bincount(np.asarray(pairs, dtype=int).ravel(), minlength=node_count)
    return pairs, {
        "method": "learned_degree",
        "status": status,
        "optimal": bool(result.success and valid),
        "gap": float(result.mip_gap)
        if valid and getattr(result, "mip_gap", None) is not None
        else None,
        "expected_edges_from_degrees": expected_edges,
        "edge_positive_weight": float(positive_weight),
        "predicted_degrees": caps.tolist(),
        "selected_degrees": degrees.tolist(),
    }
