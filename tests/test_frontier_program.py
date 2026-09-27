import torch

from urban_model.frontier_data import (
    OP_BOS,
    OP_CLOSE,
    OP_EOS,
    OP_GROW,
    OP_LINK,
    OP_ROOT,
    FrontierProgramConfig,
    build_frontier_program,
)


def make_graph_sample():
    max_nodes = 8
    max_edges = 8
    edge_points = 3
    node_xy = torch.zeros(max_nodes, 2)
    node_xy[:3] = torch.tensor(
        [
            [-0.8, -0.8],
            [0.0, 0.6],
            [0.8, -0.2],
        ]
    )
    edge_from = torch.zeros(max_edges, dtype=torch.long)
    edge_to = torch.zeros(max_edges, dtype=torch.long)
    edge_from[:3] = torch.tensor([0, 0, 1])
    edge_to[:3] = torch.tensor([1, 2, 2])
    return {
        "node_count": torch.tensor(3),
        "node_xy": node_xy,
        "node_mode": torch.zeros(max_nodes, dtype=torch.long),
        "node_vertical": torch.zeros(max_nodes, dtype=torch.long),
        "node_boundary": torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "edge_count": torch.tensor(3),
        "edge_from": edge_from,
        "edge_to": edge_to,
        "edge_mode": torch.zeros(max_edges, dtype=torch.long),
        "edge_class": torch.zeros(max_edges, dtype=torch.long),
        "edge_vertical": torch.zeros(max_edges, dtype=torch.long),
        "edge_width": torch.ones(max_edges, 1) * 0.25,
        "edge_shape": torch.zeros(max_edges, edge_points, 2),
    }


def test_frontier_program_reconstructs_graph_actions():
    sample = make_graph_sample()
    config = FrontierProgramConfig(
        max_steps=32,
        max_nodes=8,
        curve_points=3,
    )
    program = build_frontier_program(sample, config)
    length = int(program["program_length"])
    ops = program["program_op"][:length].tolist()

    assert ops[0] == OP_BOS
    assert ops[-1] == OP_EOS
    assert ops.count(OP_ROOT) == 1
    assert ops.count(OP_GROW) == 2
    assert ops.count(OP_LINK) == 1
    assert ops.count(OP_CLOSE) == 3
    assert int(program["program_nodes"]) == 3
    assert int(program["program_edges"]) == 3

    link_index = ops.index(OP_LINK)
    assert int(program["program_pointer"][link_index]) in {1, 2}
