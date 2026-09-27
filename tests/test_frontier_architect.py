import torch

from urban_model.frontier_architect import (
    FrontierArchitect,
    FrontierArchitectConfig,
)
from urban_model.frontier_data import (
    FrontierProgramConfig,
    build_frontier_program,
)
from urban_model.frontier_loss import frontier_loss


def make_batch():
    max_nodes = 8
    max_edges = 8
    edge_points = 3
    sample = {
        "node_count": torch.tensor(3),
        "node_xy": torch.tensor(
            [
                [-0.8, -0.8],
                [0.0, 0.6],
                [0.8, -0.2],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        "node_mode": torch.zeros(max_nodes, dtype=torch.long),
        "node_vertical": torch.zeros(max_nodes, dtype=torch.long),
        "node_boundary": torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "edge_count": torch.tensor(3),
        "edge_from": torch.tensor([0, 0, 1, 0, 0, 0, 0, 0]),
        "edge_to": torch.tensor([1, 2, 2, 0, 0, 0, 0, 0]),
        "edge_mode": torch.zeros(max_edges, dtype=torch.long),
        "edge_class": torch.zeros(max_edges, dtype=torch.long),
        "edge_vertical": torch.zeros(max_edges, dtype=torch.long),
        "edge_width": torch.ones(max_edges, 1) * 0.25,
        "edge_shape": torch.zeros(max_edges, edge_points, 2),
    }
    program = build_frontier_program(
        sample,
        FrontierProgramConfig(
            max_steps=32,
            max_nodes=max_nodes,
            curve_points=edge_points,
        ),
    )
    context = {
        "context_cells": torch.randn(4, 7),
        "style": torch.randn(4),
        "controls": torch.randn(4),
        "context_line_points": torch.randn(5, 2, 2),
        "context_line_mode": torch.zeros(5, dtype=torch.long),
        "context_line_class": torch.zeros(5, dtype=torch.long),
        "context_line_vertical": torch.zeros(5, dtype=torch.long),
        "context_line_width": torch.rand(5, 1),
        "context_line_length": torch.rand(5, 1),
        "context_line_padding": torch.zeros(5, dtype=torch.bool),
        "ports": torch.zeros(3, 5),
        "port_mode": torch.zeros(3, dtype=torch.long),
        "port_class": torch.zeros(3, dtype=torch.long),
        "port_vertical": torch.zeros(3, dtype=torch.long),
        "port_padding": torch.zeros(3, dtype=torch.bool),
    }
    batch = {**context, **program}
    return {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def test_frontier_architect_forward_and_loss():
    batch = make_batch()
    config = FrontierArchitectConfig(
        context_dimensions=7,
        style_dimensions=4,
        max_steps=32,
        max_nodes=8,
        context_line_points=2,
        curve_points=3,
        model_dimensions=48,
        heads=4,
        context_layers=1,
        decoder_layers=2,
        feedforward_dimensions=96,
        dropout=0.0,
    )
    model = FrontierArchitect(config)
    length = int(batch["program_length"].max()) - 1
    output = model(batch, input_length=length)

    assert output["op"].shape == (1, length, 7)
    assert output["xy_mean"].shape == (1, length, 2)
    assert output["curve_mean"].shape == (1, length, 3)
    assert output["pointer"].shape == (1, length, 8)

    loss, metrics = frontier_loss(output, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert "op" in metrics
    assert "pointer" in metrics
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []
