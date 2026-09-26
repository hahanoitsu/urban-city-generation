import torch

from urban_model.spatial_world import SpatialWorldArchitect, SpatialWorldModelConfig
from urban_model.spatial_world_loss import spatial_world_loss


def test_spatial_world_architect_forward_and_loss():
    config = SpatialWorldModelConfig(
        context_dimensions=8,
        style_dimensions=5,
        max_nodes=12,
        max_edges=16,
        context_line_points=4,
        edge_shape_points=3,
        model_dimensions=64,
        latent_dimensions=24,
        heads=4,
        context_layers=2,
        target_layers=2,
        node_layers=2,
        edge_layers=2,
        feedforward_dimensions=128,
    )
    model = SpatialWorldArchitect(config)
    batch_size = 2
    batch = {
        "context_cells": torch.randn(batch_size, 9, 8),
        "style": torch.randn(batch_size, 5),
        "controls": torch.randn(batch_size, 5),
        "context_line_points": torch.randn(batch_size, 10, 4, 2),
        "context_line_mode": torch.randint(0, 2, (batch_size, 10)),
        "context_line_class": torch.randint(0, 8, (batch_size, 10)),
        "context_line_vertical": torch.randint(0, 4, (batch_size, 10)),
        "context_line_width": torch.rand(batch_size, 10, 1),
        "context_line_length": torch.rand(batch_size, 10, 1),
        "context_line_padding": torch.zeros(batch_size, 10, dtype=torch.bool),
        "ports": torch.randn(batch_size, 6, 5),
        "port_mode": torch.randint(0, 2, (batch_size, 6)),
        "port_class": torch.randint(0, 8, (batch_size, 6)),
        "port_vertical": torch.randint(0, 4, (batch_size, 6)),
        "port_padding": torch.zeros(batch_size, 6, dtype=torch.bool),
        "node_count": torch.tensor([7, 9]),
        "node_xy": torch.rand(batch_size, 12, 2) * 2.0 - 1.0,
        "node_mode": torch.randint(0, 2, (batch_size, 12)),
        "node_vertical": torch.randint(0, 4, (batch_size, 12)),
        "node_boundary": torch.randint(0, 2, (batch_size, 12)).float(),
        "edge_count": torch.tensor([10, 13]),
        "edge_from": torch.randint(0, 7, (batch_size, 16)),
        "edge_to": torch.randint(0, 7, (batch_size, 16)),
        "edge_mode": torch.randint(0, 2, (batch_size, 16)),
        "edge_class": torch.randint(0, 8, (batch_size, 16)),
        "edge_vertical": torch.randint(0, 4, (batch_size, 16)),
        "edge_width": torch.rand(batch_size, 16, 1),
        "edge_shape": torch.rand(batch_size, 16, 3, 2) - 0.5,
    }
    output = model(batch)
    assert output["node_count"].shape == (batch_size, 13)
    assert output["node_xy"].shape == (batch_size, 12, 2)
    assert output["edge_count"].shape == (batch_size, 17)
    assert output["edge_from"].shape == (batch_size, 16, 12)
    assert output["edge_shape"].shape == (batch_size, 16, 3, 2)
    loss, metrics = spatial_world_loss(
        output,
        batch,
        max_nodes=12,
        max_edges=16,
        kl_weight=0.01,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "kl" in metrics
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []
