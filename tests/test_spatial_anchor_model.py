import torch

from urban_model.spatial_anchor import SpatialAnchorArchitect, SpatialAnchorModelConfig
from urban_model.spatial_anchor_loss import spatial_anchor_loss


def test_spatial_anchor_architect_forward_and_loss():
    config = SpatialAnchorModelConfig(
        context_dimensions=8,
        style_dimensions=5,
        grid_size=4,
        slots_per_cell=3,
        max_active_nodes=12,
        max_edges=16,
        context_line_points=4,
        edge_shape_points=3,
        model_dimensions=64,
        latent_dimensions=12,
        heads=4,
        context_layers=2,
        cell_layers=2,
        feedforward_dimensions=128,
        edge_dimensions=24,
    )
    model = SpatialAnchorArchitect(config)
    batch_size = 2
    cells = 16
    slots = 3
    active = 12
    edges = 16
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
        "cell_count": torch.zeros(batch_size, cells, dtype=torch.long),
        "slot_presence": torch.zeros(batch_size, cells, slots),
        "node_offset": torch.zeros(batch_size, cells, slots, 2),
        "node_mode": torch.zeros(batch_size, cells, slots, dtype=torch.long),
        "node_vertical": torch.zeros(batch_size, cells, slots, dtype=torch.long),
        "node_boundary": torch.zeros(batch_size, cells, slots),
        "active_count": torch.tensor([4, 5]),
        "active_anchor_ids": torch.zeros(batch_size, active, dtype=torch.long),
        "node_degree": torch.zeros(batch_size, active, dtype=torch.long),
        "edge_count": torch.tensor([3, 4]),
        "edge_pairs": torch.zeros(batch_size, edges, 2, dtype=torch.long),
        "edge_class": torch.zeros(batch_size, edges, dtype=torch.long),
        "edge_vertical": torch.zeros(batch_size, edges, dtype=torch.long),
        "edge_width": torch.rand(batch_size, edges, 1),
        "edge_curve": torch.rand(batch_size, edges, 3) - 0.5,
    }

    for batch_index, count in enumerate((4, 5)):
        ids = torch.arange(count)
        batch["active_anchor_ids"][batch_index, :count] = ids
        for anchor in ids:
            cell = int(anchor) // slots
            slot = int(anchor) % slots
            batch["slot_presence"][batch_index, cell, slot] = 1.0
            batch["cell_count"][batch_index, cell] += 1
        for edge_index in range(count - 1):
            batch["edge_pairs"][batch_index, edge_index] = torch.tensor(
                [edge_index, edge_index + 1]
            )
            batch["node_degree"][batch_index, edge_index] += 1
            batch["node_degree"][batch_index, edge_index + 1] += 1

    output = model(batch)
    model.eval()
    deterministic_a = model(batch, sample_latent=False)
    deterministic_b = model(batch, sample_latent=False)
    model.train()
    assert torch.equal(
        deterministic_a["node_offset"],
        deterministic_b["node_offset"],
    )
    assert output["global_node_count"].shape == (batch_size,)
    assert output["global_edge_count"].shape == (batch_size,)
    assert output["cell_occupancy"].shape == (batch_size, cells)
    assert output["cell_count"].shape == (batch_size, cells, slots)
    assert output["slot_score"].shape == (batch_size, cells, slots)
    assert output["node_offset"].shape == (batch_size, cells, slots, 2)
    assert output["node_degree"].shape == (
        batch_size,
        active,
        config.max_degree + 1,
    )
    assert output["edge_exists"].shape == (batch_size, active, active)
    assert output["edge_class"].shape == (batch_size, active, active, 8)
    assert output["edge_curve"].shape == (
        batch_size,
        active,
        active,
        config.edge_shape_points,
    )

    loss, metrics = spatial_anchor_loss(
        output,
        batch,
        kl_weight=0.01,
        max_active_nodes=12,
        max_edges=16,
    )
    model.eval()
    generated = model.generate(batch, temperature=0.0)
    assert generated["active_anchor_ids"].shape == (batch_size, 12)
    assert generated["active_count"].shape == (batch_size,)
    assert generated["edge_exists"].shape == (batch_size, 12, 12)
    assert generated["edge_class"].shape == (batch_size, 12, 12, 8)
    assert generated["predicted_edge_count"].shape == (batch_size,)
    assert torch.all(model._decode_count(torch.zeros(2), 12) > 0)
    model.train()
    loss.backward()
    assert torch.isfinite(loss)
    assert "edge_exists" in metrics
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []



def test_spatial_anchor_prior_overfits_nonempty_graph():
    torch.manual_seed(7)
    config = SpatialAnchorModelConfig(
        context_dimensions=4,
        style_dimensions=2,
        grid_size=2,
        slots_per_cell=2,
        max_active_nodes=8,
        max_edges=8,
        context_line_points=2,
        edge_shape_points=2,
        model_dimensions=32,
        latent_dimensions=8,
        heads=4,
        context_layers=1,
        cell_layers=1,
        feedforward_dimensions=64,
        edge_dimensions=16,
        dropout=0.0,
    )
    model = SpatialAnchorArchitect(config)
    batch = {
        "context_cells": torch.randn(1, 4, 4),
        "style": torch.randn(1, 2),
        "controls": torch.randn(1, 2),
        "context_line_points": torch.randn(1, 2, 2, 2),
        "context_line_mode": torch.zeros(1, 2, dtype=torch.long),
        "context_line_class": torch.zeros(1, 2, dtype=torch.long),
        "context_line_vertical": torch.zeros(1, 2, dtype=torch.long),
        "context_line_width": torch.ones(1, 2, 1),
        "context_line_length": torch.ones(1, 2, 1),
        "context_line_padding": torch.zeros(1, 2, dtype=torch.bool),
        "ports": torch.zeros(1, 2, 5),
        "port_mode": torch.zeros(1, 2, dtype=torch.long),
        "port_class": torch.zeros(1, 2, dtype=torch.long),
        "port_vertical": torch.zeros(1, 2, dtype=torch.long),
        "port_padding": torch.zeros(1, 2, dtype=torch.bool),
        "cell_count": torch.tensor([[2, 2, 2, 0]]),
        "slot_presence": torch.tensor(
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]
        ),
        "node_offset": torch.zeros(1, 4, 2, 2),
        "node_mode": torch.zeros(1, 4, 2, dtype=torch.long),
        "node_vertical": torch.zeros(1, 4, 2, dtype=torch.long),
        "node_boundary": torch.zeros(1, 4, 2),
        "active_count": torch.tensor([6]),
        "active_anchor_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 0, 0]]),
        "node_degree": torch.tensor([[1, 2, 2, 2, 2, 1, 0, 0]]),
        "edge_count": torch.tensor([5]),
        "edge_pairs": torch.tensor(
            [[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 0], [0, 0], [0, 0]]]
        ),
        "edge_class": torch.zeros(1, 8, dtype=torch.long),
        "edge_vertical": torch.zeros(1, 8, dtype=torch.long),
        "edge_width": torch.ones(1, 8, 1) * 0.2,
        "edge_curve": torch.zeros(1, 8, 2),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch,
            use_posterior=False,
            sample_latent=False,
        )
        loss, _metrics = spatial_anchor_loss(
            output,
            batch,
            kl_weight=0.0,
            max_active_nodes=8,
            max_edges=8,
        )
        loss.backward()
        optimizer.step()

    model.eval()
    generated = model.generate(batch, temperature=0.0)
    assert abs(int(generated["active_count"][0]) - 6) <= 1
    assert abs(int(generated["predicted_edge_count"][0]) - 5) <= 1
    selected = generated["active_anchor_ids"][0, : int(generated["active_count"][0])]
    assert len(torch.unique(selected)) == len(selected)
