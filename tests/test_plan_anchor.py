import torch

from urban_model.plan_anchor import (
    PlanAnchorArchitect,
    PlanAnchorModelConfig,
)
from urban_model.plan_anchor_loss import plan_anchor_loss


def make_batch():
    batch_size = 2
    plan_cells = 16
    anchor_cells = 16
    slots = 2
    max_active = 8
    max_edges = 8
    plan_dimensions = 8
    orientation_dimensions = 4

    slot_presence = torch.zeros(
        batch_size,
        anchor_cells,
        slots,
    )
    node_offset = torch.zeros(
        batch_size,
        anchor_cells,
        slots,
        2,
    )
    node_mode = torch.zeros(
        batch_size,
        anchor_cells,
        slots,
        dtype=torch.long,
    )
    node_vertical = torch.zeros(
        batch_size,
        anchor_cells,
        slots,
        dtype=torch.long,
    )
    node_boundary = torch.zeros(
        batch_size,
        anchor_cells,
        slots,
    )
    active_ids = torch.zeros(
        batch_size,
        max_active,
        dtype=torch.long,
    )
    active_count = torch.tensor([4, 5])
    node_degree = torch.zeros(
        batch_size,
        max_active,
        dtype=torch.long,
    )
    edge_count = torch.tensor([3, 4])
    edge_pairs = torch.zeros(
        batch_size,
        max_edges,
        2,
        dtype=torch.long,
    )

    for batch_index, count in enumerate((4, 5)):
        ids = torch.arange(count) * 2
        active_ids[
            batch_index,
            :count,
        ] = ids
        for active in ids:
            cell = int(active) // slots
            slot = int(active) % slots
            slot_presence[
                batch_index,
                cell,
                slot,
            ] = 1.0
        for edge_index in range(count - 1):
            edge_pairs[
                batch_index,
                edge_index,
            ] = torch.tensor(
                [edge_index, edge_index + 1]
            )
            node_degree[
                batch_index,
                edge_index,
            ] += 1
            node_degree[
                batch_index,
                edge_index + 1,
            ] += 1

    return {
        "plan_presence": torch.randint(
            0,
            2,
            (
                batch_size,
                plan_cells,
                plan_dimensions,
            ),
        ).to(torch.float32),
        "plan_log_counts": torch.rand(
            batch_size,
            plan_cells,
            plan_dimensions,
        ),
        "plan_orientation": torch.rand(
            batch_size,
            plan_cells,
            orientation_dimensions,
            2,
        )
        * 2.0
        - 1.0,
        "plan_global": torch.randn(batch_size, 8),
        "plan_global_raw": torch.tensor(
            [
                [4.0, 3.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [5.0, 4.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
        "slot_presence": slot_presence,
        "node_offset": node_offset,
        "node_mode": node_mode,
        "node_vertical": node_vertical,
        "node_boundary": node_boundary,
        "active_count": active_count,
        "active_anchor_ids": active_ids,
        "node_degree": node_degree,
        "edge_count": edge_count,
        "edge_pairs": edge_pairs,
        "edge_class": torch.zeros(
            batch_size,
            max_edges,
            dtype=torch.long,
        ),
        "edge_vertical": torch.zeros(
            batch_size,
            max_edges,
            dtype=torch.long,
        ),
        "edge_width": torch.rand(
            batch_size,
            max_edges,
            1,
        ),
        "edge_curve": torch.zeros(
            batch_size,
            max_edges,
            3,
        ),
    }


def test_plan_anchor_forward_loss_and_generation():
    batch = make_batch()
    config = PlanAnchorModelConfig(
        plan_dimensions=8,
        orientation_dimensions=4,
        global_dimensions=8,
        plan_grid_size=4,
        anchor_grid_size=4,
        slots_per_cell=2,
        max_active_nodes=8,
        max_edges=8,
        max_degree=4,
        edge_shape_points=3,
        model_dimensions=64,
        edge_dimensions=24,
        heads=4,
        plan_layers=2,
        anchor_layers=2,
        feedforward_dimensions=128,
        dropout=0.0,
    )
    model = PlanAnchorArchitect(config)
    output = model(batch)

    assert output["slot_presence"].shape == (2, 16, 2)
    assert output["node_offset"].shape == (2, 16, 2, 2)
    assert output["node_degree"].shape == (2, 5, 5)
    assert output["edge_exists"].shape == (2, 5, 5)
    assert output["edge_curve"].shape == (2, 5, 5, 3)

    loss, metrics = plan_anchor_loss(
        output,
        batch,
        target_size_m=1024.0,
        anchor_grid_size=4,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert "anchor_recall" in metrics
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []

    model.eval()
    generated = model.generate(batch)
    assert int(generated["active_count"][0]) == 4
    assert int(generated["active_count"][1]) == 5
    assert generated["edge_exists"].shape == (2, 5, 5)
