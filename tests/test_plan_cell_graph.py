import torch

from urban_model.plan_cell_graph import (
    PlanCellGraphArchitect,
    PlanCellGraphConfig,
    build_query_layout,
)
from urban_model.plan_cell_graph_loss import canonical_cell_order, plan_cell_graph_loss


def make_batch():
    batch_size = 2
    plan_grid = 4
    plan_cells = plan_grid * plan_grid
    max_nodes = 8
    max_edges = 8
    shape_points = 3

    node_count = torch.tensor([4, 5])
    node_xy = torch.zeros(batch_size, max_nodes, 2)
    node_xy[0, :4] = torch.tensor([[-0.85, -0.80], [-0.60, -0.55], [0.20, 0.25], [0.72, 0.70]])
    node_xy[1, :5] = torch.tensor(
        [[-0.75, 0.75], [-0.20, 0.55], [0.15, 0.05], [0.55, -0.20], [0.80, -0.75]]
    )

    plan_counts = torch.zeros(batch_size, plan_cells, 8)
    for batch_index in range(batch_size):
        count = int(node_count[batch_index])
        unit = (node_xy[batch_index, :count] + 1.0) * 0.5
        column = torch.floor(unit[:, 0] * plan_grid).long().clamp(0, plan_grid - 1)
        row = torch.floor(unit[:, 1] * plan_grid).long().clamp(0, plan_grid - 1)
        cells = row * plan_grid + column
        for cell in cells:
            plan_counts[batch_index, int(cell), 0] += 1.0

    edge_count = torch.tensor([3, 4])
    edge_from = torch.zeros(batch_size, max_edges, dtype=torch.long)
    edge_to = torch.zeros_like(edge_from)
    for batch_index, count in enumerate((4, 5)):
        for edge_index in range(count - 1):
            edge_from[batch_index, edge_index] = edge_index
            edge_to[batch_index, edge_index] = edge_index + 1

    return {
        "plan_presence": (plan_counts > 0).to(torch.float32),
        "plan_log_counts": torch.log1p(plan_counts),
        "plan_counts": plan_counts,
        "plan_orientation": torch.zeros(batch_size, plan_cells, 4, 2),
        "plan_global": torch.randn(batch_size, 8),
        "node_count": node_count,
        "node_xy": node_xy,
        "node_mode": torch.zeros(batch_size, max_nodes, dtype=torch.long),
        "node_vertical": torch.zeros(batch_size, max_nodes, dtype=torch.long),
        "node_boundary": torch.zeros(batch_size, max_nodes),
        "edge_count": edge_count,
        "edge_from": edge_from,
        "edge_to": edge_to,
        "edge_class": torch.zeros(batch_size, max_edges, dtype=torch.long),
        "edge_vertical": torch.zeros(batch_size, max_edges, dtype=torch.long),
        "edge_width": torch.rand(batch_size, max_edges, 1),
        "edge_shape": torch.zeros(batch_size, max_edges, shape_points, 2),
    }


def test_query_layout_matches_plan_counts():
    batch = make_batch()
    cell_ids, slot_ids, padding = build_query_layout(
        batch["plan_counts"], batch["node_count"], max_slots_per_cell=8
    )
    assert cell_ids.shape == (2, 5)
    assert slot_ids.shape == (2, 5)
    assert padding.shape == (2, 5)
    assert int((~padding[0]).sum()) == 4
    assert int((~padding[1]).sum()) == 5


def test_plan_cell_graph_forward_and_loss():
    batch = make_batch()
    config = PlanCellGraphConfig(
        plan_dimensions=8,
        orientation_dimensions=4,
        global_dimensions=8,
        plan_grid_size=4,
        max_nodes=8,
        max_edges=8,
        max_slots_per_cell=8,
        max_degree=4,
        edge_shape_points=3,
        model_dimensions=64,
        edge_dimensions=24,
        heads=4,
        plan_layers=2,
        node_layers=2,
        feedforward_dimensions=128,
        dropout=0.0,
    )
    model = PlanCellGraphArchitect(config)
    output = model(batch)

    assert output["node_xy"].shape == (2, 5, 2)
    assert output["edge_exists"].shape == (2, 5, 5)
    assert output["edge_curve"].shape == (2, 5, 5, 3, 2)

    for batch_index in range(2):
        count = int(batch["node_count"][batch_index])
        order = canonical_cell_order(batch["node_xy"][batch_index, :count], 4)
        target = batch["node_xy"][batch_index, :count][order]
        predicted = output["node_xy"][batch_index, :count]
        target_cell = torch.floor((target + 1.0) * 0.5 * 4).long().clamp(0, 3)
        predicted_cell = torch.floor((predicted + 1.0) * 0.5 * 4).long().clamp(0, 3)
        assert torch.equal(target_cell, predicted_cell)

    loss, metrics, orders = plan_cell_graph_loss(output, batch, target_size_m=1024.0, grid_size=4)
    loss.backward()

    assert torch.isfinite(loss)
    assert len(orders) == 2
    assert metrics["node_position_mae_m"] < 200.0
    unused = [
        name
        for name, parameter in model.named_parameters()
        if (parameter.requires_grad and parameter.grad is None)
    ]
    assert unused == []
