import torch

from urban_model.plan_set_graph import (
    PlanSetGraphArchitect,
    PlanSetGraphConfig,
)
from urban_model.plan_set_graph_loss import (
    plan_set_graph_loss,
)


def make_batch():
    batch_size = 2
    queries = 8
    plan_cells = 16
    plan_dimensions = 8
    orientation_dimensions = 4
    max_edges = 8
    shape_points = 3

    node_count = torch.tensor([4, 5])
    node_xy = torch.zeros(
        batch_size,
        queries,
        2,
    )
    node_xy[0, :4] = torch.tensor(
        [
            [-0.8, -0.7],
            [-0.2, 0.2],
            [0.5, 0.3],
            [0.8, -0.4],
        ]
    )
    node_xy[1, :5] = torch.tensor(
        [
            [-0.7, 0.7],
            [-0.2, 0.5],
            [0.1, 0.0],
            [0.6, -0.2],
            [0.7, -0.7],
        ]
    )
    edge_count = torch.tensor([3, 4])
    edge_from = torch.zeros(
        batch_size,
        max_edges,
        dtype=torch.long,
    )
    edge_to = torch.zeros(
        batch_size,
        max_edges,
        dtype=torch.long,
    )
    for batch_index, count in enumerate((4, 5)):
        for edge_index in range(count - 1):
            edge_from[
                batch_index,
                edge_index,
            ] = edge_index
            edge_to[
                batch_index,
                edge_index,
            ] = edge_index + 1

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
        "plan_global": torch.randn(
            batch_size,
            8,
        ),
        "plan_global_raw": torch.tensor(
            [
                [4.0, 3.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [5.0, 4.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
        "node_count": node_count,
        "node_xy": node_xy,
        "node_mode": torch.zeros(
            batch_size,
            queries,
            dtype=torch.long,
        ),
        "node_vertical": torch.zeros(
            batch_size,
            queries,
            dtype=torch.long,
        ),
        "node_boundary": torch.zeros(
            batch_size,
            queries,
        ),
        "edge_count": edge_count,
        "edge_from": edge_from,
        "edge_to": edge_to,
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
        "edge_shape": torch.zeros(
            batch_size,
            max_edges,
            shape_points,
            2,
        ),
    }


def test_plan_set_graph_forward_and_loss():
    batch = make_batch()
    config = PlanSetGraphConfig(
        plan_dimensions=8,
        orientation_dimensions=4,
        global_dimensions=8,
        plan_grid_size=4,
        max_nodes=8,
        max_edges=8,
        max_degree=4,
        edge_shape_points=3,
        model_dimensions=64,
        edge_dimensions=24,
        heads=4,
        plan_layers=2,
        query_layers=2,
        feedforward_dimensions=128,
        dropout=0.0,
    )
    model = PlanSetGraphArchitect(config)
    output = model(batch)

    assert output["node_presence"].shape == (2, 8)
    assert output["node_xy"].shape == (2, 8, 2)
    assert output["edge_exists"].shape == (2, 8, 8)
    assert output["edge_curve"].shape == (2, 8, 8, 3)

    loss, metrics, assignments = plan_set_graph_loss(
        output,
        batch,
        target_size_m=1024.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert len(assignments) == 2
    assert len(torch.unique(assignments[0])) == 4
    assert len(torch.unique(assignments[1])) == 5
    assert "set_chamfer_m" in metrics
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is None
    ]
    assert unused == []
