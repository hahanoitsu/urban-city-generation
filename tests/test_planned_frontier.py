import torch

from urban_model.city_plan_data import CityPlanConfig, build_city_plan
from urban_model.frontier_data import FrontierProgramConfig, build_frontier_program
from urban_model.planned_frontier import (
    PlannedFrontierArchitect,
    PlannedFrontierConfig,
)
from urban_model.planned_frontier_data import build_program_progress
from urban_model.planned_frontier_loss import planned_frontier_loss


def make_sample():
    max_nodes = 8
    max_edges = 8
    edge_points = 3
    node_xy = torch.zeros(max_nodes, 2)
    node_xy[:4] = torch.tensor(
        [
            [-0.8, -0.7],
            [-0.2, 0.2],
            [0.5, 0.3],
            [0.8, -0.4],
        ]
    )
    return {
        "node_count": torch.tensor(4),
        "node_xy": node_xy,
        "node_mode": torch.tensor([0, 0, 1, 1, 0, 0, 0, 0]),
        "node_vertical": torch.zeros(max_nodes, dtype=torch.long),
        "node_boundary": torch.tensor(
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "edge_count": torch.tensor(4),
        "edge_from": torch.tensor([0, 1, 1, 2, 0, 0, 0, 0]),
        "edge_to": torch.tensor([1, 2, 3, 3, 0, 0, 0, 0]),
        "edge_mode": torch.tensor([0, 0, 0, 1, 0, 0, 0, 0]),
        "edge_class": torch.tensor([0, 2, 1, 3, 0, 0, 0, 0]),
        "edge_vertical": torch.zeros(max_edges, dtype=torch.long),
        "edge_width": torch.ones(max_edges, 1) * 0.2,
        "edge_shape": torch.zeros(max_edges, edge_points, 2),
    }


def test_planned_frontier_forward_and_loss():
    sample = make_sample()
    counts, orientation, orientation_mask, global_raw = build_city_plan(
        sample,
        CityPlanConfig(grid_size=4),
    )
    program = build_frontier_program(
        sample,
        FrontierProgramConfig(
            max_steps=32,
            max_nodes=8,
            curve_points=3,
        ),
    )
    progress = build_program_progress(program, global_raw)

    batch = {
        "plan_presence": (counts > 0).to(torch.float32).unsqueeze(0),
        "plan_log_counts": torch.log1p(counts).unsqueeze(0),
        "plan_orientation": orientation.unsqueeze(0),
        "plan_orientation_mask": orientation_mask.unsqueeze(0),
        "plan_global": global_raw.unsqueeze(0),
        **{
            key: value.unsqueeze(0)
            for key, value in program.items()
            if torch.is_tensor(value)
        },
        "program_progress": progress.unsqueeze(0),
    }

    config = PlannedFrontierConfig(
        plan_dimensions=8,
        orientation_dimensions=4,
        global_dimensions=8,
        grid_size=4,
        max_steps=32,
        curve_points=3,
        model_dimensions=64,
        heads=4,
        decoder_layers=2,
        feedforward_dimensions=128,
        dropout=0.0,
    )
    model = PlannedFrontierArchitect(config)
    length = int(batch["program_length"].max()) - 1
    output = model(batch, input_length=length)

    assert output["op"].shape == (1, length, 7)
    assert output["xy_mean"].shape == (1, length, 2)
    assert output["curve_mean"].shape == (1, length, 3)

    loss, metrics = planned_frontier_loss(output, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert "op" in metrics
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []

    link = batch["program_op"][0] == 4
    if bool(link.any()):
        assert bool(
            batch["program_xy"][0, link].abs().sum(dim=-1).gt(0).any()
        )
