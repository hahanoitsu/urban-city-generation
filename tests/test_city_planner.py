import torch

from urban_model.city_plan_loss import city_plan_loss
from urban_model.city_planner import CityPlanner, CityPlannerConfig


def make_batch():
    batch_size = 2
    context_cells = 9
    context_lines = 6
    ports = 4
    grid_cells = 16
    plan_dimensions = 8
    global_dimensions = 8
    return {
        "context_cells": torch.randn(batch_size, context_cells, 7),
        "style": torch.randn(batch_size, 4),
        "controls": torch.zeros(batch_size, 4),
        "context_line_points": torch.randn(
            batch_size,
            context_lines,
            3,
            2,
        ),
        "context_line_mode": torch.zeros(
            batch_size,
            context_lines,
            dtype=torch.long,
        ),
        "context_line_class": torch.zeros(
            batch_size,
            context_lines,
            dtype=torch.long,
        ),
        "context_line_vertical": torch.zeros(
            batch_size,
            context_lines,
            dtype=torch.long,
        ),
        "context_line_width": torch.rand(
            batch_size,
            context_lines,
            1,
        ),
        "context_line_length": torch.rand(
            batch_size,
            context_lines,
            1,
        ),
        "context_line_padding": torch.zeros(
            batch_size,
            context_lines,
            dtype=torch.bool,
        ),
        "ports": torch.randn(batch_size, ports, 5),
        "port_mode": torch.zeros(
            batch_size,
            ports,
            dtype=torch.long,
        ),
        "port_class": torch.zeros(
            batch_size,
            ports,
            dtype=torch.long,
        ),
        "port_vertical": torch.zeros(
            batch_size,
            ports,
            dtype=torch.long,
        ),
        "port_padding": torch.zeros(
            batch_size,
            ports,
            dtype=torch.bool,
        ),
        "plan_presence": torch.randint(
            0,
            2,
            (batch_size, grid_cells, plan_dimensions),
        ).to(torch.float32),
        "plan_log_counts": torch.rand(
            batch_size,
            grid_cells,
            plan_dimensions,
        ),
        "plan_counts": torch.rand(
            batch_size,
            grid_cells,
            plan_dimensions,
        ),
        "plan_orientation": torch.rand(
            batch_size,
            grid_cells,
            4,
            2,
        ) * 2.0 - 1.0,
        "plan_orientation_mask": torch.randint(
            0,
            2,
            (batch_size, grid_cells, 4),
        ).to(torch.bool),
        "plan_global": torch.randn(
            batch_size,
            global_dimensions,
        ),
        "plan_global_raw": torch.rand(
            batch_size,
            global_dimensions,
        ),
    }


def test_city_planner_forward_and_loss():
    batch = make_batch()
    config = CityPlannerConfig(
        context_dimensions=7,
        style_dimensions=4,
        plan_dimensions=8,
        orientation_dimensions=4,
        global_dimensions=8,
        grid_size=4,
        context_line_points=3,
        model_dimensions=64,
        heads=4,
        context_layers=2,
        planner_layers=2,
        feedforward_dimensions=128,
        dropout=0.0,
    )
    model = CityPlanner(config)
    output = model(batch)

    assert output["plan_presence"].shape == (2, 16, 8)
    assert output["plan_log_count"].shape == (2, 16, 8)
    assert output["plan_orientation"].shape == (2, 16, 4, 2)
    assert output["plan_global"].shape == (2, 8)

    loss, metrics = city_plan_loss(
        output,
        batch,
        presence_pos_weight=torch.ones(8),
        global_mean=torch.zeros(8),
        global_std=torch.ones(8),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert "node_mae" in metrics
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []
