import torch

from urban_model.city_planner import CityPlannerConfig
from urban_model.context_encoder_v2 import SpatialContextEncoderV2


def test_balanced_context_encoder_handles_empty_transport_sets():
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
    model = SpatialContextEncoderV2(config)
    batch = {
        "context_cells": torch.randn(2, 9, 7),
        "style": torch.randn(2, 4),
        "controls": torch.zeros(2, 4),
        "context_line_points": torch.zeros(2, 5, 3, 2),
        "context_line_mode": torch.zeros(2, 5, dtype=torch.long),
        "context_line_class": torch.zeros(2, 5, dtype=torch.long),
        "context_line_vertical": torch.zeros(2, 5, dtype=torch.long),
        "context_line_width": torch.zeros(2, 5, 1),
        "context_line_length": torch.zeros(2, 5, 1),
        "context_line_padding": torch.ones(2, 5, dtype=torch.bool),
        "ports": torch.zeros(2, 4, 5),
        "port_mode": torch.zeros(2, 4, dtype=torch.long),
        "port_class": torch.zeros(2, 4, dtype=torch.long),
        "port_vertical": torch.zeros(2, 4, dtype=torch.long),
        "port_padding": torch.ones(2, 4, dtype=torch.bool),
    }
    memory, padding, pool = model(batch)
    assert memory.shape == (2, 98, 64)
    assert padding.shape == (2, 98)
    assert pool.shape == (2, 64)
    assert torch.isfinite(memory).all()
    assert torch.isfinite(pool).all()
