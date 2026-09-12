from __future__ import annotations

import numpy as np

import torch

from urban_model.whole_city_overfit import (
    _balanced_pixel_weights,
    _coordinate_grid,
    _metrics,
    _square_bounds,
    _training_timestep,
)


def test_square_bounds_preserves_centre_and_aspect():
    bounds = _square_bounds((0.0, 0.0, 20.0, 10.0), 0.0)
    assert bounds == [0.0, -5.0, 20.0, 15.0]


def test_overfit_metrics_are_perfect_for_identical_city():
    target = np.asarray(
        [
            [6, 6, 0, 0],
            [6, 2, 3, 0],
            [1, 2, 4, 0],
            [6, 5, 0, 0],
        ],
        dtype=np.int64,
    )
    values = _metrics(target, target.copy())

    assert values["accuracy"] == 1.0
    assert values["mean_iou"] == 1.0
    assert values["road_iou"] == 1.0
    assert values["urban_iou"] == 1.0


def test_overfit_metrics_penalise_sprinkle_output():
    target = np.zeros((8, 8), dtype=np.int64)
    target[2:6, 2:6] = 2
    target[4, :] = 3

    sample = np.zeros_like(target)
    sample[::2, ::2] = 2
    sample[1::2, 1::2] = 4

    values = _metrics(target, sample)

    assert values["accuracy"] < 0.8
    assert values["urban_iou"] < 0.5
    assert values["road_iou"] < 0.5


def test_coordinate_grid_marks_absolute_position():
    grid = _coordinate_grid(8, torch.device("cpu"))
    assert grid.shape == (1, 2, 8, 8)
    assert float(grid[0, 0, 0, 0]) == -1.0
    assert float(grid[0, 0, 0, -1]) == 1.0
    assert float(grid[0, 1, 0, 0]) == -1.0
    assert float(grid[0, 1, -1, 0]) == 1.0


def test_class_balancing_upweights_rare_city_classes():
    classes = np.full((10, 10), 6, dtype=np.int64)
    classes[0, 0] = 5
    classes[0, 1:6] = 3
    tensor, weights = _balanced_pixel_weights(classes, torch.device("cpu"))
    assert tensor.shape == (1, 1, 10, 10)
    assert weights[5] > weights[6]
    assert weights[3] > weights[6]


def test_training_schedule_explicitly_revisits_high_noise():
    generator = torch.Generator(device="cpu").manual_seed(1)
    high = []
    for step in range(1, 21):
        value = int(_training_timestep(step, 1000, generator).item())
        high.append(value >= 750)
    assert sum(high) >= 10
