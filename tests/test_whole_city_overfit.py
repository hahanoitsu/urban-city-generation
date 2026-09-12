from __future__ import annotations

import numpy as np

from urban_model.whole_city_overfit import _metrics, _square_bounds


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
