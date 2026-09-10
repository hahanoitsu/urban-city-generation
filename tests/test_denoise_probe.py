from __future__ import annotations

import torch

from urban_analysis.denoise_probe import _binary_iou, _surface_metrics


def _values(classes: torch.Tensor) -> torch.Tensor:
    result = torch.full((19, *classes.shape), -1.0)
    for row in range(classes.shape[0]):
        for column in range(classes.shape[1]):
            result[int(classes[row, column]), row, column] = 1.0
    return result


def test_binary_iou_handles_empty_masks():
    empty = torch.zeros((4, 4), dtype=torch.bool)
    assert _binary_iou(empty, empty) == 1.0


def test_surface_metrics_detects_road_damage():
    reference_classes = torch.tensor(
        [
            [0, 3, 3, 0],
            [0, 3, 3, 0],
            [2, 2, 0, 0],
            [2, 2, 0, 6],
        ]
    )
    candidate_classes = reference_classes.clone()
    candidate_classes[0, 1] = 0

    metrics = _surface_metrics(
        _values(reference_classes),
        _values(candidate_classes),
    )

    assert metrics["surface_accuracy"] < 1.0
    assert metrics["road_iou"] < 1.0
    assert metrics["building_iou"] == 1.0
    assert metrics["rail_iou"] == 1.0
