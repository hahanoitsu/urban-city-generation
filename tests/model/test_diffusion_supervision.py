from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from urban_model.data import MODEL_CHANNELS, PROFILE_NAMES
from urban_model.model import (
    PROFILE_BACKGROUND_WEIGHT,
    _add_profile_background_supervision,
    weighted_diffusion_loss,
)
from urban_model.train import _preview_seed


def _base_mask(height: int = 2, width: int = 2):
    mask = torch.zeros((1, MODEL_CHANNELS, height, width), dtype=torch.float32)
    mask[:, : MODEL_CHANNELS - len(PROFILE_NAMES)] = 1.0
    return mask


def test_profile_background_gets_weak_supervision() -> None:
    mask = _base_mask()
    result = _add_profile_background_supervision(mask)
    profile_start = MODEL_CHANNELS - len(PROFILE_NAMES)

    expected = torch.full_like(result[:, profile_start:], PROFILE_BACKGROUND_WEIGHT)
    assert torch.allclose(result[:, profile_start:], expected)
    assert torch.all(result[:, :profile_start] == 1.0)


def test_real_profile_confidence_is_not_reduced() -> None:
    mask = _base_mask()
    profile_start = MODEL_CHANNELS - len(PROFILE_NAMES)
    mask[:, profile_start + 2] = 0.8

    result = _add_profile_background_supervision(mask)

    assert torch.allclose(result[:, profile_start + 2], torch.full_like(result[:, 0], 0.8))
    assert torch.allclose(
        result[:, profile_start + 1],
        torch.full_like(result[:, 0], PROFILE_BACKGROUND_WEIGHT),
    )


def test_weighted_loss_normalises_channels_separately() -> None:
    prediction = torch.zeros((1, MODEL_CHANNELS, 1, 1), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    mask = _base_mask(1, 1)
    profile_start = MODEL_CHANNELS - len(PROFILE_NAMES)
    prediction[:, profile_start] = 1.0

    weights = tuple(1.0 for _ in range(MODEL_CHANNELS))
    loss = weighted_diffusion_loss(prediction, target, mask, weights)

    assert loss.item() == pytest.approx(1.0 / MODEL_CHANNELS)

    prediction.zero_()
    prediction[:, 0] = 1.0
    weighted = (2.0,) + tuple(1.0 for _ in range(MODEL_CHANNELS - 1))
    loss = weighted_diffusion_loss(prediction, target, torch.ones_like(mask), weighted)
    assert loss.item() == pytest.approx(2.0 / (MODEL_CHANNELS + 1.0))


def test_sparse_profile_error_is_not_lost_in_dense_channels() -> None:
    prediction = torch.zeros((1, MODEL_CHANNELS, 4, 4), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    mask = torch.zeros_like(prediction)
    profile_start = MODEL_CHANNELS - len(PROFILE_NAMES)

    mask[:, 0] = 1.0
    mask[:, profile_start, 0, 0] = 1.0
    prediction[:, 0] = 1.0
    prediction[:, profile_start] = 1.0

    weights = tuple(1.0 for _ in range(MODEL_CHANNELS))
    loss = weighted_diffusion_loss(prediction, target, mask, weights)

    assert loss.item() == pytest.approx(2.0 / (1.0 + len(PROFILE_NAMES)))


def test_preview_seed_is_fixed_for_a_run() -> None:
    config = SimpleNamespace(seed=5132)
    assert _preview_seed(config) == _preview_seed(config)
    assert _preview_seed(config) != config.seed
