from pathlib import Path

import torch

from urban_model.config import LayeredDiffusionConfig
from urban_model.surface_distribution_v2 import (
    MODEL_INPUT_CHANNELS,
    POSITION_CHANNELS,
    SURFACE_CLASS_COUNT,
    _build_model,
    _coordinate_grid,
    _direct_x0_loss,
    _sample_timesteps,
)


def _config() -> LayeredDiffusionConfig:
    return LayeredDiffusionConfig(
        train_manifest=Path("train.jsonl"),
        validation_manifest=Path("validation.jsonl"),
        output_dir=Path("runs/test"),
        resolution=(64, 64),
        block_out_channels=(32, 64, 64),
        attention_levels=(False, False, True),
        batch_size=1,
        num_workers=0,
    )


def test_coordinate_grid_adds_two_absolute_position_channels():
    grid = _coordinate_grid(8, torch.device("cpu"))
    assert POSITION_CHANNELS == 2
    assert grid.shape == (1, 2, 8, 8)
    assert float(grid[0, 0, 0, 0]) == -1.0
    assert float(grid[0, 0, 0, -1]) == 1.0
    assert float(grid[0, 1, 0, 0]) == -1.0
    assert float(grid[0, 1, -1, 0]) == 1.0


def test_high_noise_schedule_is_deliberately_overrepresented():
    torch.manual_seed(7)
    timesteps = _sample_timesteps(10_000, 1000, torch.device("cpu"))
    high_fraction = float((timesteps >= 750).float().mean())
    medium_fraction = float(((timesteps >= 350) & (timesteps < 750)).float().mean())
    low_fraction = float((timesteps < 350).float().mean())
    assert 0.46 < high_fraction < 0.54
    assert 0.26 < medium_fraction < 0.34
    assert 0.16 < low_fraction < 0.24


def test_direct_x0_loss_is_zero_for_exact_prediction():
    target = torch.full((1, SURFACE_CLASS_COUNT, 4, 4), -1.0)
    target[:, 0] = 1.0
    supervision = torch.ones_like(target)
    weights = torch.ones(SURFACE_CLASS_COUNT)
    loss = _direct_x0_loss(
        target.clone(),
        target,
        supervision,
        weights,
        (1.0,) * 19,
    )
    assert float(loss) == 0.0


def test_model_accepts_surface_plus_position_and_predicts_surface():
    config = _config()
    model = _build_model(config)
    values = torch.randn(1, MODEL_INPUT_CHANNELS, 64, 64)
    output = model(values, torch.tensor([999])).sample
    assert output.shape == (1, SURFACE_CLASS_COUNT, 64, 64)
