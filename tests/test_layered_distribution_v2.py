from pathlib import Path

import torch

from urban_model.config import LayeredDiffusionConfig
from urban_model.data import MODEL_CHANNELS
from urban_model.layered_distribution_v2 import (
    MODEL_INPUT_CHANNELS,
    _build_model,
    _direct_x0_loss,
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


def test_full_layered_model_accepts_position_conditioning():
    config = _config()
    model = _build_model(config)
    values = torch.randn(1, MODEL_INPUT_CHANNELS, 64, 64)
    output = model(values, torch.tensor([999])).sample
    assert output.shape == (1, MODEL_CHANNELS, 64, 64)


def test_full_layered_x0_loss_is_zero_for_exact_prediction():
    target = torch.full((1, MODEL_CHANNELS, 4, 4), -1.0)
    target[:, 0] = 1.0
    supervision = torch.ones_like(target)
    class_weights = torch.ones(8)
    loss = _direct_x0_loss(
        target.clone(),
        target,
        supervision,
        class_weights,
        (1.0,) * MODEL_CHANNELS,
    )
    assert float(loss) == 0.0
