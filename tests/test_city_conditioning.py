from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from urban_model.conditioning import build_model_input, parse_city_mix
from urban_model.config import LayeredDiffusionConfig


def _config() -> LayeredDiffusionConfig:
    return LayeredDiffusionConfig(
        train_manifest=Path("train.jsonl"),
        validation_manifest=Path("validation.jsonl"),
        output_dir=Path("runs/test"),
        city_names=("singapore", "tokyo", "dubai"),
        resolution=(32, 32),
        crop_size_pixels=32,
        crop_stride_pixels=32,
        block_out_channels=(32, 64),
        attention_levels=(False, True),
        norm_num_groups=8,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        precision="fp32",
        device="cpu",
    )


def test_city_mix_is_normalised() -> None:
    mix = parse_city_mix(
        ("singapore", "tokyo", "dubai"),
        "singapore=40,tokyo=35,dubai=25",
    )
    assert mix == pytest.approx(
        {"singapore": 0.4, "tokyo": 0.35, "dubai": 0.25}
    )


def test_city_and_coordinate_channels_are_added() -> None:
    config = _config()
    noisy = torch.zeros((2, 19, 32, 32))
    city = torch.tensor([[1.0, 0.0, 0.0], [0.4, 0.35, 0.25]])
    model_input = build_model_input(noisy, city, config)

    assert model_input.shape == (2, 24, 32, 32)
    assert torch.all(model_input[0, 19] == 1.0)
    assert torch.allclose(model_input[1, 19:22, 0, 0], city[1])
    assert model_input[:, -2:].min().item() == pytest.approx(-1.0)
    assert model_input[:, -2:].max().item() == pytest.approx(1.0)
