from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from urban_model.conditioning import (
    SURFACE_TRANSPORT_BACKGROUND_MAX,
    SURFACE_TRANSPORT_BACKGROUND_MIN,
    VERTICAL_BACKGROUND_MAX,
    VERTICAL_BACKGROUND_MIN,
    balance_surface_transport_supervision,
    balance_vertical_supervision,
    build_model_input,
    parse_city_mix,
)
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


def test_surface_transport_background_uses_moderate_balance() -> None:
    values = torch.full((19, 2, 2), -1.0)
    values[3, 0, 1] = 1.0
    supervision = torch.ones_like(values)

    result = balance_surface_transport_supervision(values, supervision)

    assert result[3, 0, 1].item() == pytest.approx(1.0)
    assert result[3, 0, 0].item() == pytest.approx((1.0 / 3.0) ** 0.25)
    assert torch.all(result[:3] == 1.0)
    assert torch.all(result[7:] == 1.0)


def test_surface_transport_background_uses_bounds_and_keeps_empty_channels_negative() -> None:
    values = torch.full((19, 128, 128), -1.0)
    values[3, 0, 0] = 1.0
    values[4, :110] = 1.0
    supervision = torch.ones_like(values)

    result = balance_surface_transport_supervision(values, supervision)

    assert result[3, 1, 1].item() == pytest.approx(SURFACE_TRANSPORT_BACKGROUND_MIN)
    assert result[4, 120, 0].item() == pytest.approx(SURFACE_TRANSPORT_BACKGROUND_MAX)
    assert result[5, 0, 0].item() == pytest.approx(1.0)


def test_vertical_background_balances_positive_pixels() -> None:
    values = torch.full((19, 3, 3), -1.0)
    values[9, 0, 1] = 1.0
    supervision = torch.ones_like(values)

    result = balance_vertical_supervision(values, supervision)

    assert result[9, 0, 1].item() == pytest.approx(1.0)
    assert result[9, 0, 0].item() == pytest.approx((1.0 / 8.0) ** 0.5)
    assert torch.all(result[:8] == 1.0)


def test_vertical_background_uses_bounds_and_keeps_empty_channels_negative() -> None:
    values = torch.full((19, 100, 100), -1.0)
    values[8, 0, 0] = 1.0
    values[9, :80] = 1.0
    supervision = torch.ones_like(values)

    result = balance_vertical_supervision(values, supervision)

    assert result[8, 1, 1].item() == pytest.approx(VERTICAL_BACKGROUND_MIN)
    assert result[9, 90, 0].item() == pytest.approx(VERTICAL_BACKGROUND_MAX)
    assert result[10, 0, 0].item() == pytest.approx(1.0)
