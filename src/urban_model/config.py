from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

Precision = Literal["fp32", "fp16", "bf16"]
PredictionType = Literal["epsilon", "v_prediction", "sample"]


@dataclass(frozen=True)
class LayeredDiffusionConfig:
    train_manifest: Path
    validation_manifest: Path
    output_dir: Path
    city_names: tuple[str, ...] = ("singapore",)
    resolution: tuple[int, int] = (128, 128)
    crop_size_pixels: int = 128
    crop_stride_pixels: int = 32
    augment: bool = True
    vertical_crop_repeat: int = 2
    balance_cities: bool = True
    block_out_channels: tuple[int, ...] = (64, 128, 256, 256)
    layers_per_block: int = 2
    attention_levels: tuple[bool, ...] = (False, False, False, True)
    norm_num_groups: int = 8
    coordinate_channels: bool = True
    diffusion_steps: int = 1000
    inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: PredictionType = "v_prediction"
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    pin_memory: bool = True
    precision: Precision = "bf16"
    seed: int = 5132
    device: str = "cuda"
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0
    checkpoint_every: int = 5
    preview_every: int = 5
    early_stopping_patience: int = 20
    channel_loss_weights: tuple[float, ...] = (
        1.0,
        1.0,
        1.0,
        1.1,
        1.1,
        1.1,
        1.2,
        1.0,
        1.2,
        1.2,
        1.4,
        1.4,
        0.6,
        0.5,
        0.8,
        0.8,
        0.5,
        1.0,
        1.0,
    )
    max_height_m: float = 180.0
    max_surface_offset_m: float = 12.0
    max_underground_depth_m: float = 40.0
    max_elevated_height_m: float = 30.0
    road_max_grade: float = 0.08
    rail_max_grade: float = 0.035
    auxiliary_threshold: float = 0.35
    minimum_vector_component_pixels: int = 4


class LayeredDiffusionConfigError(ValueError):
    pass


def _resolve(value: Any, *, base: Path, name: str) -> Path:
    if value is None:
        raise LayeredDiffusionConfigError(f"Missing '{name}'")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _tuple(value: Any, *, name: str, cast) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise LayeredDiffusionConfigError(f"'{name}' must be a list")
    try:
        result = tuple(cast(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise LayeredDiffusionConfigError(f"Invalid values in '{name}'") from exc
    if not result:
        raise LayeredDiffusionConfigError(f"'{name}' cannot be empty")
    return result


def load_layered_diffusion_config(path: str | Path) -> LayeredDiffusionConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise LayeredDiffusionConfigError("The config must be a mapping")

    data = raw.get("data", {})
    model = raw.get("model", {})
    diffusion = raw.get("diffusion", {})
    run = raw.get("run", {})
    vector = raw.get("vector", {})
    if not all(isinstance(section, dict) for section in (data, model, diffusion, run, vector)):
        raise LayeredDiffusionConfigError("Config sections must be mappings")

    base = config_path.parent.parent
    city_names = _tuple(
        data.get("cities", ["singapore"]),
        name="data.cities",
        cast=lambda value: str(value).strip(),
    )
    resolution = _tuple(data.get("resolution", [128, 128]), name="data.resolution", cast=int)
    block_channels = _tuple(
        model.get("block_out_channels", [64, 128, 256, 256]),
        name="model.block_out_channels",
        cast=int,
    )
    attention = _tuple(
        model.get("attention_levels", [False, False, False, True]),
        name="model.attention_levels",
        cast=bool,
    )
    weights = _tuple(
        run.get(
            "channel_loss_weights",
            [
                1.0,
                1.0,
                1.0,
                1.1,
                1.1,
                1.1,
                1.2,
                1.0,
                1.2,
                1.2,
                1.4,
                1.4,
                0.6,
                0.5,
                0.8,
                0.8,
                0.5,
                1.0,
                1.0,
            ],
        ),
        name="run.channel_loss_weights",
        cast=float,
    )

    config = LayeredDiffusionConfig(
        train_manifest=_resolve(data.get("train_manifest"), base=base, name="data.train_manifest"),
        validation_manifest=_resolve(
            data.get("validation_manifest"),
            base=base,
            name="data.validation_manifest",
        ),
        output_dir=_resolve(run.get("output_dir"), base=base, name="run.output_dir"),
        city_names=city_names,
        resolution=(int(resolution[0]), int(resolution[1])),
        crop_size_pixels=int(data.get("crop_size_pixels", 128)),
        crop_stride_pixels=int(data.get("crop_stride_pixels", 32)),
        augment=bool(data.get("augment", True)),
        vertical_crop_repeat=int(data.get("vertical_crop_repeat", 2)),
        balance_cities=bool(data.get("balance_cities", True)),
        block_out_channels=block_channels,
        layers_per_block=int(model.get("layers_per_block", 2)),
        attention_levels=attention,
        norm_num_groups=int(model.get("norm_num_groups", 8)),
        coordinate_channels=bool(model.get("coordinate_channels", True)),
        diffusion_steps=int(diffusion.get("train_steps", 1000)),
        inference_steps=int(diffusion.get("inference_steps", 100)),
        beta_schedule=str(diffusion.get("beta_schedule", "squaredcos_cap_v2")),
        prediction_type=str(diffusion.get("prediction_type", "v_prediction")),
        epochs=int(run.get("epochs", 100)),
        batch_size=int(run.get("batch_size", 16)),
        learning_rate=float(run.get("learning_rate", 1e-4)),
        weight_decay=float(run.get("weight_decay", 1e-4)),
        num_workers=int(run.get("num_workers", 8)),
        pin_memory=bool(run.get("pin_memory", True)),
        precision=str(run.get("precision", "bf16")).lower(),
        seed=int(run.get("seed", 5132)),
        device=str(run.get("device", "cuda")),
        ema_decay=float(run.get("ema_decay", 0.999)),
        gradient_clip_norm=float(run.get("gradient_clip_norm", 1.0)),
        checkpoint_every=int(run.get("checkpoint_every", 5)),
        preview_every=int(run.get("preview_every", 5)),
        early_stopping_patience=int(run.get("early_stopping_patience", 20)),
        channel_loss_weights=weights,
        max_height_m=float(vector.get("max_height_m", 180.0)),
        max_surface_offset_m=float(vector.get("max_surface_offset_m", 12.0)),
        max_underground_depth_m=float(vector.get("max_underground_depth_m", 40.0)),
        max_elevated_height_m=float(vector.get("max_elevated_height_m", 30.0)),
        road_max_grade=float(vector.get("road_max_grade", 0.08)),
        rail_max_grade=float(vector.get("rail_max_grade", 0.035)),
        auxiliary_threshold=float(vector.get("auxiliary_threshold", 0.35)),
        minimum_vector_component_pixels=int(vector.get("minimum_component_pixels", 4)),
    )
    _validate(config)
    return config


def _validate(config: LayeredDiffusionConfig) -> None:
    if len(set(config.city_names)) != len(config.city_names) or any(
        not name for name in config.city_names
    ):
        raise LayeredDiffusionConfigError("data.cities must contain unique non-empty names")
    if len(config.resolution) != 2 or config.resolution[0] != config.resolution[1]:
        raise LayeredDiffusionConfigError("data.resolution must be a square [height, width]")
    scale = 2 ** (len(config.block_out_channels) - 1)
    if any(size <= 0 or size % scale for size in config.resolution):
        raise LayeredDiffusionConfigError(
            f"Resolution dimensions must be positive and divisible by {scale}"
        )
    if len(config.attention_levels) != len(config.block_out_channels):
        raise LayeredDiffusionConfigError(
            "model.attention_levels must match model.block_out_channels"
        )
    if any(channel <= 0 or channel % config.norm_num_groups for channel in config.block_out_channels):
        raise LayeredDiffusionConfigError(
            "model.norm_num_groups must divide every block channel count"
        )
    if len(config.channel_loss_weights) != 19:
        raise LayeredDiffusionConfigError("run.channel_loss_weights must contain 19 values")
    if any(weight <= 0 for weight in config.channel_loss_weights):
        raise LayeredDiffusionConfigError("Channel loss weights must be positive")
    if config.crop_size_pixels <= 0 or config.crop_stride_pixels <= 0:
        raise LayeredDiffusionConfigError("Crop size and stride must be positive")
    if config.vertical_crop_repeat <= 0:
        raise LayeredDiffusionConfigError("vertical_crop_repeat must be positive")
    if config.diffusion_steps < 2:
        raise LayeredDiffusionConfigError("diffusion.train_steps must be at least 2")
    if not 1 <= config.inference_steps <= config.diffusion_steps:
        raise LayeredDiffusionConfigError("Invalid inference step count")
    if config.prediction_type not in {"epsilon", "v_prediction", "sample"}:
        raise LayeredDiffusionConfigError("Invalid diffusion.prediction_type")
    if config.epochs <= 0 or config.batch_size <= 0 or config.num_workers < 0:
        raise LayeredDiffusionConfigError("Invalid run dimensions")
    if config.precision not in {"fp32", "fp16", "bf16"}:
        raise LayeredDiffusionConfigError("run.precision must be fp32, fp16 or bf16")
    if not 0.0 < config.ema_decay < 1.0:
        raise LayeredDiffusionConfigError("run.ema_decay must be between 0 and 1")
    if not 0.0 <= config.auxiliary_threshold < 1.0:
        raise LayeredDiffusionConfigError("vector.auxiliary_threshold must be in [0,1)")
    if config.road_max_grade <= 0 or config.rail_max_grade <= 0:
        raise LayeredDiffusionConfigError("Vector grade limits must be positive")
    if any(
        value <= 0
        for value in (
            config.max_height_m,
            config.max_surface_offset_m,
            config.max_underground_depth_m,
            config.max_elevated_height_m,
        )
    ):
        raise LayeredDiffusionConfigError("Vector height and depth limits must be positive")
