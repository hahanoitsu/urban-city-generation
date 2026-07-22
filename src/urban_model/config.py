from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class TrainingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DataConfig:
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path | None = None
    augment: bool = True
    height_scale_m: float = 180.0


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 12
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.0


@dataclass(frozen=True)
class LossConfig:
    road: float = 1.0
    landuse: float = 1.0
    binary: float = 1.0
    height: float = 0.5
    centerline: float = 0.5
    dice: float = 0.5
    road_class_weights: tuple[float, ...] = (0.2, 1.0, 1.0, 1.0)
    landuse_class_weights: tuple[float, ...] = (0.25, 1.0, 1.0, 1.0, 1.0, 1.0)
    binary_positive_weights: tuple[float, ...] = (4.0, 2.0, 8.0)
    centerline_positive_weight: float = 10.0
    height_confidence_weights: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 0
    seed: int = 5132
    device: str = "auto"
    amp: bool = True
    gradient_clip_norm: float = 1.0
    checkpoint_every: int = 5
    preview_every: int = 5
    early_stopping_patience: int = 20
    max_train_steps_per_epoch: int | None = None
    max_validation_steps: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    run: RunConfig
    source_path: Path
    raw: dict[str, Any] = field(repr=False)


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise TrainingConfigError(f"'{key}' must be a mapping")
    return value


def _path(value: Any, *, base: Path, name: str, required: bool = True) -> Path | None:
    if value is None:
        if required:
            raise TrainingConfigError(f"Missing required path '{name}'")
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _float_tuple(value: Any, default: tuple[float, ...], name: str) -> tuple[float, ...]:
    if value is None:
        return default
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigError(f"'{name}' must be a list of numbers") from exc


def _int_tuple(value: Any, default: tuple[int, ...], name: str) -> tuple[int, ...]:
    if value is None:
        return default
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TrainingConfigError(f"'{name}' must be a list of integers") from exc
    if not result or any(item <= 0 for item in result):
        raise TrainingConfigError(f"'{name}' must contain positive integers")
    return result


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TrainingConfigError("The training config must be a mapping")

    data_raw = _mapping(raw, "data")
    model_raw = _mapping(raw, "model")
    loss_raw = _mapping(raw, "loss")
    optimizer_raw = _mapping(raw, "optimizer")
    run_raw = _mapping(raw, "run")
    base = config_path.parent.parent

    data = DataConfig(
        train_manifest=_path(
            data_raw.get("train_manifest"), base=base, name="data.train_manifest"
        ),
        validation_manifest=_path(
            data_raw.get("validation_manifest"),
            base=base,
            name="data.validation_manifest",
        ),
        test_manifest=_path(
            data_raw.get("test_manifest"),
            base=base,
            name="data.test_manifest",
            required=False,
        ),
        augment=bool(data_raw.get("augment", True)),
        height_scale_m=float(data_raw.get("height_scale_m", 180.0)),
    )

    model = ModelConfig(
        input_channels=int(model_raw.get("input_channels", 12)),
        base_channels=int(model_raw.get("base_channels", 32)),
        channel_multipliers=_int_tuple(
            model_raw.get("channel_multipliers"),
            (1, 2, 4, 8),
            "model.channel_multipliers",
        ),
        dropout=float(model_raw.get("dropout", 0.0)),
    )
    if model.input_channels <= 0 or model.base_channels <= 0:
        raise TrainingConfigError("Model channel counts must be positive")
    if not 0.0 <= model.dropout < 1.0:
        raise TrainingConfigError("model.dropout must be in [0, 1)")

    loss = LossConfig(
        road=float(loss_raw.get("road", 1.0)),
        landuse=float(loss_raw.get("landuse", 1.0)),
        binary=float(loss_raw.get("binary", 1.0)),
        height=float(loss_raw.get("height", 0.5)),
        centerline=float(loss_raw.get("centerline", 0.5)),
        dice=float(loss_raw.get("dice", 0.5)),
        road_class_weights=_float_tuple(
            loss_raw.get("road_class_weights"),
            (0.2, 1.0, 1.0, 1.0),
            "loss.road_class_weights",
        ),
        landuse_class_weights=_float_tuple(
            loss_raw.get("landuse_class_weights"),
            (0.25, 1.0, 1.0, 1.0, 1.0, 1.0),
            "loss.landuse_class_weights",
        ),
        binary_positive_weights=_float_tuple(
            loss_raw.get("binary_positive_weights"),
            (4.0, 2.0, 8.0),
            "loss.binary_positive_weights",
        ),
        centerline_positive_weight=float(loss_raw.get("centerline_positive_weight", 10.0)),
        height_confidence_weights=_float_tuple(
            loss_raw.get("height_confidence_weights"),
            (0.25, 0.5, 0.75, 1.0),
            "loss.height_confidence_weights",
        ),
    )
    if len(loss.road_class_weights) != 4:
        raise TrainingConfigError("loss.road_class_weights must have four entries")
    if len(loss.landuse_class_weights) != 6:
        raise TrainingConfigError("loss.landuse_class_weights must have six entries")
    if len(loss.binary_positive_weights) != 3:
        raise TrainingConfigError("loss.binary_positive_weights must have three entries")
    if len(loss.height_confidence_weights) != 4:
        raise TrainingConfigError("loss.height_confidence_weights must have four entries")

    optimizer = OptimizerConfig(
        learning_rate=float(optimizer_raw.get("learning_rate", 3e-4)),
        weight_decay=float(optimizer_raw.get("weight_decay", 1e-4)),
    )

    output_dir = _path(run_raw.get("output_dir"), base=base, name="run.output_dir")
    run = RunConfig(
        output_dir=output_dir,
        epochs=int(run_raw.get("epochs", 100)),
        batch_size=int(run_raw.get("batch_size", 4)),
        num_workers=int(run_raw.get("num_workers", 0)),
        seed=int(run_raw.get("seed", 5132)),
        device=str(run_raw.get("device", "auto")),
        amp=bool(run_raw.get("amp", True)),
        gradient_clip_norm=float(run_raw.get("gradient_clip_norm", 1.0)),
        checkpoint_every=int(run_raw.get("checkpoint_every", 5)),
        preview_every=int(run_raw.get("preview_every", 5)),
        early_stopping_patience=int(run_raw.get("early_stopping_patience", 20)),
        max_train_steps_per_epoch=(
            int(run_raw["max_train_steps_per_epoch"])
            if run_raw.get("max_train_steps_per_epoch") is not None
            else None
        ),
        max_validation_steps=(
            int(run_raw["max_validation_steps"])
            if run_raw.get("max_validation_steps") is not None
            else None
        ),
    )
    if run.epochs <= 0 or run.batch_size <= 0 or run.num_workers < 0:
        raise TrainingConfigError(
            "epochs and batch_size must be positive; num_workers cannot be negative"
        )

    return TrainingConfig(
        data=data,
        model=model,
        loss=loss,
        optimizer=optimizer,
        run=run,
        source_path=config_path,
        raw=raw,
    )
