from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

SemanticTask = Literal["block", "outpaint"]

# CityGen models one categorical semantic field and synthesises height separately.
SEMANTIC_NAMES = (
    "terrain",
    "vegetation",
    "building",
    "road",
    "rail",
    "water",
)

SEMANTIC_PALETTE = np.asarray(
    [
        (225, 220, 210),
        (128, 190, 120),
        (125, 125, 125),
        (45, 45, 45),
        (225, 185, 80),
        (82, 170, 226),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class SemanticDiffusionConfig:
    task: SemanticTask
    train_manifest: Path
    validation_manifest: Path
    output_dir: Path
    resolution: tuple[int, int]
    crop_size_pixels: int = 128
    crop_stride_pixels: int = 32
    augment: bool = True
    directions: tuple[str, ...] = ("east", "west", "north", "south")
    boundary_width: int = 3
    guide_length: int = 24
    block_out_channels: tuple[int, ...] = (64, 128, 256)
    layers_per_block: int = 2
    attention_levels: tuple[bool, ...] = (False, False, True)
    norm_num_groups: int = 8
    diffusion_steps: int = 1000
    inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 5132
    device: str = "auto"
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0
    checkpoint_every: int = 10
    preview_every: int = 10
    early_stopping_patience: int = 20
    initial_checkpoint: Path | None = None
    max_train_steps_per_epoch: int | None = None
    max_validation_steps: int | None = None


class SemanticDiffusionConfigError(ValueError):
    pass


def _resolve_path(
    value: Any,
    *,
    base: Path,
    name: str,
    required: bool = True,
) -> Path | None:
    if value is None:
        if required:
            raise SemanticDiffusionConfigError(f"Missing '{name}'")
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _int_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise SemanticDiffusionConfigError(f"'{name}' must be a list of integers") from exc
    if not result or any(item <= 0 for item in result):
        raise SemanticDiffusionConfigError(f"'{name}' must contain positive integers")
    return result


def _bool_tuple(value: Any, *, name: str) -> tuple[bool, ...]:
    if not isinstance(value, list):
        raise SemanticDiffusionConfigError(f"'{name}' must be a list")
    return tuple(bool(item) for item in value)


def load_semantic_diffusion_config(path: str | Path) -> SemanticDiffusionConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SemanticDiffusionConfigError("The semantic diffusion config must be a mapping")

    data = raw.get("data", {})
    model = raw.get("model", {})
    diffusion = raw.get("diffusion", {})
    run = raw.get("run", {})
    if not all(isinstance(section, dict) for section in (data, model, diffusion, run)):
        raise SemanticDiffusionConfigError("Config sections must be mappings")

    base = config_path.parent.parent
    task = str(data.get("task", "block")).strip().lower()
    if task not in {"block", "outpaint"}:
        raise SemanticDiffusionConfigError("data.task must be 'block' or 'outpaint'")

    default_resolution = [128, 128] if task == "block" else [128, 256]
    resolution = _int_tuple(data.get("resolution", default_resolution), name="data.resolution")
    if len(resolution) != 2:
        raise SemanticDiffusionConfigError("data.resolution must contain [height, width]")

    block_out_channels = _int_tuple(
        model.get("block_out_channels", [64, 128, 256]),
        name="model.block_out_channels",
    )
    attention_levels = _bool_tuple(
        model.get("attention_levels", [False, False, True]),
        name="model.attention_levels",
    )

    config = SemanticDiffusionConfig(
        task=task,
        train_manifest=_resolve_path(
            data.get("train_manifest"),
            base=base,
            name="data.train_manifest",
        ),
        validation_manifest=_resolve_path(
            data.get("validation_manifest"),
            base=base,
            name="data.validation_manifest",
        ),
        output_dir=_resolve_path(run.get("output_dir"), base=base, name="run.output_dir"),
        resolution=(resolution[0], resolution[1]),
        crop_size_pixels=int(data.get("crop_size_pixels", 128)),
        crop_stride_pixels=int(data.get("crop_stride_pixels", 32)),
        augment=bool(data.get("augment", True)),
        directions=tuple(
            str(item).strip().lower()
            for item in data.get("directions", ["east", "west", "north", "south"])
        ),
        boundary_width=int(data.get("boundary_width", 3)),
        guide_length=int(data.get("guide_length", 24)),
        block_out_channels=block_out_channels,
        layers_per_block=int(model.get("layers_per_block", 2)),
        attention_levels=attention_levels,
        norm_num_groups=int(model.get("norm_num_groups", 8)),
        diffusion_steps=int(diffusion.get("train_steps", 1000)),
        inference_steps=int(diffusion.get("inference_steps", 100)),
        beta_schedule=str(diffusion.get("beta_schedule", "squaredcos_cap_v2")),
        epochs=int(run.get("epochs", 100)),
        batch_size=int(run.get("batch_size", 4)),
        learning_rate=float(run.get("learning_rate", 1e-4)),
        weight_decay=float(run.get("weight_decay", 1e-4)),
        num_workers=int(run.get("num_workers", 0)),
        seed=int(run.get("seed", 5132)),
        device=str(run.get("device", "auto")),
        ema_decay=float(run.get("ema_decay", 0.999)),
        gradient_clip_norm=float(run.get("gradient_clip_norm", 1.0)),
        checkpoint_every=int(run.get("checkpoint_every", 10)),
        preview_every=int(run.get("preview_every", 10)),
        early_stopping_patience=int(run.get("early_stopping_patience", 20)),
        initial_checkpoint=_resolve_path(
            run.get("initial_checkpoint"),
            base=base,
            name="run.initial_checkpoint",
            required=False,
        ),
        max_train_steps_per_epoch=(
            int(run["max_train_steps_per_epoch"])
            if run.get("max_train_steps_per_epoch") is not None
            else None
        ),
        max_validation_steps=(
            int(run["max_validation_steps"])
            if run.get("max_validation_steps") is not None
            else None
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: SemanticDiffusionConfig) -> None:
    scale = 2 ** (len(config.block_out_channels) - 1)
    if any(size <= 0 or size % scale for size in config.resolution):
        raise SemanticDiffusionConfigError(
            f"Resolution dimensions must be positive and divisible by {scale}"
        )
    if config.task == "block" and config.resolution[0] != config.resolution[1]:
        raise SemanticDiffusionConfigError("Block generation expects a square resolution")
    if config.task == "outpaint" and config.resolution[1] != config.resolution[0] * 2:
        raise SemanticDiffusionConfigError("Outpainting expects width to equal twice the height")
    if config.crop_size_pixels <= 0 or config.crop_stride_pixels <= 0:
        raise SemanticDiffusionConfigError("Crop size and stride must be positive")
    if len(config.attention_levels) != len(config.block_out_channels):
        raise SemanticDiffusionConfigError(
            "model.attention_levels must match model.block_out_channels"
        )
    if any(channel % config.norm_num_groups for channel in config.block_out_channels):
        raise SemanticDiffusionConfigError(
            "model.norm_num_groups must divide every block channel count"
        )
    allowed_directions = {"east", "west", "north", "south"}
    if not config.directions or any(item not in allowed_directions for item in config.directions):
        raise SemanticDiffusionConfigError("data.directions contains an invalid direction")
    if config.diffusion_steps < 2:
        raise SemanticDiffusionConfigError("diffusion.train_steps must be at least 2")
    if not 1 <= config.inference_steps <= config.diffusion_steps:
        raise SemanticDiffusionConfigError(
            "diffusion.inference_steps must be between 1 and train_steps"
        )
    if config.epochs <= 0 or config.batch_size <= 0 or config.num_workers < 0:
        raise SemanticDiffusionConfigError(
            "Epochs and batch size must be positive; workers cannot be negative"
        )
    if not 0.0 < config.ema_decay < 1.0:
        raise SemanticDiffusionConfigError("run.ema_decay must be between 0 and 1")
