from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .config import LayeredDiffusionConfig
from .data import LAYER_NAMES, MODEL_CHANNELS, layers_to_model_space
from .model import (
    autocast_context,
    build_inference_scheduler,
    build_model,
    build_noise_scheduler,
)
from .preview import render_triptych

_TIMESTEPS = (10, 100, 500, 900)
_COMPATIBILITY_FIELDS = (
    "resolution",
    "block_out_channels",
    "layers_per_block",
    "attention_levels",
    "norm_num_groups",
    "diffusion_steps",
    "beta_schedule",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _normalise(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return tuple(_normalise(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _checkpoint_config_mismatches(
    config: LayeredDiffusionConfig,
    checkpoint: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        return {"checkpoint.config": {"runtime": "present", "checkpoint": "missing"}}
    mismatches: dict[str, dict[str, Any]] = {}
    for name in _COMPATIBILITY_FIELDS:
        runtime_value = _normalise(getattr(config, name))
        saved_value = _normalise(saved.get(name))
        if runtime_value != saved_value:
            mismatches[name] = {
                "runtime": runtime_value,
                "checkpoint": saved_value,
            }
    return mismatches


def _checkpoint_state(
    checkpoint: dict[str, Any],
    source: str,
) -> dict[str, torch.Tensor]:
    if source == "raw":
        state = checkpoint.get("model")
    elif source == "ema":
        ema = checkpoint.get("ema")
        state = ema.get("shadow", ema) if isinstance(ema, dict) else None
    else:
        raise ValueError(f"Unknown checkpoint weight source: {source}")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain usable {source} weights")
    return state


def _load_model(
    config: LayeredDiffusionConfig,
    checkpoint: dict[str, Any],
    source: str,
    device: torch.device,
):
    model = build_model(config).to(device)
    model.load_state_dict(_checkpoint_state(checkpoint, source))
    model.eval()
    return model


def _crop_tensor(values: torch.Tensor, top: int, left: int, size: int) -> torch.Tensor:
    return values[..., top : top + size, left : left + size]


def _load_clean_tensor(
    sample_path: str | Path,
    config: LayeredDiffusionConfig,
    *,
    crop_top: int | None,
    crop_left: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(sample_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        direct_key = next((key for key in ("model_values", "x0") if key in archive), None)
        if direct_key is not None:
            values = torch.from_numpy(archive[direct_key].astype(np.float32))
            if values.ndim == 4 and values.shape[0] == 1:
                values = values[0]
            if values.ndim != 3 or values.shape[0] != MODEL_CHANNELS:
                raise ValueError(
                    f"{direct_key} must have shape [{MODEL_CHANNELS},H,W], found {tuple(values.shape)}"
                )
            source = {"path": str(path), "mode": direct_key}
        else:
            required = (
                "layers",
                "road_vertical_masks",
                "rail_vertical_masks",
                "road_vertical_profiles_m",
                "rail_vertical_profiles_m",
            )
            missing = [name for name in required if name not in archive]
            if missing:
                raise ValueError(
                    "Sample archive must contain model_values/x0 or the corpus arrays: "
                    + ", ".join(missing)
                )
            arrays = {
                name: torch.from_numpy(archive[name].astype(np.float32))
                for name in required
            }
            height, width = arrays["layers"].shape[-2:]
            size = min(config.crop_size_pixels, height, width)
            top = (height - size) // 2 if crop_top is None else int(crop_top)
            left = (width - size) // 2 if crop_left is None else int(crop_left)
            if top < 0 or left < 0 or top + size > height or left + size > width:
                raise ValueError(
                    f"Requested crop top={top}, left={left}, size={size} is outside {height}x{width}"
                )
            arrays = {
                name: _crop_tensor(value, top, left, size)
                for name, value in arrays.items()
            }
            values = layers_to_model_space(
                arrays["layers"],
                arrays["road_vertical_masks"],
                arrays["rail_vertical_masks"],
                arrays["road_vertical_profiles_m"],
                arrays["rail_vertical_profiles_m"],
                config,
            )
            source = {
                "path": str(path),
                "mode": "corpus-arrays",
                "crop_top": top,
                "crop_left": left,
                "crop_size": size,
            }

    if values.shape[-2:] != config.resolution:
        values = F.interpolate(
            values.unsqueeze(0),
            size=config.resolution,
            mode="nearest",
        )[0]
        source["resized_to"] = list(config.resolution)
    return values.float().clamp(-1.0, 1.0), source


def _tensor_stats(values: torch.Tensor) -> dict[str, Any]:
    detached = values.detach().float()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    result = {
        "shape": list(detached.shape),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "inf_count": int(torch.isinf(detached).sum().item()),
    }
    if finite_values.numel():
        result.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std(unbiased=False).item()),
            }
        )
    return result


def _per_channel_stats(values: torch.Tensor) -> dict[str, Any]:
    return {
        name: _tensor_stats(values[index])
        for index, name in enumerate(LAYER_NAMES)
    }


def _prediction_to_x0(
    scheduler,
    noisy: torch.Tensor,
    prediction: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alpha = scheduler.alphas_cumprod.to(device=noisy.device, dtype=noisy.dtype)[timesteps]
    alpha = alpha.reshape(-1, 1, 1, 1)
    beta = 1.0 - alpha
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return (noisy - beta.sqrt() * prediction) / alpha.sqrt()
    if prediction_type == "sample":
        return prediction
    if prediction_type == "v_prediction":
        return alpha.sqrt() * noisy - beta.sqrt() * prediction
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def _mse_by_channel(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    squared = (prediction.float() - target.float()).square().mean(dim=(0, 2, 3))
    return {
        name: float(squared[index].item())
        for index, name in enumerate(LAYER_NAMES)
    }


def _labelled_panel(label: str, image: Image.Image) -> Image.Image:
    header = 20
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text((4, 4), label, fill="black")
    return canvas


def _row(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    labelled = [_labelled_panel(label, image) for label, image in panels]
    width = sum(image.width for image in labelled)
    height = max(image.height for image in labelled)
    canvas = Image.new("RGB", (width, height), "white")
    left = 0
    for image in labelled:
        canvas.paste(image, (left, 0))
        left += image.width
    return canvas


def _stack(rows: list[Image.Image]) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), "white")
    top = 0
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height
    return canvas


def _render_error(error: torch.Tensor) -> Image.Image:
    values = error.detach().float().mean(dim=0).cpu().numpy()
    scale = max(float(np.percentile(values, 99)), 1e-8)
    pixels = np.round(np.clip(values / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").convert("RGB")
    return image.resize((error.shape[-1] * 3, error.shape[-2]), Image.Resampling.NEAREST)


def _denoise_case(
    models: dict[str, Any],
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: int,
    scheduler,
    config: LayeredDiffusionConfig,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    batch_timestep = torch.full((1,), timestep, device=device, dtype=torch.long)
    noisy = scheduler.add_noise(clean, noise, batch_timestep)
    panels: list[tuple[str, Image.Image]] = [
        ("clean", render_triptych(clean[0].cpu())),
        (f"noisy t={timestep}", render_triptych(noisy[0].detach().cpu().clamp(-1.0, 1.0))),
    ]
    result: dict[str, Any] = {}
    for source, model in models.items():
        with torch.inference_mode(), autocast_context(config, device):
            model_input = scheduler.scale_model_input(noisy, batch_timestep)
            prediction = model(model_input, batch_timestep).sample
        prediction = prediction.float()
        reconstructed = _prediction_to_x0(scheduler, noisy.float(), prediction, batch_timestep)
        error = (reconstructed - clean.float()).abs()
        result[source] = {
            "noise_prediction_mse": float((prediction - noise.float()).square().mean().item()),
            "x0_reconstruction_mse": float((reconstructed - clean.float()).square().mean().item()),
            "noise_prediction_mse_per_channel": _mse_by_channel(prediction, noise),
            "x0_reconstruction_mse_per_channel": _mse_by_channel(reconstructed, clean),
            "prediction_stats": _tensor_stats(prediction),
            "reconstruction_stats": _tensor_stats(reconstructed),
        }
        panels.append(
            (
                f"{source} x0",
                render_triptych(reconstructed[0].detach().cpu().clamp(-1.0, 1.0)),
            )
        )
        panels.append((f"{source} abs error", _render_error(error[0].cpu())))
    _row(panels).save(output / f"denoise-t{timestep:04d}.png", optimize=True)
    return result


def _initial_noise(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


def _reverse_trace(
    model,
    config: LayeredDiffusionConfig,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[tuple[str, torch.Tensor]]]:
    train_scheduler = build_noise_scheduler(config)
    scheduler = build_inference_scheduler(config, train_scheduler)
    scheduler.set_timesteps(config.inference_steps, device=device)
    sample = _initial_noise((1, MODEL_CHANNELS, *config.resolution), seed, device)
    count = len(scheduler.timesteps)
    capture = {0, count // 4, count // 2, (3 * count) // 4, count - 1}
    records: list[dict[str, Any]] = []
    snapshots: list[tuple[str, torch.Tensor]] = []
    with torch.inference_mode():
        for index, timestep in enumerate(scheduler.timesteps):
            model_input = scheduler.scale_model_input(sample, timestep)
            with autocast_context(config, device):
                prediction = model(model_input, timestep).sample
            step = scheduler.step(
                prediction.float(),
                timestep,
                sample,
                eta=0.0,
            )
            sample = step.prev_sample
            record = {
                "step": index,
                "timestep": int(timestep.item()),
                **_tensor_stats(sample),
            }
            records.append(record)
            if index in capture:
                snapshots.append((f"step {index} / t={int(timestep.item())}", sample.detach().cpu()))
    return sample, records, snapshots


def diagnose_layered_diffusion(
    config: LayeredDiffusionConfig,
    checkpoint_path: str | Path,
    sample_path: str | Path,
    destination: str | Path,
    *,
    device_name: str | None = None,
    seed: int | None = None,
    crop_top: int | None = None,
    crop_left: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Diagnostic output is not empty: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    device = _select_device(device_name or config.device)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mismatches = _checkpoint_config_mismatches(config, checkpoint)
    models = {
        source: _load_model(config, checkpoint, source, device)
        for source in ("raw", "ema")
    }
    clean_cpu, sample_source = _load_clean_tensor(
        sample_path,
        config,
        crop_top=crop_top,
        crop_left=crop_left,
    )
    clean = clean_cpu.unsqueeze(0).to(device)

    train_scheduler = build_noise_scheduler(config)
    sample_scheduler = build_inference_scheduler(config, train_scheduler)
    _write_json(destination / "scheduler-train.json", dict(train_scheduler.config))
    _write_json(destination / "scheduler-sample.json", dict(sample_scheduler.config))
    _write_json(
        destination / "checkpoint-keys.json",
        {
            "top_level_keys": sorted(checkpoint.keys()),
            "epoch": checkpoint.get("epoch"),
            "best_validation_loss": checkpoint.get("best_validation_loss"),
            "raw_parameter_tensors": len(_checkpoint_state(checkpoint, "raw")),
            "ema_parameter_tensors": len(_checkpoint_state(checkpoint, "ema")),
            "ema_updates": checkpoint.get("ema", {}).get("updates")
            if isinstance(checkpoint.get("ema"), dict)
            else None,
            "config_mismatches": mismatches,
        },
    )
    _write_json(
        destination / "model-space-stats.json",
        {
            "sample_source": sample_source,
            "overall": _tensor_stats(clean_cpu),
            "channels": _per_channel_stats(clean_cpu),
        },
    )

    generator = torch.Generator(device="cpu").manual_seed(int(config.seed + 300_000))
    known_noise = torch.randn(clean_cpu.unsqueeze(0).shape, generator=generator).to(device)
    denoise: dict[str, Any] = {}
    for timestep in _TIMESTEPS:
        if timestep >= config.diffusion_steps:
            continue
        denoise[str(timestep)] = _denoise_case(
            models,
            clean,
            known_noise,
            timestep,
            train_scheduler,
            config,
            device,
            destination,
        )

    sample_seed = int(config.seed + 400_000 if seed is None else seed)
    reverse: dict[str, Any] = {}
    final_samples: dict[str, torch.Tensor] = {}
    trajectory_rows: list[Image.Image] = []
    for source, model in models.items():
        final, records, snapshots = _reverse_trace(model, config, device, sample_seed)
        final_samples[source] = final.detach().cpu()
        reverse[source] = {
            "steps": records,
            "final_stats": _tensor_stats(final),
        }
        trajectory_rows.append(
            _row(
                [
                    (f"{source} {label}", render_triptych(values[0].clamp(-1.0, 1.0)))
                    for label, values in snapshots
                ]
            )
        )
    _stack(trajectory_rows).save(destination / "reverse-trajectory.png", optimize=True)
    _row(
        [
            (source, render_triptych(values[0].clamp(-1.0, 1.0)))
            for source, values in final_samples.items()
        ]
    ).save(destination / "raw-vs-ema.png", optimize=True)

    diagnostics = {
        "checkpoint": str(checkpoint_path),
        "sample": sample_source,
        "device": str(device),
        "seed": sample_seed,
        "prediction_type": train_scheduler.config.prediction_type,
        "train_scheduler_class": type(train_scheduler).__name__,
        "sample_scheduler_class": type(sample_scheduler).__name__,
        "runtime_config": asdict(config),
        "checkpoint_config_mismatches": mismatches,
        "clean": _tensor_stats(clean),
        "denoise": denoise,
        "reverse": reverse,
    }
    _write_json(destination / "diagnostics.json", diagnostics)
    return {
        "output": str(destination),
        "checkpoint": str(checkpoint_path),
        "sample": str(Path(sample_path).expanduser().resolve()),
        "device": str(device),
        "prediction_type": train_scheduler.config.prediction_type,
        "checkpoint_config_mismatches": mismatches,
        "files": sorted(path.name for path in destination.iterdir()),
    }


__all__ = ["diagnose_layered_diffusion"]
