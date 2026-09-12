from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import LayeredDiffusionConfig, load_layered_diffusion_config
from .data import (
    LAYER_NAMES,
    MODEL_CHANNELS,
    PROFILE_NAMES,
    SURFACE_CLASS_COUNT,
    SURFACE_NAMES,
    LayeredBlockDataset,
    model_space_to_layers,
)
from .model import autocast_context
from .preview import save_sheet as save_layered_sheet
from .surface_distribution_v2 import (
    _collect_real_class_maps,
    _coordinate_grid,
    _distribution_metrics,
    _sample_timesteps,
    _save_sheet as save_surface_sheet,
    _surface_class_weights,
)

POSITION_CHANNELS = 2
MODEL_INPUT_CHANNELS = MODEL_CHANNELS + POSITION_CHANNELS
PROFILE_BACKGROUND_WEIGHT = 0.05


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_diffusers():
    try:
        from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
    except ImportError as exc:
        raise RuntimeError("Install the diffusion dependencies first") from exc
    return UNet2DModel, DDPMScheduler, DDIMScheduler


def _block_types(config: LayeredDiffusionConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    down = tuple(
        "AttnDownBlock2D" if attention else "DownBlock2D"
        for attention in config.attention_levels
    )
    up = tuple(
        "AttnUpBlock2D" if attention else "UpBlock2D"
        for attention in reversed(config.attention_levels)
    )
    return down, up


def _build_model(config: LayeredDiffusionConfig) -> nn.Module:
    UNet2DModel, _DDPM, _DDIM = _require_diffusers()
    down, up = _block_types(config)
    return UNet2DModel(
        sample_size=config.resolution,
        in_channels=MODEL_INPUT_CHANNELS,
        out_channels=MODEL_CHANNELS,
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=down,
        up_block_types=up,
        norm_num_groups=config.norm_num_groups,
        add_attention=True,
    )


def _schedulers(config: LayeredDiffusionConfig):
    _UNet, DDPMScheduler, DDIMScheduler = _require_diffusers()
    noise = DDPMScheduler(
        num_train_timesteps=config.diffusion_steps,
        beta_schedule=config.beta_schedule,
        prediction_type="sample",
        clip_sample=True,
    )
    inference = DDIMScheduler.from_config(noise.config)
    return noise, inference


def _loader(
    dataset: LayeredBlockDataset,
    config: LayeredDiffusionConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    workers = max(0, int(config.num_workers))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=config.pin_memory,
        persistent_workers=workers > 0,
    )


def _batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = batch["x0"].to(device, non_blocking=device.type == "cuda")
    supervision = batch["valid_mask"].to(
        device, non_blocking=device.type == "cuda"
    )
    return x0, supervision


def _direct_x0_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    class_weights: torch.Tensor,
    channel_weights: tuple[float, ...],
) -> torch.Tensor:
    if prediction.shape[1] != MODEL_CHANNELS:
        raise ValueError(f"Expected {MODEL_CHANNELS} prediction channels")
    if supervision.shape != prediction.shape:
        raise ValueError(
            f"Supervision shape {tuple(supervision.shape)} does not match "
            f"prediction shape {tuple(prediction.shape)}"
        )

    surface_class = target[:, :SURFACE_CLASS_COUNT].argmax(dim=1)
    pixel_weights = class_weights[surface_class].unsqueeze(1)

    mask = supervision.to(dtype=prediction.dtype).clone()
    valid = mask[:, :1].clamp(0.0, 1.0)
    profile_start = MODEL_CHANNELS - len(PROFILE_NAMES)
    profile_background = valid * PROFILE_BACKGROUND_WEIGHT
    mask[:, profile_start:] = torch.maximum(
        mask[:, profile_start:],
        profile_background,
    )

    channel = prediction.new_tensor(channel_weights).reshape(1, -1, 1, 1)
    weighted = mask * pixel_weights * channel
    squared = (prediction.float() - target.float()).square() * weighted.float()
    return squared.sum() / weighted.sum().clamp_min(1.0)


class _EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        warm = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for name, value in model.state_dict().items():
            source = value.detach()
            if source.is_floating_point():
                self.shadow[name].mul_(warm).add_(source, alpha=1.0 - warm)
            else:
                self.shadow[name].copy_(source)

    def state_dict(self, *, cpu: bool = False) -> dict[str, Any]:
        shadow = self.shadow
        if cpu:
            shadow = {name: value.detach().cpu() for name, value in shadow.items()}
        return {"updates": self.updates, "shadow": shadow}

    def load_into(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def _checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: _EMA,
    optimizer: AdamW,
    epoch: int,
    updates: int,
    best_high_noise_loss: float,
    config: LayeredDiffusionConfig,
    class_weights: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "updates": int(updates),
            "model": model.state_dict(),
            "ema": ema.state_dict(cpu=True),
            "optimizer": optimizer.state_dict(),
            "best_high_noise_loss": float(best_high_noise_loss),
            "config": asdict(config),
            "layer_names": list(LAYER_NAMES),
            "prediction_type": "sample_x0",
            "position_channels": POSITION_CHANNELS,
            "class_weights": class_weights.tolist(),
        },
        path,
    )


@torch.inference_mode()
def _validation(
    model: nn.Module,
    loader: DataLoader,
    config: LayeredDiffusionConfig,
    noise_scheduler,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    coordinates = _coordinate_grid(config.resolution[0], device)
    uniform_total = 0.0
    high_total = 0.0
    batches = 0
    generator = torch.Generator(device=device).manual_seed(config.seed + 810_019)

    for batch in loader:
        x0, supervision = _batch_to_device(batch, device)
        batch_size = x0.shape[0]
        coords = coordinates.expand(batch_size, -1, -1, -1)
        noise = torch.randn(
            x0.shape,
            dtype=x0.dtype,
            device=device,
            generator=generator,
        )

        uniform_t = torch.randint(
            0,
            config.diffusion_steps,
            (batch_size,),
            device=device,
            generator=generator,
        )
        high_t = torch.randint(
            int(config.diffusion_steps * 0.90),
            config.diffusion_steps,
            (batch_size,),
            device=device,
            generator=generator,
        )

        uniform_noisy = noise_scheduler.add_noise(x0, noise, uniform_t)
        with autocast_context(config, device):
            uniform_prediction = model(
                torch.cat([uniform_noisy, coords], dim=1),
                uniform_t,
            ).sample
            uniform_loss = _direct_x0_loss(
                uniform_prediction,
                x0,
                supervision,
                class_weights,
                config.channel_loss_weights,
            )

        high_noisy = noise_scheduler.add_noise(x0, noise, high_t)
        with autocast_context(config, device):
            high_prediction = model(
                torch.cat([high_noisy, coords], dim=1),
                high_t,
            ).sample
            high_loss = _direct_x0_loss(
                high_prediction,
                x0,
                supervision,
                class_weights,
                config.channel_loss_weights,
            )

        uniform_total += float(uniform_loss)
        high_total += float(high_loss)
        batches += 1

    return uniform_total / max(batches, 1), high_total / max(batches, 1)


@torch.inference_mode()
def _sample(
    model: nn.Module,
    config: LayeredDiffusionConfig,
    *,
    batch_size: int,
    seed: int,
    inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    _noise, scheduler = _schedulers(config)
    scheduler.set_timesteps(inference_steps, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randn(
        (batch_size, MODEL_CHANNELS, *config.resolution),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    coordinates = _coordinate_grid(config.resolution[0], device).expand(
        batch_size, -1, -1, -1
    )

    model.eval()
    for timestep in scheduler.timesteps:
        with autocast_context(config, device):
            prediction = model(
                torch.cat([values, coordinates], dim=1),
                timestep,
            ).sample
        values = scheduler.step(
            prediction.float(),
            timestep,
            values,
            eta=0.0,
        ).prev_sample
    return values.clamp(-1.0, 1.0)


def _surface_classes(values: torch.Tensor) -> np.ndarray:
    return (
        values[:, :SURFACE_CLASS_COUNT]
        .argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


def _layered_summary(values: torch.Tensor) -> dict[str, Any]:
    decoded = model_space_to_layers(values)
    result: dict[str, Any] = {}

    surface = decoded["surface"].detach().cpu().numpy()
    counts = np.bincount(surface.reshape(-1), minlength=SURFACE_CLASS_COUNT)
    result["surface_class_fraction"] = {
        name: float(count / surface.size)
        for name, count in zip(SURFACE_NAMES, counts.tolist(), strict=True)
    }

    mask_names = (
        "road_underground",
        "road_elevated",
        "rail_underground",
        "rail_elevated",
    )
    result["auxiliary_pixel_fraction"] = {
        name: float(decoded[name].float().mean().item())
        for name in mask_names
    }

    building_height = decoded["building_height"]
    building_active = building_height > 0
    result["building_height_mean_norm"] = (
        float(building_height[building_active].mean().item())
        if building_active.any()
        else 0.0
    )

    profile_names = (
        "road_surface_offset_m",
        "road_underground_depth_m",
        "road_elevated_height_m",
        "rail_surface_offset_m",
        "rail_underground_depth_m",
        "rail_elevated_height_m",
    )
    profile_summary = {}
    for name in profile_names:
        tensor = decoded[name]
        active = tensor.abs() > 1e-6
        profile_summary[name] = {
            "active_fraction": float(active.float().mean().item()),
            "mean_abs_active_m": (
                float(tensor[active].abs().mean().item())
                if active.any()
                else 0.0
            ),
        }
    result["profiles"] = profile_summary
    return result


def train_distribution(
    config: LayeredDiffusionConfig,
    output: Path,
    *,
    max_epochs: int,
    max_hours: float,
    device_name: str,
    preview_every: int,
    checkpoint_every: int,
    final_samples: int,
    final_inference_steps: int,
    overwrite: bool,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    config = replace(
        config,
        output_dir=output,
        epochs=int(max_epochs),
        device=device_name,
    )

    _seed_everything(config.seed)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset = LayeredBlockDataset(
        config,
        config.train_manifest,
        augment=config.augment,
    )
    histogram_dataset = LayeredBlockDataset(
        config,
        config.train_manifest,
        augment=False,
    )
    validation_dataset = LayeredBlockDataset(
        config,
        config.validation_manifest,
        augment=False,
    )
    if not train_dataset or not validation_dataset:
        raise RuntimeError("Training and validation datasets must both be non-empty")

    counts, class_weight_values = _surface_class_weights(histogram_dataset, config)
    class_weights = torch.tensor(
        class_weight_values,
        dtype=torch.float32,
        device=device,
    )

    print("surface class frequencies / pixel weights:", flush=True)
    for name, count, weight in zip(
        SURFACE_NAMES,
        counts.tolist(),
        class_weight_values.tolist(),
        strict=True,
    ):
        print(f"  {name:16s} pixels={count:10d} weight={weight:.3f}", flush=True)

    train_loader = _loader(train_dataset, config, shuffle=True)
    validation_loader = _loader(validation_dataset, config, shuffle=False)

    model = _build_model(config).to(device)
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    ema = _EMA(model, config.ema_decay)
    noise_scheduler, _ = _schedulers(config)
    coordinates = _coordinate_grid(config.resolution[0], device)

    _write_json(
        output / "experiment.json",
        {
            "name": "layered-distribution-v2-full",
            "question": (
                "Does the x0 + XY + high-noise formulation that solved the "
                "whole-Singapore capacity test also learn a real multi-sample "
                "distribution while preserving the full 19-layer representation?"
            ),
            "prediction_type": "sample_x0",
            "position_channels": POSITION_CHANNELS,
            "high_noise_oversampling": True,
            "layer_names": list(LAYER_NAMES),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "max_epochs": max_epochs,
            "max_hours": max_hours,
            "config": asdict(config),
            "class_counts": dict(zip(SURFACE_NAMES, counts.tolist(), strict=True)),
            "class_weights": dict(
                zip(SURFACE_NAMES, class_weight_values.tolist(), strict=True)
            ),
        },
    )

    real_train = _collect_real_class_maps(histogram_dataset, limit=16)
    real_validation = _collect_real_class_maps(validation_dataset, limit=16)
    save_surface_sheet(np.stack(real_train), output / "real-train.png")
    save_surface_sheet(np.stack(real_validation), output / "real-validation.png")

    metric_fields = [
        "epoch",
        "updates",
        "train_loss",
        "validation_loss",
        "validation_high_noise_loss",
        "elapsed_seconds",
    ]
    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=metric_fields).writeheader()

    started = time.time()
    updates = 0
    best_high = math.inf
    best_epoch = 0
    epochs_completed = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        batches = 0

        for batch in train_loader:
            x0, supervision = _batch_to_device(batch, device)
            batch_size = x0.shape[0]
            timestep = _sample_timesteps(
                batch_size,
                config.diffusion_steps,
                device,
            )
            noise = torch.randn_like(x0)
            noised = noise_scheduler.add_noise(x0, noise, timestep)
            coords = coordinates.expand(batch_size, -1, -1, -1)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                prediction = model(
                    torch.cat([noised, coords], dim=1),
                    timestep,
                ).sample
                loss = _direct_x0_loss(
                    prediction,
                    x0,
                    supervision,
                    class_weights,
                    config.channel_loss_weights,
                )

            loss.backward()
            if config.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                )
            optimizer.step()
            ema.update(model)

            total += float(loss.detach())
            batches += 1
            updates += 1

        train_loss = total / max(batches, 1)

        eval_model = _build_model(config).to(device)
        ema.load_into(eval_model)
        validation_loss, validation_high = _validation(
            eval_model,
            validation_loader,
            config,
            noise_scheduler,
            class_weights,
            device,
        )
        del eval_model

        elapsed = time.time() - started
        row = {
            "epoch": epoch,
            "updates": updates,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_high_noise_loss": validation_high,
            "elapsed_seconds": round(elapsed, 2),
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=metric_fields).writerow(row)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} updates={updates} "
                f"train={train_loss:.6f} val={validation_loss:.6f} "
                f"high={validation_high:.6f} elapsed={elapsed / 3600:.2f}h",
                flush=True,
            )

        checkpoint_due = (
            epoch == 1
            or epoch % checkpoint_every == 0
            or epoch == max_epochs
        )
        if checkpoint_due:
            _checkpoint(
                output / "latest.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                updates=updates,
                best_high_noise_loss=min(best_high, validation_high),
                config=config,
                class_weights=class_weight_values,
            )
            if validation_high < best_high:
                best_high = validation_high
                best_epoch = epoch
                _checkpoint(
                    output / "best.pt",
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    epoch=epoch,
                    updates=updates,
                    best_high_noise_loss=best_high,
                    config=config,
                    class_weights=class_weight_values,
                )

        if epoch == 1 or epoch % preview_every == 0:
            preview_model = _build_model(config).to(device)
            ema.load_into(preview_model)
            generated = _sample(
                preview_model,
                config,
                batch_size=4,
                seed=config.seed + 200_019,
                inference_steps=config.inference_steps,
                device=device,
            )
            save_layered_sheet(
                generated,
                output / "previews-layered" / f"epoch-{epoch:04d}.png",
            )
            save_surface_sheet(
                _surface_classes(generated),
                output / "previews-surface" / f"epoch-{epoch:04d}.png",
            )
            del generated, preview_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        epochs_completed = epoch
        if elapsed >= max_hours * 3600:
            print(
                f"time budget reached after epoch {epoch}: {elapsed / 3600:.2f}h",
                flush=True,
            )
            break

    _checkpoint(
        output / "latest.pt",
        model=model,
        ema=ema,
        optimizer=optimizer,
        epoch=epochs_completed,
        updates=updates,
        best_high_noise_loss=best_high,
        config=config,
        class_weights=class_weight_values,
    )

    checkpoint_path = output / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    final_model = _build_model(config).to(device)
    ema_state = checkpoint["ema"]
    final_model.load_state_dict(ema_state.get("shadow", ema_state))
    final_model.eval()

    final_batches = []
    remaining = int(final_samples)
    batch_index = 0
    while remaining > 0:
        size = min(4, remaining)
        generated = _sample(
            final_model,
            config,
            batch_size=size,
            seed=config.seed + 900_019 + batch_index * 10_000,
            inference_steps=final_inference_steps,
            device=device,
        )
        final_batches.append(generated.detach().cpu())
        del generated
        remaining -= size
        batch_index += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_values = torch.cat(final_batches, dim=0)
    final_surface = _surface_classes(final_values)

    save_layered_sheet(final_values, output / "final-layered.png")
    save_surface_sheet(
        final_surface,
        output / "final-surface.png",
        labels=[f"generated {index + 1}" for index in range(len(final_surface))],
    )
    np.savez_compressed(
        output / "final-model-values.npz",
        model_values=final_values.numpy().astype(np.float16),
        layer_names=np.asarray(LAYER_NAMES),
    )

    all_train_real = _collect_real_class_maps(histogram_dataset)
    all_validation_real = _collect_real_class_maps(validation_dataset)
    distribution = _distribution_metrics(
        final_surface,
        all_train_real,
        all_validation_real,
    )
    _write_json(output / "distribution-metrics.json", distribution)

    nearest_maps = []
    nearest_labels = []
    for index, item in enumerate(distribution["nearest_real"]):
        nearest_maps.append(final_surface[index])
        nearest_labels.append(f"gen {index + 1}")
        nearest_maps.append(all_train_real[item["best_train_index"]])
        nearest_labels.append(
            f"train mIoU {item['best_train_mean_iou_64']:.3f}"
        )
    save_surface_sheet(
        np.stack(nearest_maps),
        output / "nearest-neighbours.png",
        labels=nearest_labels,
        columns=4,
    )

    layered = _layered_summary(final_values)
    _write_json(output / "layered-metrics.json", layered)

    total_hours = (time.time() - started) / 3600
    summary = {
        "output": str(output),
        "epochs_completed": epochs_completed,
        "updates": updates,
        "total_hours": round(total_hours, 3),
        "best_high_noise_loss": best_high,
        "best_epoch": best_epoch,
        "final_checkpoint": str(checkpoint_path),
        "final_inference_steps": final_inference_steps,
        "final_samples": final_samples,
        "mean_nearest_train_agreement_64": distribution[
            "mean_nearest_train_agreement_64"
        ],
        "max_nearest_train_agreement_64": distribution[
            "max_nearest_train_agreement_64"
        ],
        "mean_nearest_train_mean_iou_64": distribution[
            "mean_nearest_train_mean_iou_64"
        ],
        "max_nearest_train_mean_iou_64": distribution[
            "max_nearest_train_mean_iou_64"
        ],
        "mean_pairwise_generated_agreement_64": distribution[
            "mean_pairwise_generated_agreement_64"
        ],
        "final_surface_image": str(output / "final-surface.png"),
        "final_layered_image": str(output / "final-layered.png"),
        "nearest_neighbours_image": str(output / "nearest-neighbours.png"),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the successful x0/XY/high-noise formulation across the "
            "full 19-layer corrected Singapore corpus."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--max-hours", type=float, default=5.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preview-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--final-samples", type=int, default=8)
    parser.add_argument("--final-inference-steps", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_layered_diffusion_config(args.config)
    train_distribution(
        config,
        args.output,
        max_epochs=args.max_epochs,
        max_hours=args.max_hours,
        device_name=args.device,
        preview_every=args.preview_every,
        checkpoint_every=args.checkpoint_every,
        final_samples=args.final_samples,
        final_inference_steps=args.final_inference_steps,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
