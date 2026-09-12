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
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import LayeredDiffusionConfig, load_layered_diffusion_config
from .data import LayeredBlockDataset, SURFACE_CLASS_COUNT, SURFACE_NAMES
from .model import autocast_context

POSITION_CHANNELS = 2
MODEL_INPUT_CHANNELS = SURFACE_CLASS_COUNT + POSITION_CHANNELS

PALETTE = np.asarray(
    [
        (226, 221, 209),  # terrain
        (111, 174, 105),  # vegetation
        (137, 142, 148),  # building
        (215, 58, 48),    # road major
        (239, 116, 66),   # road secondary
        (246, 180, 90),   # road local
        (85, 176, 194),   # rail
        (78, 151, 211),   # water
    ],
    dtype=np.uint8,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _coordinate_grid(resolution: int, device: torch.device) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, resolution, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx, yy], dim=0).unsqueeze(0)


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
        out_channels=SURFACE_CLASS_COUNT,
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


def _surface(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = batch["x0"][:, :SURFACE_CLASS_COUNT].to(
        device, non_blocking=device.type == "cuda"
    )
    supervision = batch["valid_mask"][:, :SURFACE_CLASS_COUNT].to(
        device, non_blocking=device.type == "cuda"
    )
    return x0, supervision


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


def _surface_class_weights(
    dataset: LayeredBlockDataset,
    config: LayeredDiffusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(SURFACE_CLASS_COUNT, dtype=np.int64)
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(8, config.batch_size * 2)),
        shuffle=False,
        num_workers=max(0, min(4, config.num_workers)),
        pin_memory=False,
    )
    for batch in loader:
        x0 = batch["x0"][:, :SURFACE_CLASS_COUNT]
        valid = batch["valid_mask"][:, 0] > 0
        classes = x0.argmax(dim=1)
        values = classes[valid]
        counts += np.bincount(
            values.reshape(-1).numpy(),
            minlength=SURFACE_CLASS_COUNT,
        )

    fractions = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    raw = np.power(np.maximum(fractions, 0.005), -0.35)
    raw = np.minimum(raw, 3.0)
    raw /= np.sum(raw * fractions)
    return counts, raw


def _sample_timesteps(
    batch_size: int,
    diffusion_steps: int,
    device: torch.device,
) -> torch.Tensor:
    # The whole-city capacity experiment only learned the global prior once
    # high-noise states were deliberately revisited. Keep that same curriculum:
    # 50% high noise, 30% medium, 20% low.
    bucket = torch.rand(batch_size, device=device)
    timesteps = torch.empty(batch_size, dtype=torch.long, device=device)

    high = bucket < 0.50
    mid = (bucket >= 0.50) & (bucket < 0.80)
    low = bucket >= 0.80

    if high.any():
        timesteps[high] = torch.randint(
            int(diffusion_steps * 0.75),
            diffusion_steps,
            (int(high.sum().item()),),
            device=device,
        )
    if mid.any():
        timesteps[mid] = torch.randint(
            int(diffusion_steps * 0.35),
            int(diffusion_steps * 0.75),
            (int(mid.sum().item()),),
            device=device,
        )
    if low.any():
        timesteps[low] = torch.randint(
            0,
            int(diffusion_steps * 0.35),
            (int(low.sum().item()),),
            device=device,
        )
    return timesteps


def _direct_x0_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    class_weights: torch.Tensor,
    channel_weights: tuple[float, ...],
) -> torch.Tensor:
    classes = target.argmax(dim=1)
    pixel_weights = class_weights[classes].unsqueeze(1)
    channel = prediction.new_tensor(channel_weights[:SURFACE_CLASS_COUNT]).reshape(
        1, -1, 1, 1
    )
    mask = supervision.to(dtype=prediction.dtype)
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
            shadow = {k: v.detach().cpu() for k, v in shadow.items()}
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
    update: int,
    best_high_noise_loss: float,
    config: LayeredDiffusionConfig,
    class_weights: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "update": int(update),
            "model": model.state_dict(),
            "ema": ema.state_dict(cpu=True),
            "optimizer": optimizer.state_dict(),
            "best_high_noise_loss": float(best_high_noise_loss),
            "config": asdict(config),
            "surface_names": list(SURFACE_NAMES),
            "position_channels": POSITION_CHANNELS,
            "prediction_type": "sample_x0",
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
    generator = torch.Generator(device=device).manual_seed(config.seed + 810_001)

    for batch in loader:
        x0, supervision = _surface(batch, device)
        batch_size = x0.shape[0]
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

        coords = coordinates.expand(batch_size, -1, -1, -1)

        uniform_noisy = noise_scheduler.add_noise(x0, noise, uniform_t)
        with autocast_context(config, device):
            uniform_pred = model(
                torch.cat([uniform_noisy, coords], dim=1),
                uniform_t,
            ).sample
            uniform_loss = _direct_x0_loss(
                uniform_pred,
                x0,
                supervision,
                class_weights,
                config.channel_loss_weights,
            )

        high_noisy = noise_scheduler.add_noise(x0, noise, high_t)
        with autocast_context(config, device):
            high_pred = model(
                torch.cat([high_noisy, coords], dim=1),
                high_t,
            ).sample
            high_loss = _direct_x0_loss(
                high_pred,
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
    _noise_scheduler, scheduler = _schedulers(config)
    scheduler.set_timesteps(inference_steps, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randn(
        (batch_size, SURFACE_CLASS_COUNT, *config.resolution),
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


def _classes(values: torch.Tensor) -> np.ndarray:
    return values.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)


def _class_image(classes: np.ndarray) -> Image.Image:
    return Image.fromarray(PALETTE[classes.astype(np.int64)])


def _save_sheet(
    class_maps: np.ndarray,
    path: Path,
    *,
    labels: list[str] | None = None,
    columns: int = 4,
) -> None:
    if class_maps.ndim == 2:
        class_maps = class_maps[None]
    images = [_class_image(item) for item in class_maps]
    width, height = images[0].size
    header = 24 if labels else 0
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * width, rows * (height + header)), "white")
    draw = ImageDraw.Draw(canvas)

    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        x = col * width
        y = row * (height + header)
        if labels:
            draw.text((x + 5, y + 6), labels[index], fill="black")
        canvas.paste(image, (x, y + header))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def _collect_real_class_maps(
    dataset: LayeredBlockDataset,
    *,
    limit: int | None = None,
) -> list[np.ndarray]:
    maps: list[np.ndarray] = []
    count = len(dataset) if limit is None else min(len(dataset), limit)
    for index in range(count):
        x0 = dataset[index]["x0"][:SURFACE_CLASS_COUNT]
        maps.append(x0.argmax(dim=0).numpy().astype(np.uint8))
    return maps


def _downsample_classes(classes: np.ndarray, size: int = 64) -> np.ndarray:
    tensor = torch.from_numpy(classes.astype(np.float32)).unsqueeze(1)
    small = F.interpolate(tensor, size=(size, size), mode="nearest")
    return small[:, 0].numpy().astype(np.uint8)


def _distribution_metrics(
    generated: np.ndarray,
    train_real: list[np.ndarray],
    validation_real: list[np.ndarray],
) -> dict[str, Any]:
    train = np.stack(train_real)
    validation = np.stack(validation_real)

    generated_small = _downsample_classes(generated)
    train_small = _downsample_classes(train)
    validation_small = _downsample_classes(validation)

    nearest = []
    for index, item in enumerate(generated_small):
        train_agreement = (train_small == item[None]).mean(axis=(1, 2))
        validation_agreement = (validation_small == item[None]).mean(axis=(1, 2))
        best_train = int(np.argmax(train_agreement))
        best_validation = int(np.argmax(validation_agreement))
        nearest.append(
            {
                "sample": index,
                "best_train_index": best_train,
                "best_train_agreement_64": float(train_agreement[best_train]),
                "best_validation_index": best_validation,
                "best_validation_agreement_64": float(
                    validation_agreement[best_validation]
                ),
            }
        )

    pairwise = []
    for first in range(len(generated_small)):
        for second in range(first + 1, len(generated_small)):
            pairwise.append(
                float((generated_small[first] == generated_small[second]).mean())
            )

    def fractions(values: np.ndarray) -> dict[str, float]:
        counts = np.bincount(
            values.reshape(-1),
            minlength=SURFACE_CLASS_COUNT,
        )
        return {
            name: float(count / values.size)
            for name, count in zip(SURFACE_NAMES, counts.tolist(), strict=True)
        }

    generated_per_sample = [fractions(sample) for sample in generated]
    return {
        "generated_class_fraction": fractions(generated),
        "train_class_fraction": fractions(train),
        "validation_class_fraction": fractions(validation),
        "generated_per_sample_class_fraction": generated_per_sample,
        "nearest_real": nearest,
        "mean_nearest_train_agreement_64": float(
            np.mean([item["best_train_agreement_64"] for item in nearest])
        ),
        "max_nearest_train_agreement_64": float(
            np.max([item["best_train_agreement_64"] for item in nearest])
        ),
        "mean_nearest_validation_agreement_64": float(
            np.mean([item["best_validation_agreement_64"] for item in nearest])
        ),
        "mean_pairwise_generated_agreement_64": (
            float(np.mean(pairwise)) if pairwise else 1.0
        ),
        "min_pairwise_generated_agreement_64": (
            float(np.min(pairwise)) if pairwise else 1.0
        ),
    }


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
    print("surface class frequencies / loss weights:", flush=True)
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
            "name": "layered-distribution-v2",
            "purpose": (
                "Test whether the x0 + XY + high-noise formulation that memorised "
                "whole Singapore can learn a distribution across the corrected "
                "Singapore tile corpus."
            ),
            "surface_names": list(SURFACE_NAMES),
            "prediction_type": "sample_x0",
            "position_channels": POSITION_CHANNELS,
            "high_noise_oversampling": True,
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

    # Save real examples so the generated sheets can be judged beside the data.
    train_examples = _collect_real_class_maps(histogram_dataset, limit=16)
    validation_examples = _collect_real_class_maps(validation_dataset, limit=16)
    _save_sheet(np.stack(train_examples), output / "real-train.png")
    _save_sheet(np.stack(validation_examples), output / "real-validation.png")

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
    update = 0
    best_high = math.inf
    best_epoch = 0
    epochs_completed = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        batches = 0

        for batch in train_loader:
            x0, supervision = _surface(batch, device)
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
            update += 1

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
            "updates": update,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_high_noise_loss": validation_high,
            "elapsed_seconds": round(elapsed, 2),
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=metric_fields).writerow(row)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} updates={update} "
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
                update=update,
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
                    update=update,
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
                seed=config.seed + 200_000,
                inference_steps=config.inference_steps,
                device=device,
            )
            preview_classes = _classes(generated)
            _save_sheet(
                preview_classes,
                output / "previews" / f"epoch-{epoch:04d}.png",
                labels=[f"sample {i + 1}" for i in range(len(preview_classes))],
            )
            del generated, preview_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        epochs_completed = epoch

        # Reserve time for the full 1000-step evaluation and packaging.
        if elapsed >= max_hours * 3600:
            print(
                f"time budget reached after epoch {epoch}: {elapsed / 3600:.2f}h",
                flush=True,
            )
            break

    # Always preserve the actual final state, even if the wall-clock budget ended
    # between normal checkpoint intervals.
    _checkpoint(
        output / "latest.pt",
        model=model,
        ema=ema,
        optimizer=optimizer,
        epoch=epochs_completed,
        update=update,
        best_high_noise_loss=best_high,
        config=config,
        class_weights=class_weight_values,
    )

    # For the final scientific readout use the best high-noise EMA checkpoint.
    checkpoint_path = output / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = output / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    final_model = _build_model(config).to(device)
    ema_state = checkpoint["ema"]
    final_model.load_state_dict(ema_state.get("shadow", ema_state))
    final_model.eval()

    batches = []
    remaining = int(final_samples)
    sample_batch = min(4, final_samples)
    seed = config.seed + 900_000
    batch_index = 0
    while remaining > 0:
        size = min(sample_batch, remaining)
        generated = _sample(
            final_model,
            config,
            batch_size=size,
            seed=seed + batch_index * 10_000,
            inference_steps=final_inference_steps,
            device=device,
        )
        batches.append(_classes(generated))
        del generated
        remaining -= size
        batch_index += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_classes = np.concatenate(batches, axis=0)
    _save_sheet(
        final_classes,
        output / "final-samples.png",
        labels=[f"generated {i + 1}" for i in range(len(final_classes))],
    )

    train_real = _collect_real_class_maps(histogram_dataset)
    validation_real = _collect_real_class_maps(validation_dataset)
    distribution = _distribution_metrics(
        final_classes,
        train_real,
        validation_real,
    )
    _write_json(output / "distribution-metrics.json", distribution)

    # Pair each generated sample with its nearest training tile at 64x64. This
    # makes memorisation/mode collapse visible rather than hiding it in one score.
    nearest_maps = []
    nearest_labels = []
    for index, item in enumerate(distribution["nearest_real"]):
        nearest_maps.append(final_classes[index])
        nearest_labels.append(f"gen {index + 1}")
        nearest_maps.append(train_real[item["best_train_index"]])
        nearest_labels.append(
            f"nearest train {item['best_train_agreement_64']:.3f}"
        )
    _save_sheet(
        np.stack(nearest_maps),
        output / "nearest-neighbours.png",
        labels=nearest_labels,
        columns=4,
    )

    summary = {
        "output": str(output),
        "epochs_completed": epochs_completed,
        "updates": update,
        "training_hours": round((time.time() - started) / 3600, 3),
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
        "mean_pairwise_generated_agreement_64": distribution[
            "mean_pairwise_generated_agreement_64"
        ],
        "final_samples_image": str(output / "final-samples.png"),
        "nearest_neighbours_image": str(output / "nearest-neighbours.png"),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the successful whole-city x0/XY/high-noise formulation across "
            "the corrected Singapore surface-layout corpus."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--max-hours", type=float, default=5.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preview-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--final-samples", type=int, default=16)
    parser.add_argument("--final-inference-steps", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
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
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
