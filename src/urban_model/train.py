from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from urban_dataset.obj_export import export_city_state_obj

from .conditioning import (
    CityConditionedDataset,
    build_model_input,
    check_conditioned_data,
    city_mix_tensor,
    diffusion_target,
    parse_city_mix,
    preview_city_mix,
)
from .config import LayeredDiffusionConfig
from .data import LAYER_NAMES, model_space_to_layers
from .model import (
    ModelEMA,
    autocast_context,
    build_model,
    build_noise_scheduler,
    sample_model,
    weighted_diffusion_loss,
)
from .preview import render_triptych, save_sheet
from .vectorize import generated_layers_to_city_state

PREVIEW_SEED_OFFSET = 200_000


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _environment(device: torch.device) -> dict[str, Any]:
    result = {
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["cuda"] = torch.version.cuda
    return result


def _loader(
    dataset: CityConditionedDataset,
    config: LayeredDiffusionConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    workers = max(0, int(config.num_workers))
    sampler = None
    if shuffle and config.balance_cities:
        counts = Counter(dataset.city_indices)
        if len(counts) > 1:
            weights = [1.0 / counts[index] for index in dataset.city_indices]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            shuffle = False
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=config.pin_memory,
        persistent_workers=workers > 0,
    )


def _to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=device.type == "cuda")


def _preview_seed(config: LayeredDiffusionConfig) -> int:
    return int(config.seed + PREVIEW_SEED_OFFSET)


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_loss: float,
    config: LayeredDiffusionConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "ema": ema.state_dict(cpu=True),
        "optimizer": optimizer.state_dict(),
        "best_validation_loss": float(best_validation_loss),
        "layer_names": LAYER_NAMES,
        "config": asdict(config),
    }
    if scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def _validation_loss(
    model: nn.Module,
    loader: DataLoader,
    config: LayeredDiffusionConfig,
    noise_scheduler,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    generator = torch.Generator(device=device).manual_seed(config.seed + 100_000)
    with torch.inference_mode():
        for batch in loader:
            x0 = _to_device(batch["x0"], device)
            city = _to_device(batch["city"], device)
            supervision = _to_device(batch["valid_mask"], device)
            timesteps = torch.randint(
                0,
                config.diffusion_steps,
                (x0.shape[0],),
                device=device,
                generator=generator,
            )
            noise = torch.randn(
                x0.shape,
                dtype=x0.dtype,
                device=device,
                generator=generator,
            )
            noisy = noise_scheduler.add_noise(x0, noise, timesteps)
            target = diffusion_target(
                noise_scheduler,
                x0,
                noise,
                timesteps,
                config.prediction_type,
            )
            model_input = build_model_input(noisy, city, config)
            with autocast_context(config, device):
                prediction = model(model_input, timesteps).sample
                loss = weighted_diffusion_loss(
                    prediction,
                    target,
                    supervision,
                    config.channel_loss_weights,
                )
            total += float(loss.detach())
            batches += 1
    return total / max(batches, 1)


def train_layered_diffusion(
    config: LayeredDiffusionConfig,
    *,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    resume: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if epochs is not None:
        config = replace(config, epochs=int(epochs))
    if batch_size is not None:
        config = replace(config, batch_size=int(batch_size))

    _seed_everything(config.seed)
    device = _select_device(device_name or config.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset = CityConditionedDataset(
        config,
        config.train_manifest,
        augment=config.augment,
    )
    validation_dataset = CityConditionedDataset(
        config,
        config.validation_manifest,
        augment=False,
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("Training requires non-empty train and validation datasets")

    output = config.output_dir
    if overwrite and output.exists():
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    train_loader = _loader(train_dataset, config, shuffle=True)
    validation_loader = _loader(validation_dataset, config, shuffle=False)
    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and config.precision == "fp16",
    )
    ema = ModelEMA(model, config.ema_decay)
    noise_scheduler = build_noise_scheduler(config)

    start_epoch = 1
    best = float("inf")
    if resume is not None:
        checkpoint = torch.load(
            Path(resume).expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )
        _check_checkpoint_config(config, checkpoint)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler.is_enabled() and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_validation_loss", best))

    preview_city = preview_city_mix(config, 4, device)
    _write_json(output / "config.json", asdict(config))
    _write_json(output / "environment.json", _environment(device))
    _write_json(
        output / "preview.json",
        {
            "seed": _preview_seed(config),
            "cities": [config.city_names[index % len(config.city_names)] for index in range(4)],
        },
    )
    metrics_path = output / "metrics.jsonl"
    started = time.time()
    patience = 0
    epochs_completed = start_epoch - 1

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}", leave=False)
        for batch in progress:
            x0 = _to_device(batch["x0"], device)
            city = _to_device(batch["city"], device)
            supervision = _to_device(batch["valid_mask"], device)
            timesteps = torch.randint(
                0,
                config.diffusion_steps,
                (x0.shape[0],),
                device=device,
            )
            noise = torch.randn_like(x0)
            noisy = noise_scheduler.add_noise(x0, noise, timesteps)
            target = diffusion_target(
                noise_scheduler,
                x0,
                noise,
                timesteps,
                config.prediction_type,
            )
            model_input = build_model_input(noisy, city, config)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                prediction = model(model_input, timesteps).sample
                loss = weighted_diffusion_loss(
                    prediction,
                    target,
                    supervision,
                    config.channel_loss_weights,
                )
            scaler.scale(loss).backward()
            if config.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total += float(loss.detach())
            batches += 1
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        train_loss = total / max(batches, 1)
        validation_model = ema.copy_model(config, device)
        validation_loss = _validation_loss(
            validation_model,
            validation_loader,
            config,
            noise_scheduler,
            device,
        )
        del validation_model

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        checkpoint_args = {
            "model": model,
            "ema": ema,
            "optimizer": optimizer,
            "scaler": scaler,
            "epoch": epoch,
            "best_validation_loss": min(best, validation_loss),
            "config": config,
        }
        _save_checkpoint(output / "latest.pt", **checkpoint_args)
        if epoch % config.checkpoint_every == 0:
            _save_checkpoint(
                output / "checkpoints" / f"epoch-{epoch:04d}.pt",
                **checkpoint_args,
            )
        if validation_loss < best:
            best = validation_loss
            patience = 0
            checkpoint_args["best_validation_loss"] = best
            _save_checkpoint(output / "best.pt", **checkpoint_args)
        else:
            patience += 1

        if epoch == 1 or epoch % config.preview_every == 0:
            preview_model = ema.copy_model(config, device)
            generated = sample_model(
                preview_model,
                config,
                city=preview_city,
                batch_size=4,
                device=device,
                seed=_preview_seed(config),
            )
            save_sheet(generated, output / "previews" / f"epoch-{epoch:04d}.png")
            del generated, preview_model

        print(f"epoch={epoch} train={train_loss:.4f} validation={validation_loss:.4f}")
        epochs_completed = epoch
        if patience >= config.early_stopping_patience:
            break

    summary = {
        "output_dir": str(output),
        "device": str(device),
        "precision": config.precision,
        "prediction_type": config.prediction_type,
        "cities": list(config.city_names),
        "epochs_completed": epochs_completed,
        "best_validation_loss": best,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_city_samples": train_dataset.city_counts,
        "validation_city_samples": validation_dataset.city_counts,
        "layers": list(LAYER_NAMES),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "preview_seed": _preview_seed(config),
    }
    _write_json(output / "summary.json", summary)
    return summary


def _normalise(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalise(item) for item in value)
    return value


def _check_checkpoint_config(
    config: LayeredDiffusionConfig,
    checkpoint: dict[str, Any],
) -> None:
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint does not contain its training config")
    fields = (
        "city_names",
        "resolution",
        "block_out_channels",
        "layers_per_block",
        "attention_levels",
        "norm_num_groups",
        "coordinate_channels",
        "diffusion_steps",
        "beta_schedule",
        "prediction_type",
    )
    mismatches = [
        name
        for name in fields
        if _normalise(saved.get(name)) != _normalise(getattr(config, name))
    ]
    if mismatches:
        raise ValueError("Checkpoint config mismatch: " + ", ".join(mismatches))


def _load_sample_model(
    config: LayeredDiffusionConfig,
    checkpoint_path: str | Path,
    device: torch.device,
    weights: str,
) -> nn.Module:
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if tuple(checkpoint.get("layer_names", ())) != tuple(LAYER_NAMES):
        raise ValueError("The checkpoint does not use the current layered channel schema")
    _check_checkpoint_config(config, checkpoint)
    model = build_model(config).to(device)
    if weights == "raw":
        state = checkpoint["model"]
    elif weights == "ema":
        ema_state = checkpoint.get("ema")
        if not ema_state:
            raise ValueError("Checkpoint does not contain EMA weights")
        state = ema_state.get("shadow", ema_state)
    else:
        raise ValueError("weights must be 'ema' or 'raw'")
    model.load_state_dict(state)
    model.eval()
    return model


def sample_layered_checkpoint(
    config: LayeredDiffusionConfig,
    checkpoint_path: str | Path,
    destination: str | Path,
    *,
    count: int = 8,
    seed: int | None = None,
    mixture: str | None = None,
    weights: str = "ema",
    device_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Sample output is not empty: {destination}")
        import shutil

        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    device = _select_device(device_name or config.device)
    model = _load_sample_model(config, checkpoint_path, device, weights)
    sample_seed = int(config.seed if seed is None else seed)
    mix = parse_city_mix(config.city_names, mixture)
    city = city_mix_tensor(
        config.city_names,
        mix,
        batch_size=count,
        device=device,
    )
    generated = sample_model(
        model,
        config,
        city=city,
        batch_size=count,
        device=device,
        seed=sample_seed,
    )
    save_sheet(generated, destination / "preview.png")

    results: list[dict[str, Any]] = []
    bounds = [0.0, 0.0, 1024.0, 1024.0]
    mix_values = np.asarray([mix[name] for name in config.city_names], dtype=np.float32)
    for index in range(count):
        sample_dir = destination / f"sample-{index + 1:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        values = generated[index].detach().cpu()
        decoded = model_space_to_layers(
            values,
            auxiliary_threshold=config.auxiliary_threshold,
            max_surface_offset_m=config.max_surface_offset_m,
            max_underground_depth_m=config.max_underground_depth_m,
            max_elevated_height_m=config.max_elevated_height_m,
        )
        np.savez_compressed(
            sample_dir / "layers.npz",
            model_values=values.numpy().astype(np.float32),
            surface=decoded["surface"].numpy().astype(np.uint8),
            road_underground=decoded["road_underground"].numpy().astype(np.uint8),
            road_elevated=decoded["road_elevated"].numpy().astype(np.uint8),
            rail_underground=decoded["rail_underground"].numpy().astype(np.uint8),
            rail_elevated=decoded["rail_elevated"].numpy().astype(np.uint8),
            building_height=decoded["building_height"].numpy().astype(np.float32),
            road_surface_offset_m=decoded["road_surface_offset_m"].numpy().astype(np.float32),
            road_underground_depth_m=decoded["road_underground_depth_m"].numpy().astype(np.float32),
            road_elevated_height_m=decoded["road_elevated_height_m"].numpy().astype(np.float32),
            rail_surface_offset_m=decoded["rail_surface_offset_m"].numpy().astype(np.float32),
            rail_underground_depth_m=decoded["rail_underground_depth_m"].numpy().astype(np.float32),
            rail_elevated_height_m=decoded["rail_elevated_height_m"].numpy().astype(np.float32),
            city_mix_names=np.asarray(config.city_names),
            city_mix_values=mix_values,
            layer_names=np.asarray(LAYER_NAMES),
        )
        render_triptych(values).save(sample_dir / "preview.png", optimize=True)
        city_json = generated_layers_to_city_state(
            values,
            bounds_m=bounds,
            max_height_m=config.max_height_m,
            minimum_component_pixels=config.minimum_vector_component_pixels,
            seed=sample_seed + index,
        )
        city_json["city_mix"] = mix
        _write_json(sample_dir / "city.json", city_json)
        obj = export_city_state_obj(sample_dir / "city.json", sample_dir / "city.obj")
        results.append(
            {
                "index": index,
                "seed": sample_seed + index,
                "preview": str(sample_dir / "preview.png"),
                "layers": str(sample_dir / "layers.npz"),
                "city": str(sample_dir / "city.json"),
                "obj": obj["obj"],
                "statistics": city_json["statistics"],
            }
        )

    summary = {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "output": str(destination),
        "device": str(device),
        "seed": sample_seed,
        "weights": weights,
        "city_mix": mix,
        "samples": results,
    }
    _write_json(destination / "summary.json", summary)
    return summary


__all__ = [
    "check_conditioned_data",
    "sample_layered_checkpoint",
    "train_layered_diffusion",
]
