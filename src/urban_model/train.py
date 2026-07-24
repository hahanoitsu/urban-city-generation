from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from urban_dataset.obj_export import export_city_state_obj

from .config import LayeredDiffusionConfig
from .data import (
    LAYER_NAMES,
    LayeredBlockDataset,
    check_layered_data,
    model_space_to_layers,
)
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


def _to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=device.type == "cuda")


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
            valid_mask = _to_device(batch["valid_mask"], device)
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
            with autocast_context(config, device):
                prediction = model(noisy, timesteps).sample
                loss = weighted_diffusion_loss(
                    prediction,
                    noise,
                    valid_mask,
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

    train_dataset = LayeredBlockDataset(
        config,
        config.train_manifest,
        augment=config.augment,
    )
    validation_dataset = LayeredBlockDataset(
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
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler.is_enabled() and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_validation_loss", best))

    _write_json(output / "config.json", asdict(config))
    _write_json(output / "environment.json", _environment(device))
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
            valid_mask = _to_device(batch["valid_mask"], device)
            timesteps = torch.randint(
                0,
                config.diffusion_steps,
                (x0.shape[0],),
                device=device,
            )
            noise = torch.randn_like(x0)
            noisy = noise_scheduler.add_noise(x0, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                prediction = model(noisy, timesteps).sample
                loss = weighted_diffusion_loss(
                    prediction,
                    noise,
                    valid_mask,
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
                batch_size=4,
                device=device,
                seed=config.seed + epoch,
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
        "epochs_completed": epochs_completed,
        "best_validation_loss": best,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "layers": list(LAYER_NAMES),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    _write_json(output / "summary.json", summary)
    return summary


def _load_sample_model(
    config: LayeredDiffusionConfig,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if tuple(checkpoint.get("layer_names", ())) != tuple(LAYER_NAMES):
        raise ValueError("The checkpoint does not use the current layered channel schema")
    model = build_model(config).to(device)
    ema_state = checkpoint.get("ema")
    state = ema_state.get("shadow", ema_state) if ema_state else checkpoint["model"]
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
    model = _load_sample_model(config, checkpoint_path, device)
    sample_seed = int(config.seed if seed is None else seed)
    generated = sample_model(
        model,
        config,
        batch_size=count,
        device=device,
        seed=sample_seed,
    )
    save_sheet(generated, destination / "preview.png")

    results: list[dict[str, Any]] = []
    bounds = [0.0, 0.0, 1024.0, 1024.0]
    for index in range(count):
        sample_dir = destination / f"sample-{index + 1:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        values = generated[index].detach().cpu()
        decoded = model_space_to_layers(values)
        np.savez_compressed(
            sample_dir / "layers.npz",
            model_values=values.numpy().astype(np.float32),
            surface=decoded["surface"].numpy().astype(np.uint8),
            road_underground=decoded["road_underground"].numpy().astype(np.uint8),
            road_elevated=decoded["road_elevated"].numpy().astype(np.uint8),
            rail_underground=decoded["rail_underground"].numpy().astype(np.uint8),
            rail_elevated=decoded["rail_elevated"].numpy().astype(np.uint8),
            building_height=decoded["building_height"].numpy().astype(np.float32),
            layer_names=np.asarray(LAYER_NAMES),
        )
        render_triptych(values).save(sample_dir / "preview.png", optimize=True)
        city = generated_layers_to_city_state(
            values,
            bounds_m=bounds,
            max_height_m=config.max_height_m,
            minimum_component_pixels=config.minimum_vector_component_pixels,
            seed=sample_seed + index,
        )
        _write_json(sample_dir / "city.json", city)
        obj = export_city_state_obj(sample_dir / "city.json", sample_dir / "city.obj")
        results.append(
            {
                "index": index,
                "seed": sample_seed + index,
                "preview": str(sample_dir / "preview.png"),
                "layers": str(sample_dir / "layers.npz"),
                "city": str(sample_dir / "city.json"),
                "obj": obj["obj"],
                "statistics": city["statistics"],
            }
        )

    summary = {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "output": str(destination),
        "device": str(device),
        "seed": sample_seed,
        "samples": results,
    }
    _write_json(destination / "summary.json", summary)
    return summary


__all__ = [
    "check_layered_data",
    "sample_layered_checkpoint",
    "train_layered_diffusion",
]
