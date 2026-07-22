from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .runtime import environment_info, select_device
from .semantic_config import SEMANTIC_NAMES, SemanticDiffusionConfig
from .semantic_data import (
    dataset_for_semantic_task,
    layers_to_semantic,
    model_space_to_semantic,
    semantic_to_model_space,
)
from .semantic_model import (
    ModelEMA,
    _copy_ema_model,
    _initialise_outpainting_from_block,
    _loader,
    _save_block_preview,
    _save_checkpoint,
    _save_outpaint_preview,
    _seed_everything,
    build_noise_scheduler,
    build_semantic_model,
    diffusion_loss,
    diffusion_model_input,
    render_semantic,
    sample_semantic,
)


def check_semantic_data(
    config: SemanticDiffusionConfig,
    *,
    samples_per_split: int = 4,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task": config.task,
        "semantic_classes": list(SEMANTIC_NAMES),
        "resolution": list(config.resolution),
    }
    for split, manifest in (
        ("train", config.train_manifest),
        ("validation", config.validation_manifest),
    ):
        if not manifest.exists():
            raise FileNotFoundError(f"Missing {split} manifest: {manifest}")
        dataset = dataset_for_semantic_task(config, manifest, augment=False)
        shapes = []
        counts = torch.zeros(len(SEMANTIC_NAMES), dtype=torch.int64)
        for index in range(min(len(dataset), samples_per_split)):
            sample = dataset[index]
            shapes.append(list(sample["x0"].shape))
            classes = model_space_to_semantic(sample["x0"])
            counts += torch.bincount(classes.flatten(), minlength=len(SEMANTIC_NAMES))
        result[split] = {
            "manifest": str(manifest),
            "samples": len(dataset),
            "sample_shapes": shapes,
            "inspected_class_pixels": {
                name: int(value)
                for name, value in zip(SEMANTIC_NAMES, counts.tolist(), strict=True)
            },
        }
    return result


def _validation_loss(
    model: nn.Module,
    loader: DataLoader,
    config: SemanticDiffusionConfig,
    noise_scheduler,
    device: torch.device,
) -> tuple[float, dict[str, Any] | None]:
    model.eval()
    total = 0.0
    batches = 0
    preview_batch = None
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if config.max_validation_steps is not None and step >= config.max_validation_steps:
                break
            x0 = batch["x0"].to(device)
            timesteps = torch.randint(
                0,
                config.diffusion_steps,
                (x0.shape[0],),
                device=device,
            )
            noise = torch.randn_like(x0)
            noisy = noise_scheduler.add_noise(x0, noise, timesteps)
            known_mask = batch.get("known_mask")
            road_guide = batch.get("road_guide")
            if known_mask is not None:
                known_mask = known_mask.to(device)
                road_guide = road_guide.to(device)
            model_input = diffusion_model_input(
                noisy,
                config.task,
                known_mask=known_mask,
                road_guide=road_guide,
            )
            prediction = model(model_input, timesteps).sample
            loss = diffusion_loss(prediction, noise, known_mask=known_mask)
            total += float(loss.detach())
            batches += 1
            if preview_batch is None:
                preview_batch = batch
    return total / max(batches, 1), preview_batch


def train_semantic_diffusion(
    config: SemanticDiffusionConfig,
    *,
    device_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    if epochs is not None:
        config = replace(config, epochs=epochs)
    if batch_size is not None:
        config = replace(config, batch_size=batch_size)

    _seed_everything(config.seed)
    device = select_device(device_name or config.device)
    train_dataset = dataset_for_semantic_task(
        config,
        config.train_manifest,
        augment=config.augment,
    )
    validation_dataset = dataset_for_semantic_task(
        config,
        config.validation_manifest,
        augment=False,
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("Semantic diffusion requires non-empty datasets")

    train_loader = _loader(train_dataset, config, shuffle=True)
    validation_loader = _loader(validation_dataset, config, shuffle=False)
    model = build_semantic_model(config).to(device)
    initialisation = None
    if config.task == "outpaint" and config.initial_checkpoint is not None:
        if not config.initial_checkpoint.exists():
            raise FileNotFoundError(
                f"Missing block checkpoint for outpainting: {config.initial_checkpoint}"
            )
        initialisation = _initialise_outpainting_from_block(
            model,
            config.initial_checkpoint,
        )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_validation_loss", best))

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment_info(device), indent=2),
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.jsonl"
    started = time.time()
    patience = 0
    epochs_completed = start_epoch - 1

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        progress = tqdm(train_loader, desc=f"semantic {epoch}/{config.epochs}", leave=False)
        for step, batch in enumerate(progress):
            if (
                config.max_train_steps_per_epoch is not None
                and step >= config.max_train_steps_per_epoch
            ):
                break
            x0 = batch["x0"].to(device)
            timesteps = torch.randint(
                0,
                config.diffusion_steps,
                (x0.shape[0],),
                device=device,
            )
            noise = torch.randn_like(x0)
            noisy = noise_scheduler.add_noise(x0, noise, timesteps)
            known_mask = batch.get("known_mask")
            road_guide = batch.get("road_guide")
            if known_mask is not None:
                known_mask = known_mask.to(device)
                road_guide = road_guide.to(device)
            model_input = diffusion_model_input(
                noisy,
                config.task,
                known_mask=known_mask,
                road_guide=road_guide,
            )
            prediction = model(model_input, timesteps).sample
            loss = diffusion_loss(prediction, noise, known_mask=known_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            ema.update(model)
            total += float(loss.detach())
            batches += 1
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        train_loss = total / max(batches, 1)
        validation_model = _copy_ema_model(config, ema, device)
        validation_loss, preview_batch = _validation_loss(
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

        _save_checkpoint(
            output_dir / "latest.pt",
            model=model,
            ema=ema,
            optimizer=optimizer,
            epoch=epoch,
            best_validation_loss=min(best, validation_loss),
            config=config,
        )
        if epoch % config.checkpoint_every == 0:
            _save_checkpoint(
                output_dir / "checkpoints" / f"epoch-{epoch:04d}.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_loss=min(best, validation_loss),
                config=config,
            )
        if validation_loss < best:
            best = validation_loss
            patience = 0
            _save_checkpoint(
                output_dir / "best.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_loss=best,
                config=config,
            )
        else:
            patience += 1

        if epoch % config.preview_every == 0:
            preview_model = _copy_ema_model(config, ema, device)
            if config.task == "block":
                _save_block_preview(
                    output_dir / "previews" / f"epoch-{epoch:04d}.png",
                    preview_model,
                    config,
                    device,
                    config.seed + epoch,
                )
            elif preview_batch is not None:
                _save_outpaint_preview(
                    output_dir / "previews" / f"epoch-{epoch:04d}.png",
                    preview_model,
                    config,
                    preview_batch,
                    device,
                    config.seed + epoch,
                )
            del preview_model

        print(f"epoch={epoch} train={train_loss:.4f} validation={validation_loss:.4f}")
        epochs_completed = epoch
        if patience >= config.early_stopping_patience:
            break

    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "task": config.task,
        "epochs_completed": epochs_completed,
        "best_validation_loss": best,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "semantic_classes": list(SEMANTIC_NAMES),
        "initialisation": initialisation,
        "method": "CityGen-style one-hot semantic diffusion using Hugging Face Diffusers",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _load_ema_checkpoint(
    config: SemanticDiffusionConfig,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    model = build_semantic_model(config).to(device)
    model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])
    model.eval()
    return model


def sample_blocks_from_checkpoint(
    config: SemanticDiffusionConfig,
    checkpoint_path: str | Path,
    destination: str | Path,
    *,
    count: int = 4,
    device_name: str | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    if config.task != "block":
        raise ValueError("The selected config is not a block-generation config")
    if count <= 0:
        raise ValueError("count must be positive")
    device = select_device(device_name or config.device)
    model = _load_ema_checkpoint(config, checkpoint_path, device)
    generated = sample_semantic(
        model,
        config,
        batch_size=count,
        device=device,
        seed=config.seed if seed is None else seed,
    )
    classes = model_space_to_semantic(generated).cpu()
    output_dir = Path(destination).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "semantic-blocks.npy", classes.numpy().astype(np.uint8))
    height, width = classes.shape[-2:]
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, height * rows), "white")
    for index in range(count):
        canvas.paste(
            render_semantic(classes[index]),
            ((index % columns) * width, (index // columns) * height),
        )
    canvas.save(output_dir / "preview.png", optimize=True)
    metadata = {
        "task": "block",
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "sample_seed": config.seed if seed is None else seed,
        "count": count,
        "semantic_classes": list(SEMANTIC_NAMES),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "semantic": str(output_dir / "semantic-blocks.npy"),
        "preview": str(output_dir / "preview.png"),
        "metadata": str(output_dir / "metadata.json"),
    }


def _turns_to_east(direction: str) -> int:
    try:
        return {"east": 0, "west": 2, "north": 3, "south": 1}[direction]
    except KeyError as exc:
        raise ValueError("direction must be east, west, north or south") from exc


def sample_outpainting_from_checkpoint(
    config: SemanticDiffusionConfig,
    checkpoint_path: str | Path,
    seed_archive: str | Path,
    destination: str | Path,
    *,
    direction: str = "east",
    device_name: str | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    if config.task != "outpaint":
        raise ValueError("The selected config is not an outpainting config")
    archive_path = Path(seed_archive).expanduser().resolve()
    with np.load(archive_path, allow_pickle=False) as archive:
        layers = torch.from_numpy(archive["layers"].astype(np.float32))
        centerlines = torch.from_numpy(archive["road_centerlines"].astype(np.float32))
    if layers.shape[0] != 12 or centerlines.shape[0] != 3:
        raise ValueError("Seed archive does not contain the expected city channels")

    direction = direction.lower()
    turns = _turns_to_east(direction)
    if turns:
        layers = torch.rot90(layers, turns, dims=(-2, -1))
        centerlines = torch.rot90(centerlines, turns, dims=(-2, -1))
    height, width = layers.shape[-2:]
    source = torch.zeros((12, height, width * 2), dtype=layers.dtype)
    source[:, :, :width] = layers
    known_x0 = semantic_to_model_space(layers_to_semantic(source))
    known_x0 = F.interpolate(known_x0.unsqueeze(0), size=config.resolution, mode="nearest")
    known_mask = torch.zeros((1, 1, *config.resolution), dtype=torch.float32)
    known_mask[:, :, :, : config.resolution[1] // 2] = 1

    guide_source = torch.zeros((3, height, width * 2), dtype=centerlines.dtype)
    boundary_width = min(max(config.boundary_width, 1), width)
    guide_length = min(max(config.guide_length, 1), width)
    guide_source[:, :, width - boundary_width : width] = centerlines[
        :, :, width - boundary_width : width
    ]
    crossings = centerlines[:, :, width - boundary_width :].amax(dim=-1) > 0.5
    guide_source[:, :, width : width + guide_length] = crossings.unsqueeze(-1)
    road_guide = F.interpolate(
        guide_source.unsqueeze(0),
        size=config.resolution,
        mode="nearest",
    )

    device = select_device(device_name or config.device)
    model = _load_ema_checkpoint(config, checkpoint_path, device)
    generated = sample_semantic(
        model,
        config,
        batch_size=1,
        device=device,
        seed=config.seed if seed is None else seed,
        known_x0=known_x0,
        known_mask=known_mask,
        road_guide=road_guide,
    )
    semantic = model_space_to_semantic(generated)[0].cpu()
    inverse_turns = (-turns) % 4
    if inverse_turns:
        semantic = torch.rot90(semantic, inverse_turns, dims=(-2, -1))

    output_dir = Path(destination).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "semantic.npy", semantic.numpy().astype(np.uint8))
    render_semantic(semantic).save(output_dir / "preview.png", optimize=True)
    metadata = {
        "task": "outpaint",
        "method": "CityGen-style masked semantic diffusion",
        "seed_archive": str(archive_path),
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "direction": direction,
        "sample_seed": config.seed if seed is None else seed,
        "semantic_classes": list(SEMANTIC_NAMES),
        "seed_preserved_in_semantic_field": True,
        "height_generation": "separate later stage",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "semantic": str(output_dir / "semantic.npy"),
        "preview": str(output_dir / "preview.png"),
        "metadata": str(output_dir / "metadata.json"),
    }
