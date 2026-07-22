from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .data import ReconstructionDataset
from .losses import ReconstructionLoss
from .metrics import MetricAccumulator
from .model import ReconstructionAutoencoder
from .preview import save_reconstruction_preview
from .runtime import (
    append_jsonl,
    environment_info,
    jsonable,
    move_batch,
    seed_everything,
    select_device,
    worker_count,
    write_json,
)


def _loader(
    dataset, config: TrainingConfig, device: torch.device, *, shuffle: bool
) -> DataLoader:
    workers = worker_count(config.run.num_workers)
    return DataLoader(
        dataset,
        batch_size=config.run.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _run_validation(
    model,
    criterion,
    loader,
    device,
    *,
    amp: bool,
    max_steps: int | None,
    height_scale_m: float,
) -> tuple[dict[str, object], dict | None, dict | None]:
    model.eval()
    accumulator = MetricAccumulator(height_scale_m=height_scale_m)
    preview_batch = None
    preview_outputs = None
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            batch = move_batch(batch, device)
            with _autocast(device, amp):
                outputs = model(batch["input"])
                losses = criterion(outputs, batch)
            accumulator.update(outputs, batch, losses)
            if preview_batch is None:
                preview_batch = batch
                preview_outputs = {key: value.detach() for key, value in outputs.items()}
    return accumulator.compute(), preview_batch, preview_outputs


def train(
    config: TrainingConfig,
    *,
    resume: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
) -> dict[str, object]:
    run_config = config.run
    if epochs is not None:
        run_config = replace(run_config, epochs=epochs)
    if batch_size is not None:
        run_config = replace(run_config, batch_size=batch_size)
    if device_name is not None:
        run_config = replace(run_config, device=device_name)
    config = replace(config, run=run_config)

    seed_everything(config.run.seed)
    device = select_device(config.run.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    train_dataset = ReconstructionDataset(config.data.train_manifest, augment=config.data.augment)
    validation_dataset = ReconstructionDataset(config.data.validation_manifest, augment=False)
    if not train_dataset:
        raise ValueError("The training manifest is empty")
    if not validation_dataset:
        raise ValueError("The validation manifest is empty")

    train_loader = _loader(train_dataset, config, device, shuffle=True)
    validation_loader = _loader(validation_dataset, config, device, shuffle=False)
    model = ReconstructionAutoencoder(config.model).to(device)
    criterion = ReconstructionLoss(config.loss).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.run.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=config.run.amp and device.type == "cuda")

    output_dir = config.run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "source": str(config.source_path),
        "data": config.data,
        "model": config.model,
        "loss": config.loss,
        "optimizer": config.optimizer,
        "run": config.run,
    }
    write_json(output_dir / "config.json", jsonable(resolved_config))
    write_json(output_dir / "environment.json", environment_info(device))
    metrics_path = output_dir / "metrics.jsonl"

    start_epoch = 1
    best_validation_loss = float("inf")
    if resume is not None:
        checkpoint = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", best_validation_loss)
        )

    patience = 0
    started = time.time()
    epochs_completed = start_epoch - 1
    for epoch in range(start_epoch, config.run.epochs + 1):
        model.train()
        train_metrics = MetricAccumulator(height_scale_m=config.data.height_scale_m)
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config.run.epochs}", leave=False)
        for step, batch in enumerate(progress):
            if (
                config.run.max_train_steps_per_epoch is not None
                and step >= config.run.max_train_steps_per_epoch
            ):
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config.run.amp):
                outputs = model(batch["input"])
                losses = criterion(outputs, batch)
            scaler.scale(losses["total"]).backward()
            if config.run.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config.run.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_metrics.update(outputs, batch, losses)
            progress.set_postfix(loss=f"{float(losses['total'].detach()):.4f}")

        validation_metrics, preview_batch, preview_outputs = _run_validation(
            model,
            criterion,
            validation_loader,
            device,
            amp=config.run.amp,
            max_steps=config.run.max_validation_steps,
            height_scale_m=config.data.height_scale_m,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()

        validation_loss = float(validation_metrics["loss"]["total"])
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics.compute(),
            "validation": validation_metrics,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        append_jsonl(metrics_path, record)

        save_checkpoint(
            output_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_validation_loss=min(best_validation_loss, validation_loss),
            config=config.raw,
        )
        if epoch % config.run.checkpoint_every == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"epoch-{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_loss=min(best_validation_loss, validation_loss),
                config=config.raw,
            )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience = 0
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_loss=best_validation_loss,
                config=config.raw,
            )
        else:
            patience += 1

        if (
            preview_batch is not None
            and preview_outputs is not None
            and epoch % config.run.preview_every == 0
        ):
            save_reconstruction_preview(
                preview_batch["input"],
                preview_outputs,
                list(preview_batch["tile_id"]),
                output_dir / "previews" / f"epoch-{epoch:04d}.png",
            )

        print(
            f"epoch={epoch} train={record['train']['loss']['total']:.4f} "
            f"validation={validation_loss:.4f}"
        )
        epochs_completed = epoch
        if patience >= config.run.early_stopping_patience:
            break

    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "epochs_completed": epochs_completed,
        "best_validation_loss": best_validation_loss,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
