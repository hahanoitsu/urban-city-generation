from __future__ import annotations

import json
import math
import platform
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from .codec import FIELDS, OP_ADD, OP_CONNECT, OP_PAD, OP_ROOT, CommandCodecConfig
from .config import load_config, resolve_path
from .dataset import GraphProgramDataset, collate_graph_programs
from .loss import graph_program_loss
from .model import GraphProgramTransformer, GraphTransformerConfig
from .prepare import write_json
from .schema import STYLE_FIELDS


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


def _shift(batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs = {field: batch[field][:, :-1] for field in FIELDS}
    targets = {field: batch[field][:, 1:] for field in FIELDS}
    return inputs, targets


def _accuracy_counts(
    logits: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
) -> dict[str, tuple[int, int]]:
    op = targets["op"]
    root = op.eq(OP_ROOT)
    add = op.eq(OP_ADD)
    connect = op.eq(OP_CONNECT)
    masks = {
        "op": op.ne(OP_PAD),
        "x": root | add,
        "y": root | add,
        "id1": add | connect,
        "id2": connect,
        "mode": root | add | connect,
        "class": add | connect,
        "width": add | connect,
        "vertical": root | add | connect,
        "layer": root | add | connect,
    }
    result: dict[str, tuple[int, int]] = {}
    for field, mask in masks.items():
        if not bool(mask.any()):
            result[field] = (0, 0)
            continue
        predicted = logits[field].argmax(dim=-1)
        expected = targets[field] if field == "op" else targets[field] - 1
        result[field] = (
            int((predicted[mask] == expected[mask]).sum().item()),
            int(mask.sum().item()),
        )
    return result


def _merge_counts(total: dict[str, list[int]], update: dict[str, tuple[int, int]]) -> None:
    for field, (correct, count) in update.items():
        values = total.setdefault(field, [0, 0])
        values[0] += correct
        values[1] += count


def _run_epoch(
    model: GraphProgramTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    precision: str,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
    gradient_clip_norm: float,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    batch_count = 0
    field_loss_sums: dict[str, float] = {}
    accuracy: dict[str, list[int]] = {}

    context = nullcontext() if training else torch.inference_mode()
    with context:
        for batch in loader:
            batch = _move(batch, device)
            inputs, targets = _shift(batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                logits = model(inputs, batch["style"])
                loss, field_losses = graph_program_loss(logits, targets)
            if training:
                assert optimizer is not None
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if gradient_clip_norm > 0:
                        clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm > 0:
                        clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
            loss_sum += float(loss.detach())
            batch_count += 1
            for name, value in field_losses.items():
                field_loss_sums[name] = field_loss_sums.get(name, 0.0) + value
            _merge_counts(accuracy, _accuracy_counts(logits, targets))

    return {
        "loss": loss_sum / max(batch_count, 1),
        "field_loss": {
            name: value / max(batch_count, 1) for name, value in sorted(field_loss_sums.items())
        },
        "accuracy": {
            name: correct / count if count else 0.0
            for name, (correct, count) in sorted(accuracy.items())
        },
        "batches": batch_count,
    }


def _model_config(config: dict[str, Any], codec: CommandCodecConfig) -> GraphTransformerConfig:
    model = config.get("model", {})
    dimensions = int(model.get("model_dimensions", 192))
    heads = int(model.get("attention_heads", 6))
    if dimensions % heads:
        raise ValueError("model_dimensions must be divisible by attention_heads")
    return GraphTransformerConfig(
        codec=codec,
        style_dimensions=len(STYLE_FIELDS),
        maximum_sequence_length=int(model.get("maximum_sequence_length", 768)),
        model_dimensions=dimensions,
        attention_heads=heads,
        layers=int(model.get("layers", 4)),
        feedforward_dimensions=int(model.get("feedforward_dimensions", dimensions * 4)),
        dropout=float(model.get("dropout", 0.1)),
    )


def _subset(dataset, maximum: int | None):
    if maximum is None or maximum <= 0 or maximum >= len(dataset):
        return dataset
    return Subset(dataset, list(range(maximum)))


def train_from_config(
    config_file: str | Path,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
    resume: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_path, config = load_config(config_file)
    data_config = config.get("data", {})
    training = config.get("training", {})
    program_root = resolve_path(config_path, data_config.get("program_root", "data/programs"))
    output_dir = resolve_path(config_path, training.get("output_dir", "runs/generator"))
    epochs = int(epochs or training.get("epochs", 40))
    batch_size = int(batch_size or training.get("batch_size", 8))
    seed = int(training.get("seed", 5132))
    precision = str(training.get("precision", "bf16")).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16 or bf16")
    device = _device(device_name or str(training.get("device", "auto")))
    _seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        if not overwrite:
            raise FileExistsError(
                f"Training output is not empty: {output_dir}. Use --overwrite or --resume."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codec = CommandCodecConfig.from_dict(
        json.loads((program_root / "codec.json").read_text(encoding="utf-8"))
    )
    style_stats = json.loads((program_root / "style-stats.json").read_text(encoding="utf-8"))
    model_config = _model_config(config, codec)
    train_dataset = GraphProgramDataset(
        program_root / "train.jsonl",
        codec,
        style_mean=style_stats["mean"],
        style_std=style_stats["std"],
        augment=bool(training.get("augment", True)),
    )
    validation_dataset = GraphProgramDataset(
        program_root / "validation.jsonl",
        codec,
        style_mean=style_stats["mean"],
        style_std=style_stats["std"],
        augment=False,
    )
    train_dataset = _subset(train_dataset, training.get("maximum_train_samples"))
    validation_dataset = _subset(
        validation_dataset, training.get("maximum_validation_samples")
    )
    workers = int(training.get("workers", 4))
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "collate_fn": collate_graph_programs,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = GraphProgramTransformer(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    start_epoch = 1
    best_validation = math.inf
    if resume is not None:
        checkpoint = torch.load(Path(resume).expanduser(), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint.get("best_validation_loss", math.inf))

    resolved = {
        "config_file": str(config_path),
        "model": model_config.to_dict(),
        "training": training,
        "program_root": str(program_root),
        "style_stats": style_stats,
    }
    write_json(output_dir / "config.json", resolved)
    write_json(
        output_dir / "environment.json",
        {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "precision": precision,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
    )
    metrics_path = output_dir / "metrics.jsonl"
    started = time.time()
    completed = start_epoch - 1

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            precision=precision,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            device,
            precision=precision,
            optimizer=None,
            scaler=None,
            gradient_clip_norm=0.0,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": model_config.to_dict(),
            "style_stats": style_stats,
            "best_validation_loss": min(best_validation, validation_metrics["loss"]),
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if validation_metrics["loss"] < best_validation:
            best_validation = float(validation_metrics["loss"])
            checkpoint["best_validation_loss"] = best_validation
            torch.save(checkpoint, output_dir / "best.pt")
        completed = epoch
        print(
            f"epoch={epoch} train={train_metrics['loss']:.4f} "
            f"validation={validation_metrics['loss']:.4f}"
        )

    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "precision": precision,
        "epochs_completed": completed,
        "best_validation_loss": best_validation,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
