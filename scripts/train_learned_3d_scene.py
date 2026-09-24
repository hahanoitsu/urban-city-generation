from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from urban_ai.object3d import (
    CityObject3DDataset,
    Object3DConfig,
    Object3DDenoiser,
    noisy_tokens,
    object_loss,
    sample_tokens,
    token_summary,
)
from urban_ai.object3d_scene import export_obj, save_topdown, write_scene
from urban_ai.schema import STYLE_FIELDS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_timesteps(batch: int, device: torch.device) -> torch.Tensor:
    bucket = torch.rand(batch, device=device)
    values = torch.empty(batch, device=device)

    high = bucket < 0.50
    mid = (bucket >= 0.50) & (bucket < 0.80)
    low = bucket >= 0.80

    values[high] = 0.75 + torch.rand(int(high.sum()), device=device) * 0.25
    values[mid] = 0.35 + torch.rand(int(mid.sum()), device=device) * 0.40
    values[low] = torch.rand(int(low.sum()), device=device) * 0.35
    return values


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def run_epoch(
    model,
    loader,
    device,
    *,
    optimizer=None,
    precision: str = "bf16",
    maximum_batches: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    batches = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            clean = batch["tokens"].to(device, non_blocking=True)
            style = batch["style"].to(device, non_blocking=True)
            timestep = sample_timesteps(clean.shape[0], device)
            noise = torch.randn_like(clean)
            noisy = noisy_tokens(clean, timestep, noise)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                prediction = model(noisy, timestep, style)
                loss = object_loss(prediction, clean)

            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += float(loss.detach())
            batches += 1
            if maximum_batches is not None and batches >= maximum_batches:
                break

    return total / max(batches, 1)


def save_samples(
    model,
    train_dataset,
    output: Path,
    device: torch.device,
    *,
    epoch: int,
    steps: int,
) -> list[dict]:
    model.eval()
    style = torch.zeros((1, len(STYLE_FIELDS)), dtype=torch.float32, device=device)
    rows = []

    for seed in (101, 202, 303):
        generated = sample_tokens(
            model,
            style,
            steps=steps,
            seed=seed,
            device=device,
        )[0]
        sample_dir = output / "samples" / f"epoch-{epoch:04d}"
        scene = write_scene(
            generated,
            sample_dir / f"seed-{seed}.json",
            seed=seed,
        )
        export_obj(scene, sample_dir / f"seed-{seed}.obj")
        save_topdown(scene, sample_dir / f"seed-{seed}.png")

        rows.append(
            {
                "epoch": epoch,
                "seed": seed,
                **token_summary(generated),
                **scene["summary"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--maximum-tokens", type=int, default=512)
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--preview-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5132)
    args = parser.parse_args()

    root = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    config = Object3DConfig(maximum_tokens=args.maximum_tokens)

    train_manifest = root / "data/manifests/corpus-v2/train.jsonl"
    validation_manifest = root / "data/manifests/corpus-v2/validation.jsonl"

    train_set = CityObject3DDataset(
        train_manifest,
        config,
        augment=True,
    )
    validation_set = CityObject3DDataset(
        validation_manifest,
        config,
        style_mean=train_set.style_mean,
        style_std=train_set.style_std,
        augment=False,
    )

    train_loader = make_loader(
        train_set,
        args.batch_size,
        args.workers,
        True,
    )
    validation_loader = make_loader(
        validation_set,
        args.batch_size,
        args.workers,
        False,
    )

    model = Object3DDenoiser(config).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    train_truncated = [max(0, state[2] - state[1]) for state in train_set.states]
    validation_truncated = [max(0, state[2] - state[1]) for state in validation_set.states]

    metadata = {
        "experiment": "learned-3d-scene-v1",
        "representation": "unordered 3d object tokens",
        "objects": ["road edge", "rail edge", "building solid"],
        "model": asdict(config),
        "style_fields": list(STYLE_FIELDS),
        "style_mean": train_set.style_mean.tolist(),
        "style_std": train_set.style_std.tolist(),
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "train_tiles_truncated": sum(value > 0 for value in train_truncated),
        "validation_tiles_truncated": sum(value > 0 for value in validation_truncated),
        "train_max_objects_dropped": max(train_truncated, default=0),
        "validation_max_objects_dropped": max(validation_truncated, default=0),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "axis_convention": "x-east, y-north, z-up",
        "notes": [
            "transport geometry is predicted by the model, not planned procedurally",
            "current metric z comes from city-state vertical data/defaults and is not claimed as measured tunnel depth",
            "building footprints are represented by learned oriented 3d boxes in this first probe",
        ],
    }
    (output / "experiment.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "validation_loss", "hours"],
        ).writeheader()

    started = time.time()
    best = math.inf
    last_epoch = 0
    sample_rows = []

    for epoch in range(1, args.max_epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            device,
            maximum_batches=16,
        )
        hours = (time.time() - started) / 3600.0
        last_epoch = epoch

        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(
                handle,
                fieldnames=["epoch", "train_loss", "validation_loss", "hours"],
            ).writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "hours": hours,
                }
            )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "config": asdict(config),
            "style_mean": train_set.style_mean.tolist(),
            "style_std": train_set.style_std.tolist(),
            "validation_loss": validation_loss,
        }
        torch.save(checkpoint, output / "latest.pt")
        if validation_loss < best:
            best = validation_loss
            torch.save(checkpoint, output / "best.pt")

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch} train={train_loss:.5f} "
                f"validation={validation_loss:.5f} time={hours:.2f}h",
                flush=True,
            )

        if epoch == 1 or epoch % args.preview_every == 0:
            sample_rows.extend(
                save_samples(
                    model,
                    train_set,
                    output,
                    device,
                    epoch=epoch,
                    steps=args.sample_steps,
                )
            )

        if hours >= args.max_hours:
            break

    sample_rows.extend(
        save_samples(
            model,
            train_set,
            output,
            device,
            epoch=last_epoch,
            steps=args.sample_steps,
        )
    )

    with (output / "sample-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if sample_rows:
            writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)

    summary = {
        "epochs": last_epoch,
        "hours": (time.time() - started) / 3600.0,
        "best_validation_loss": best,
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "checkpoint": str(output / "best.pt"),
        "samples": str(output / "samples"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
