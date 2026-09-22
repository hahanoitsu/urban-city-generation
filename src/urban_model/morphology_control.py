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

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .config import LayeredDiffusionConfig, load_layered_diffusion_config
from .data import SURFACE_CLASS_COUNT
from .model import autocast_context
from .surface_distribution_v2 import (
    _EMA,
    _block_types,
    _classes,
    _coordinate_grid,
    _direct_x0_loss,
    _sample_timesteps,
    _save_sheet,
    _schedulers,
    _surface,
    _surface_class_weights,
)

CONTROLS = (
    "water_coverage",
    "green_coverage",
    "building_coverage",
    "road_length_km_per_km2",
    "road_major_share",
)

POSITION_CHANNELS = 2


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_setup(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def make_optimizer(model: nn.Module, config: LayeredDiffusionConfig, device: torch.device):
    args = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if device.type == "cuda":
        try:
            return AdamW(model.parameters(), fused=True, **args)
        except (TypeError, RuntimeError):
            pass
    return AdamW(model.parameters(), **args)


def read_controls(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [name for name in ("tile_id", *CONTROLS) if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing descriptor columns: {missing}")
    frame = frame[["tile_id", *CONTROLS]].copy()
    for name in CONTROLS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame[list(CONTROLS)].isna().any().any():
        raise ValueError("Control descriptors contain missing values")
    return frame


def control_stats(frame: pd.DataFrame) -> dict:
    result = {}
    for name in CONTROLS:
        values = frame[name].to_numpy(dtype=np.float64)
        std = float(values.std())
        result[name] = {
            "mean": float(values.mean()),
            "std": std if std > 1e-8 else 1.0,
            "p10": float(np.quantile(values, 0.10)),
            "median": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def normalise(values: np.ndarray, stats: dict) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for index, name in enumerate(CONTROLS):
        output[..., index] = (
            values[..., index] - stats[name]["mean"]
        ) / stats[name]["std"]
    return np.clip(output, -4.0, 4.0)


class ControlDataset(Dataset):
    def __init__(self, base, descriptors: pd.DataFrame, stats: dict) -> None:
        self.base = base
        self.stats = stats
        self.values = {
            str(row.tile_id): np.asarray(
                [getattr(row, name) for name in CONTROLS],
                dtype=np.float32,
            )
            for row in descriptors.itertuples(index=False)
        }

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = dict(self.base[index])
        tile_id = str(item["tile_id"]).rsplit(":", 2)[0]
        if tile_id not in self.values:
            raise KeyError(f"No morphology descriptors for {tile_id}")
        item["controls"] = torch.from_numpy(
            normalise(self.values[tile_id][None], self.stats)[0]
        )
        return item


def build_model(config: LayeredDiffusionConfig) -> nn.Module:
    try:
        from diffusers import UNet2DModel
    except ImportError as exc:
        raise RuntimeError("Install the diffusion dependencies first") from exc

    down, up = _block_types(config)
    return UNet2DModel(
        sample_size=config.resolution,
        in_channels=SURFACE_CLASS_COUNT + POSITION_CHANNELS + len(CONTROLS),
        out_channels=SURFACE_CLASS_COUNT,
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=down,
        up_block_types=up,
        norm_num_groups=config.norm_num_groups,
        add_attention=True,
    )


def condition_planes(
    controls: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    return controls[:, :, None, None].expand(-1, -1, height, width)


def loader(dataset, config, shuffle: bool) -> DataLoader:
    workers = max(0, int(config.num_workers))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=config.pin_memory,
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def validate(
    model,
    data,
    config,
    scheduler,
    class_weights,
    device,
) -> tuple[float, float]:
    model.eval()
    xy = _coordinate_grid(config.resolution[0], device)
    generator = torch.Generator(device=device).manual_seed(config.seed + 400_000)
    normal_total = 0.0
    high_total = 0.0
    batches = 0

    for batch in data:
        x0, mask = _surface(batch, device)
        if device.type == "cuda":
            x0 = x0.contiguous(memory_format=torch.channels_last)
            mask = mask.contiguous(memory_format=torch.channels_last)
        controls = batch["controls"].to(device)
        count = x0.shape[0]
        noise = torch.randn(
            x0.shape,
            dtype=x0.dtype,
            device=device,
            generator=generator,
        )
        normal_t = torch.randint(
            0,
            config.diffusion_steps,
            (count,),
            device=device,
            generator=generator,
        )
        high_t = torch.randint(
            int(config.diffusion_steps * 0.9),
            config.diffusion_steps,
            (count,),
            device=device,
            generator=generator,
        )
        extra = torch.cat(
            [
                xy.expand(count, -1, -1, -1),
                condition_planes(controls, *config.resolution),
            ],
            dim=1,
        )

        for timesteps, bucket in ((normal_t, "normal"), (high_t, "high")):
            noisy = scheduler.add_noise(x0, noise, timesteps)
            with autocast_context(config, device):
                model_input = torch.cat([noisy, extra], dim=1)
                if device.type == "cuda":
                    model_input = model_input.contiguous(memory_format=torch.channels_last)
                prediction = model(model_input, timesteps).sample
                loss = _direct_x0_loss(
                    prediction,
                    x0,
                    mask,
                    class_weights,
                    config.channel_loss_weights,
                )
            if bucket == "normal":
                normal_total += float(loss)
            else:
                high_total += float(loss)

        batches += 1

    return normal_total / max(batches, 1), high_total / max(batches, 1)


@torch.inference_mode()
def sample(
    model,
    config,
    controls: np.ndarray,
    *,
    seed: int,
    steps: int,
    device: torch.device,
    same_noise: bool,
) -> torch.Tensor:
    _, scheduler = _schedulers(config)
    scheduler.set_timesteps(steps, device=device)

    control_tensor = torch.from_numpy(controls.astype(np.float32)).to(device)
    count = control_tensor.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if same_noise:
        base = torch.randn(
            (1, SURFACE_CLASS_COUNT, *config.resolution),
            generator=generator,
        )
        values = base.repeat(count, 1, 1, 1).to(device)
    else:
        values = torch.randn(
            (count, SURFACE_CLASS_COUNT, *config.resolution),
            generator=generator,
        ).to(device)
    if device.type == "cuda":
        values = values.contiguous(memory_format=torch.channels_last)

    xy = _coordinate_grid(config.resolution[0], device).expand(count, -1, -1, -1)
    cond = condition_planes(control_tensor, *config.resolution)
    extra = torch.cat([xy, cond], dim=1)

    model.eval()
    for timestep in scheduler.timesteps:
        with autocast_context(config, device):
            model_input = torch.cat([values, extra], dim=1)
            if device.type == "cuda":
                model_input = model_input.contiguous(memory_format=torch.channels_last)
            prediction = model(model_input, timestep).sample
        values = scheduler.step(
            prediction.float(),
            timestep,
            values,
            eta=0.0,
        ).prev_sample
    return values.clamp(-1.0, 1.0)


def skeleton_length(mask: np.ndarray, metres_per_pixel: float) -> float:
    skel = skeletonize(mask)
    total = 0.0
    root_two = 2.0**0.5
    height, width = skel.shape
    for dr, dc, scale in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, root_two), (1, -1, root_two)):
        r0 = max(0, -dr)
        r1 = min(height, height - dr)
        c0 = max(0, -dc)
        c1 = min(width, width - dc)
        first = skel[r0:r1, c0:c1]
        second = skel[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        total += float((first & second).sum()) * metres_per_pixel * scale
    return total


def measure(classes: np.ndarray) -> dict[str, float]:
    size = classes.shape[0]
    metres_per_pixel = 1000.0 / size
    roads = [(classes == index) for index in (3, 4, 5)]
    road_lengths = [skeleton_length(mask, metres_per_pixel) for mask in roads]
    total_road = sum(road_lengths)

    return {
        "water_coverage": float((classes == 7).mean()),
        "green_coverage": float((classes == 1).mean()),
        "building_coverage": float((classes == 2).mean()),
        "road_length_km_per_km2": total_road / 1000.0,
        "road_major_share": road_lengths[0] / total_road if total_road else 0.0,
    }


def save_sweep(
    model,
    config,
    stats: dict,
    output: Path,
    *,
    seeds: list[int],
    steps: int,
    device: torch.device,
) -> list[dict]:
    rows = []
    median = np.asarray([stats[name]["median"] for name in CONTROLS], dtype=np.float32)

    for control_index, control_name in enumerate(CONTROLS):
        images = []
        labels = []

        raw_conditions = []
        level_names = ("low", "mid", "high")
        for key in ("p10", "median", "p90"):
            values = median.copy()
            values[control_index] = stats[control_name][key]
            raw_conditions.append(values)
        raw_conditions = np.stack(raw_conditions)
        conditions = normalise(raw_conditions, stats)

        for seed in seeds:
            for level, raw, condition in zip(
                level_names,
                raw_conditions,
                conditions,
                strict=True,
            ):
                generated = sample(
                    model,
                    config,
                    condition[None],
                    seed=seed,
                    steps=steps,
                    device=device,
                    same_noise=True,
                )
                class_map = _classes(generated)[0]
                measured = measure(class_map)
                rows.append(
                    {
                        "control": control_name,
                        "seed": seed,
                        "level": level,
                        "target": float(raw[control_index]),
                        "measured": measured[control_name],
                        **{f"measured_{name}": value for name, value in measured.items()},
                    }
                )
                images.append(class_map)
                labels.append(
                    f"{seed} {level}: {measured[control_name]:.3f}"
                )
                del generated

        save_grid(
            np.stack(images),
            output / "control-sweeps" / f"{control_name}.png",
            labels,
            columns=3,
        )

    return rows


def save_grid(
    maps: np.ndarray,
    path: Path,
    labels: list[str],
    columns: int,
) -> None:
    from .surface_distribution_v2 import _class_image

    images = [_class_image(item) for item in maps]
    width, height = images[0].size
    header = 22
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * width, rows * (height + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        row, column = divmod(index, columns)
        x = column * width
        y = row * (height + header)
        draw.text((x + 4, y + 5), label, fill="black")
        canvas.paste(image, (x, y + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def sweep_summary(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    result = {}
    for name in CONTROLS:
        group = frame[frame.control == name]
        if len(group) < 2:
            continue
        correlation = float(group[["target", "measured"]].corr().iloc[0, 1])
        monotonic = 0
        seeds = sorted(group.seed.unique())
        for seed in seeds:
            values = group[group.seed == seed].set_index("level")["measured"]
            if values["low"] < values["mid"] < values["high"]:
                monotonic += 1
        result[name] = {
            "correlation": correlation,
            "monotonic_seeds": monotonic,
            "seeds": len(seeds),
            "low_mean": float(group[group.level == "low"].measured.mean()),
            "mid_mean": float(group[group.level == "mid"].measured.mean()),
            "high_mean": float(group[group.level == "high"].measured.mean()),
        }
    return result


def save_checkpoint(
    path: Path,
    model,
    ema,
    optimizer,
    config,
    stats,
    epoch,
    updates,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "updates": updates,
            "model": model.state_dict(),
            "ema": ema.state_dict(cpu=True),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "controls": list(CONTROLS),
            "control_stats": stats,
        },
        path,
    )


def train(
    config: LayeredDiffusionConfig,
    train_descriptors: Path,
    validation_descriptors: Path,
    output: Path,
    *,
    max_hours: float,
    max_epochs: int,
    device_name: str,
    preview_every: int,
    sweep_steps: int,
    overwrite: bool,
) -> dict:
    output = output.expanduser().resolve()
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    config = replace(
        config,
        output_dir=output,
        epochs=max_epochs,
        device=device_name,
        vertical_crop_repeat=1,
    )
    seed_everything(config.seed)
    device = torch.device(device_name)
    cuda_setup(device)

    train_frame = read_controls(train_descriptors)
    validation_frame = read_controls(validation_descriptors)
    stats = control_stats(train_frame)
    write_json(output / "control-stats.json", stats)

    from .data import LayeredBlockDataset

    base_train = LayeredBlockDataset(config, config.train_manifest, augment=config.augment)
    base_train_plain = LayeredBlockDataset(config, config.train_manifest, augment=False)
    base_validation = LayeredBlockDataset(config, config.validation_manifest, augment=False)

    train_set = ControlDataset(base_train, train_frame, stats)
    validation_set = ControlDataset(base_validation, validation_frame, stats)
    train_loader = loader(train_set, config, True)
    validation_loader = loader(validation_set, config, False)

    counts, weight_values = _surface_class_weights(base_train_plain, config)
    class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)

    model = build_model(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    optimizer = make_optimizer(model, config, device)
    ema = _EMA(model, config.ema_decay)
    noise_scheduler, _ = _schedulers(config)
    xy = _coordinate_grid(config.resolution[0], device)

    write_json(
        output / "experiment.json",
        {
            "name": "morphology-control-v1",
            "controls": list(CONTROLS),
            "question": "Does the city generator respond to requested urban morphology?",
            "prediction_type": "sample_x0",
            "high_noise_oversampling": True,
            "same_noise_control_sweeps": True,
            "config": asdict(config),
            "class_counts": counts.tolist(),
            "class_weights": weight_values.tolist(),
        },
    )

    metrics_path = output / "metrics.csv"
    fields = ["epoch", "updates", "train_loss", "validation_loss", "high_noise_loss", "hours"]
    with metrics_path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    started = time.time()
    updates = 0
    epoch = 0

    while epoch < max_epochs:
        epoch += 1
        model.train()
        total = 0.0
        batches = 0

        for batch in train_loader:
            x0, mask = _surface(batch, device)
            if device.type == "cuda":
                x0 = x0.contiguous(memory_format=torch.channels_last)
                mask = mask.contiguous(memory_format=torch.channels_last)
            controls = batch["controls"].to(device)
            count = x0.shape[0]
            timesteps = _sample_timesteps(count, config.diffusion_steps, device)
            noise = torch.randn_like(x0)
            noisy = noise_scheduler.add_noise(x0, noise, timesteps)
            extra = torch.cat(
                [
                    xy.expand(count, -1, -1, -1),
                    condition_planes(controls, *config.resolution),
                ],
                dim=1,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config, device):
                model_input = torch.cat([noisy, extra], dim=1)
                if device.type == "cuda":
                    model_input = model_input.contiguous(memory_format=torch.channels_last)
                prediction = model(model_input, timesteps).sample
                loss = _direct_x0_loss(
                    prediction,
                    x0,
                    mask,
                    class_weights,
                    config.channel_loss_weights,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            ema.update(model)

            total += float(loss)
            batches += 1
            updates += 1

        eval_model = build_model(config).to(device)
        if device.type == "cuda":
            eval_model = eval_model.to(memory_format=torch.channels_last)
        ema.load_into(eval_model)
        val_loss, high_loss = validate(
            eval_model,
            validation_loader,
            config,
            noise_scheduler,
            class_weights,
            device,
        )
        del eval_model

        hours = (time.time() - started) / 3600.0
        row = {
            "epoch": epoch,
            "updates": updates,
            "train_loss": total / max(batches, 1),
            "validation_loss": val_loss,
            "high_noise_loss": high_loss,
            "hours": hours,
        }
        with metrics_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} train={row['train_loss']:.5f} "
                f"val={val_loss:.5f} high={high_loss:.5f} "
                f"time={hours:.2f}h",
                flush=True,
            )

        if epoch == 1 or epoch % preview_every == 0:
            preview_model = build_model(config).to(device)
            if device.type == "cuda":
                preview_model = preview_model.to(memory_format=torch.channels_last)
            ema.load_into(preview_model)
            raw = train_frame[list(CONTROLS)].median().to_numpy(dtype=np.float32)
            controls = normalise(raw[None], stats)
            preview = sample(
                preview_model,
                config,
                controls,
                seed=config.seed + 200_000,
                steps=config.inference_steps,
                device=device,
                same_noise=False,
            )
            _save_sheet(
                _classes(preview),
                output / "previews" / f"epoch-{epoch:04d}.png",
            )
            del preview, preview_model

        if epoch == 1 or epoch % 25 == 0:
            save_checkpoint(
                output / "latest.pt",
                model,
                ema,
                optimizer,
                config,
                stats,
                epoch,
                updates,
            )

        if hours >= max_hours:
            break

    save_checkpoint(
        output / "latest.pt",
        model,
        ema,
        optimizer,
        config,
        stats,
        epoch,
        updates,
    )

    final_model = build_model(config).to(device)
    if device.type == "cuda":
        final_model = final_model.to(memory_format=torch.channels_last)
    ema.load_into(final_model)
    sweep_rows = save_sweep(
        final_model,
        config,
        stats,
        output,
        seeds=[101, 202, 303],
        steps=sweep_steps,
        device=device,
    )
    pd.DataFrame(sweep_rows).to_csv(output / "control-sweep.csv", index=False)
    summary = sweep_summary(sweep_rows)
    write_json(output / "control-summary.json", summary)

    result = {
        "output": str(output),
        "epochs": epoch,
        "updates": updates,
        "hours": round((time.time() - started) / 3600.0, 3),
        "controls": list(CONTROLS),
        "sweep_steps": sweep_steps,
        "control_results": summary,
    }
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--train-descriptors", required=True, type=Path)
    result.add_argument("--validation-descriptors", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--max-hours", type=float, default=4.75)
    result.add_argument("--max-epochs", type=int, default=3000)
    result.add_argument("--device", default="cuda")
    result.add_argument("--preview-every", type=int, default=100)
    result.add_argument("--sweep-steps", type=int, default=250)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    config = load_layered_diffusion_config(args.config)
    train(
        config,
        args.train_descriptors,
        args.validation_descriptors,
        args.output,
        max_hours=args.max_hours,
        max_epochs=args.max_epochs,
        device_name=args.device,
        preview_every=args.preview_every,
        sweep_steps=args.sweep_steps,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
