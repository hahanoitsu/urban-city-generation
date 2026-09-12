from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import torch
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import binary_dilation
from shapely.ops import unary_union
from torch.optim import AdamW

from urban_dataset.prepared import load_city_gpkg
from urban_dataset.vertical import VerticalMode, classify_vertical_mode

OVERVIEW_NAMES = (
    "land",
    "green",
    "urban",
    "road_major",
    "road_minor",
    "rail",
    "water",
)
OVERVIEW_CHANNELS = len(OVERVIEW_NAMES)
PALETTE = np.asarray(
    [
        (226, 221, 209),
        (111, 174, 105),
        (137, 142, 148),
        (215, 58, 48),
        (246, 180, 90),
        (85, 176, 194),
        (78, 151, 211),
    ],
    dtype=np.uint8,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _surface(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.copy()
    modes = frame.apply(classify_vertical_mode, axis=1)
    return frame[modes == VerticalMode.SURFACE].copy()


def _burn(
    geometries,
    *,
    transform,
    resolution: int,
    all_touched: bool = True,
) -> np.ndarray:
    shapes = [
        (geometry, 1)
        for geometry in geometries
        if geometry is not None and not geometry.is_empty
    ]
    if not shapes:
        return np.zeros((resolution, resolution), dtype=bool)
    return rasterize(
        shapes,
        out_shape=(resolution, resolution),
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)


def _square_bounds(bounds: tuple[float, float, float, float], padding: float) -> list[float]:
    minx, miny, maxx, maxy = [float(value) for value in bounds]
    width = maxx - minx
    height = maxy - miny
    side = max(width, height) * (1.0 + 2.0 * float(padding))
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    return [cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0]


def build_overview_target(
    prepared_city: str | Path,
    boundary_geojson: str | Path,
    *,
    resolution: int = 512,
    padding: float = 0.03,
) -> tuple[np.ndarray, dict[str, Any]]:
    city, metadata = load_city_gpkg(prepared_city)
    crs = city.roads.crs
    boundary = gpd.read_file(Path(boundary_geojson).expanduser().resolve()).to_crs(crs)
    official = unary_union([geometry for geometry in boundary.geometry if not geometry.is_empty])
    bounds = _square_bounds(official.bounds, padding)
    transform = from_bounds(*bounds, resolution, resolution)

    inside = _burn([official], transform=transform, resolution=resolution, all_touched=False)

    green_frames = []
    if not city.green.empty:
        green_frames.append(city.green)
    if not city.landuse.empty and "landuse_class" in city.landuse.columns:
        green_frames.append(city.landuse[city.landuse["landuse_class"] == "green"])
    green_geometries = [
        geometry
        for frame in green_frames
        for geometry in frame.geometry
    ]
    green = _burn(green_geometries, transform=transform, resolution=resolution)

    urban_classes = {"residential", "commercial_mixed", "industrial", "civic"}
    if not city.landuse.empty and "landuse_class" in city.landuse.columns:
        urban_frame = city.landuse[city.landuse["landuse_class"].isin(urban_classes)]
        urban = _burn(urban_frame.geometry, transform=transform, resolution=resolution)
    else:
        urban = np.zeros_like(inside)
    buildings = _burn(city.buildings.geometry, transform=transform, resolution=resolution)
    if buildings.any():
        buildings = binary_dilation(buildings, iterations=1)
    urban |= buildings

    water = _burn(city.water.geometry, transform=transform, resolution=resolution)

    roads = _surface(city.roads)
    if "road_class" in roads.columns:
        major_frame = roads[roads["road_class"] == "major"]
        minor_frame = roads[roads["road_class"].isin(["secondary", "local"])]
    else:
        major_frame = roads.iloc[0:0]
        minor_frame = roads

    major = _burn(major_frame.geometry, transform=transform, resolution=resolution)
    minor = _burn(minor_frame.geometry, transform=transform, resolution=resolution)
    # At Singapore scale one pixel is already tens of metres. Keep minor
    # streets one pixel wide and only thicken major corridors slightly so the
    # proof image stays readable without turning into a road heatmap.
    major = binary_dilation(major, iterations=1)

    rail = _surface(city.rail)
    rail_mask = _burn(rail.geometry, transform=transform, resolution=resolution)

    classes = np.full((resolution, resolution), 6, dtype=np.int64)
    classes[inside] = 0
    classes[green & inside] = 1
    classes[urban & inside] = 2
    classes[water] = 6
    classes[minor & inside] = 4
    classes[major & inside] = 3
    classes[rail_mask & inside] = 5

    counts = np.bincount(classes.reshape(-1), minlength=OVERVIEW_CHANNELS)
    summary = {
        "prepared_city": str(Path(prepared_city).expanduser().resolve()),
        "boundary": str(Path(boundary_geojson).expanduser().resolve()),
        "metric_crs": str(crs),
        "bounds_m": bounds,
        "resolution": resolution,
        "metres_per_pixel": (bounds[2] - bounds[0]) / resolution,
        "source_metadata": metadata,
        "class_fraction": {
            name: float(count / classes.size)
            for name, count in zip(OVERVIEW_NAMES, counts.tolist(), strict=True)
        },
    }
    return classes, summary


def save_class_image(classes: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(PALETTE[classes.astype(np.int64)]).save(path, optimize=True)
    return path


def _comparison(target: np.ndarray, sample: np.ndarray, path: Path, label: str) -> None:
    target_image = Image.fromarray(PALETTE[target.astype(np.int64)])
    sample_image = Image.fromarray(PALETTE[sample.astype(np.int64)])
    width, height = target_image.size
    header = 28
    canvas = Image.new("RGB", (width * 2, height + header), "white")
    canvas.paste(target_image, (0, header))
    canvas.paste(sample_image, (width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), "TARGET: whole Singapore", fill="black")
    draw.text((width + 6, 7), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def _metrics(target: np.ndarray, sample: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {
        "accuracy": float((target == sample).mean()),
    }
    ious = []
    for index, name in enumerate(OVERVIEW_NAMES):
        expected = target == index
        predicted = sample == index
        union = np.logical_or(expected, predicted).sum()
        iou = (
            float(np.logical_and(expected, predicted).sum() / union)
            if union
            else 1.0
        )
        result[f"iou_{name}"] = iou
        if expected.any():
            ious.append(iou)
    result["mean_iou"] = float(np.mean(ious)) if ious else 0.0

    expected_road = np.isin(target, [3, 4])
    predicted_road = np.isin(sample, [3, 4])
    road_union = np.logical_or(expected_road, predicted_road).sum()
    result["road_iou"] = (
        float(np.logical_and(expected_road, predicted_road).sum() / road_union)
        if road_union
        else 1.0
    )

    expected_city = target == 2
    predicted_city = sample == 2
    city_union = np.logical_or(expected_city, predicted_city).sum()
    result["urban_iou"] = (
        float(np.logical_and(expected_city, predicted_city).sum() / city_union)
        if city_union
        else 1.0
    )
    return result


def _require_diffusers():
    try:
        from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
    except ImportError as exc:
        raise RuntimeError("Install the diffusion dependencies first") from exc
    return UNet2DModel, DDPMScheduler, DDIMScheduler


def _build_model(resolution: int):
    UNet2DModel, _DDPM, _DDIM = _require_diffusers()
    return UNet2DModel(
        sample_size=(resolution, resolution),
        in_channels=OVERVIEW_CHANNELS,
        out_channels=OVERVIEW_CHANNELS,
        layers_per_block=2,
        block_out_channels=(64, 96, 128, 192, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        norm_num_groups=8,
        add_attention=True,
    )


def _schedulers(train_steps: int):
    _UNet, DDPMScheduler, DDIMScheduler = _require_diffusers()
    noise = DDPMScheduler(
        num_train_timesteps=int(train_steps),
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )
    inference = DDIMScheduler.from_config(noise.config)
    return noise, inference


@torch.inference_mode()
def _sample(
    model,
    *,
    resolution: int,
    train_steps: int,
    inference_steps: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    _noise, scheduler = _schedulers(train_steps)
    scheduler.set_timesteps(inference_steps, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randn(
        (1, OVERVIEW_CHANNELS, resolution, resolution),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    model.eval()
    for timestep in scheduler.timesteps:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(values, timestep).sample
        values = scheduler.step(
            prediction.float(),
            timestep,
            values,
            eta=0.0,
        ).prev_sample
    return values.clamp(-1.0, 1.0)


def _ema_update(ema: dict[str, torch.Tensor], model, decay: float, step: int) -> None:
    warm = min(float(decay), (1.0 + step) / (10.0 + step))
    for name, value in model.state_dict().items():
        source = value.detach()
        if source.is_floating_point():
            ema[name].mul_(warm).add_(source, alpha=1.0 - warm)
        else:
            ema[name].copy_(source)


def _checkpoint(
    path: Path,
    *,
    model,
    ema,
    optimizer,
    step: int,
    best_score: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "ema": {name: value.detach().cpu() for name, value in ema.items()},
            "optimizer": optimizer.state_dict(),
            "best_score": float(best_score),
            "config": config,
            "overview_names": list(OVERVIEW_NAMES),
        },
        path,
    )


def train_overfit(
    prepared_city: str | Path,
    boundary_geojson: str | Path,
    output: str | Path,
    *,
    resolution: int = 512,
    steps: int = 20_000,
    diffusion_steps: int = 1000,
    inference_steps: int = 250,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    sample_every: int = 1000,
    checkpoint_every: int = 1000,
    seed: int = 5132,
    sample_seed: int = 424242,
    device_name: str = "cuda",
    overwrite: bool = False,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output).expanduser().resolve()
    if output.exists() and overwrite and resume is None:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    target_classes, target_summary = build_overview_target(
        prepared_city,
        boundary_geojson,
        resolution=resolution,
    )
    save_class_image(target_classes, output / "target.png")
    _write_json(output / "target.json", target_summary)

    one_hot = np.eye(OVERVIEW_CHANNELS, dtype=np.float32)[target_classes]
    target = torch.from_numpy(one_hot).permute(2, 0, 1).mul(2.0).sub(1.0)
    target = target.unsqueeze(0).to(device)

    model = _build_model(resolution).to(device)
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    noise_scheduler, _inference = _schedulers(diffusion_steps)
    ema = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    start_step = 1
    best_score = -math.inf

    config = {
        "prepared_city": str(Path(prepared_city).expanduser().resolve()),
        "boundary": str(Path(boundary_geojson).expanduser().resolve()),
        "resolution": resolution,
        "steps": steps,
        "diffusion_steps": diffusion_steps,
        "inference_steps": inference_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "sample_every": sample_every,
        "checkpoint_every": checkpoint_every,
        "seed": seed,
        "sample_seed": sample_seed,
        "device": device_name,
    }

    if resume is not None:
        state = torch.load(Path(resume).expanduser().resolve(), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for name, value in state["ema"].items():
            ema[name].copy_(value.to(device))
        start_step = int(state["step"]) + 1
        best_score = float(state.get("best_score", -math.inf))

    metric_fields = [
        "step",
        "train_loss",
        "accuracy",
        "mean_iou",
        "road_iou",
        "urban_iou",
        *[f"iou_{name}" for name in OVERVIEW_NAMES],
    ]
    metrics_path = output / "metrics.csv"
    if start_step == 1:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=metric_fields)
            writer.writeheader()

    model.train()
    for step in range(start_step, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + step)
        noise = torch.randn(target.shape, generator=generator, dtype=torch.float32).to(device)
        timestep = torch.randint(
            0,
            diffusion_steps,
            (1,),
            generator=generator,
            dtype=torch.long,
        ).to(device)
        noised = noise_scheduler.add_noise(target, noise, timestep)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(noised, timestep).sample
            loss = torch.nn.functional.mse_loss(prediction.float(), noise)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _ema_update(ema, model, 0.9995, step)

        if step == 1 or step % 100 == 0:
            print(f"step={step}/{steps} loss={float(loss):.6f}", flush=True)

        should_sample = step == 1 or step % sample_every == 0 or step == steps
        if should_sample:
            # Swap EMA weights into the existing model for sampling instead of
            # holding a second 512px UNet on the GPU.
            current_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            model.load_state_dict(ema)
            model.eval()

            generated = _sample(
                model,
                resolution=resolution,
                train_steps=diffusion_steps,
                inference_steps=inference_steps,
                seed=sample_seed,
                device=device,
            )
            classes = generated[0].argmax(dim=0).detach().cpu().numpy().astype(np.int64)
            values = _metrics(target_classes, classes)
            values["step"] = step
            values["train_loss"] = float(loss.detach())

            save_class_image(classes, output / "samples" / f"step-{step:06d}.png")
            _comparison(
                target_classes,
                classes,
                output / "comparisons" / f"step-{step:06d}.png",
                f"PURE NOISE → sample at step {step}",
            )

            with metrics_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=metric_fields)
                writer.writerow(values)

            score = values["accuracy"] + values["urban_iou"] + values["road_iou"]
            print(
                f"sample step={step} accuracy={values['accuracy']:.4f} "
                f"urban_iou={values['urban_iou']:.4f} "
                f"road_iou={values['road_iou']:.4f} mean_iou={values['mean_iou']:.4f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                save_class_image(classes, output / "best.png")
                _comparison(
                    target_classes,
                    classes,
                    output / "best-comparison.png",
                    f"BEST PURE-NOISE SAMPLE: step {step}",
                )
                _checkpoint(
                    output / "best.pt",
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    step=step,
                    best_score=best_score,
                    config=config,
                )
            model.load_state_dict(current_state)
            model.train()
            del generated, current_state
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if step % checkpoint_every == 0 or step == steps:
            _checkpoint(
                output / "latest.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                step=step,
                best_score=best_score,
                config=config,
            )

    summary = {
        "output": str(output),
        "steps_completed": steps,
        "best_score": best_score,
        "target": str(output / "target.png"),
        "best": str(output / "best.png"),
        "best_comparison": str(output / "best-comparison.png"),
        "latest_checkpoint": str(output / "latest.pt"),
        "best_checkpoint": str(output / "best.pt"),
    }
    _write_json(output / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deliberately overfit one Singapore-wide semantic map with diffusion"
    )
    parser.add_argument("--city", required=True, type=Path)
    parser.add_argument("--boundary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--inference-steps", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5132)
    parser.add_argument("--sample-seed", type=int, default=424242)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = train_overfit(
            args.city,
            args.boundary,
            args.output,
            resolution=args.resolution,
            steps=args.steps,
            diffusion_steps=args.diffusion_steps,
            inference_steps=args.inference_steps,
            learning_rate=args.learning_rate,
            sample_every=args.sample_every,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
            sample_seed=args.sample_seed,
            device_name=args.device,
            overwrite=args.overwrite,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
