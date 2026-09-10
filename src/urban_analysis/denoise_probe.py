from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from urban_dataset.obj_export import export_city_state_obj
from urban_model.config import LayeredDiffusionConfig, load_layered_diffusion_config
from urban_model.data import LAYER_NAMES, LayeredBlockDataset
from urban_model.model import (
    autocast_context,
    build_inference_scheduler,
    build_model,
    build_noise_scheduler,
)
from urban_model.preview import render_triptych
from urban_model.vectorize import generated_layers_to_city_state

from .generated_city_audit import audit_state


DEFAULT_TIMESTEPS = (25, 100, 250, 500, 750)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binary_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.bool()
    second = second.bool()
    union = torch.logical_or(first, second).sum().item()
    if union == 0:
        return 1.0
    intersection = torch.logical_and(first, second).sum().item()
    return float(intersection / union)


def _surface_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_class = reference[:8].argmax(dim=0)
    candidate_class = candidate[:8].argmax(dim=0)
    road_reference = (reference_class >= 3) & (reference_class <= 5)
    road_candidate = (candidate_class >= 3) & (candidate_class <= 5)
    return {
        "surface_accuracy": float((reference_class == candidate_class).float().mean().item()),
        "road_iou": _binary_iou(road_reference, road_candidate),
        "building_iou": _binary_iou(reference_class == 2, candidate_class == 2),
        "rail_iou": _binary_iou(reference_class == 6, candidate_class == 6),
    }


def _reconstruction_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    supervision: torch.Tensor,
) -> dict[str, float]:
    squared = (candidate - reference).float().square()
    mask = supervision.float()
    denominator = float(mask.sum().item())
    supervised = float((squared * mask).sum().item() / max(denominator, 1.0))
    return {
        "mse_all": float(squared.mean().item()),
        "mse_surface_channels": float(squared[:8].mean().item()),
        "mse_supervised": supervised,
        **_surface_metrics(reference, candidate),
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "metrics": {}}
    ignored = {"level", "tile_id", "tile_index", "kind"}
    numeric = sorted(
        key
        for key in rows[0]
        if key not in ignored
        and all(
            row.get(key) is None
            or isinstance(row.get(key), (int, float, np.integer, np.floating))
            for row in rows
        )
    )
    result: dict[str, Any] = {"count": len(rows), "metrics": {}}
    for key in numeric:
        values = np.asarray(
            [
                float(row[key])
                for row in rows
                if row.get(key) is not None and math.isfinite(float(row[key]))
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        result["metrics"][key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def _load_model(
    config: LayeredDiffusionConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("layer_names", ())) != tuple(LAYER_NAMES):
        raise ValueError("Checkpoint channel schema does not match the current 19-channel model")
    model = build_model(config).to(device)
    ema = checkpoint.get("ema")
    state = ema.get("shadow", ema) if ema else checkpoint["model"]
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def _noise_like(values: torch.Tensor, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(values.shape, generator=generator, dtype=torch.float32).to(device)


@torch.inference_mode()
def _reverse_from_timestep(
    model: torch.nn.Module,
    config: LayeredDiffusionConfig,
    sample: torch.Tensor,
    timestep: int,
    device: torch.device,
) -> torch.Tensor:
    training_scheduler = build_noise_scheduler(config)
    scheduler = build_inference_scheduler(config, training_scheduler)
    scheduler.set_timesteps(config.diffusion_steps, device=device)
    positions = torch.nonzero(scheduler.timesteps == int(timestep), as_tuple=False)
    if positions.numel() == 0:
        raise ValueError(f"Inference schedule does not contain timestep {timestep}")
    start = int(positions[0].item())
    steps = scheduler.timesteps[start:]
    for number, current in enumerate(steps, start=1):
        with autocast_context(config, device):
            prediction = model(sample, current).sample
        sample = scheduler.step(
            prediction.float(),
            current,
            sample,
            eta=0.0,
        ).prev_sample
        if number == 1 or number % 100 == 0 or number == len(steps):
            print(
                f"    reverse t={timestep}: {number}/{len(steps)} steps",
                flush=True,
            )
    return sample.clamp(-1.0, 1.0)


def _city_metrics(
    values: torch.Tensor,
    config: LayeredDiffusionConfig,
    temp_root: Path,
    *,
    seed: int,
    save_dir: Path | None = None,
) -> dict[str, Any]:
    city = generated_layers_to_city_state(
        values.detach().cpu(),
        bounds_m=(0.0, 0.0, 1024.0, 1024.0),
        max_height_m=config.max_height_m,
        max_surface_offset_m=config.max_surface_offset_m,
        max_underground_depth_m=config.max_underground_depth_m,
        max_elevated_height_m=config.max_elevated_height_m,
        road_max_grade=config.road_max_grade,
        rail_max_grade=config.rail_max_grade,
        auxiliary_threshold=config.auxiliary_threshold,
        minimum_component_pixels=config.minimum_vector_component_pixels,
        seed=seed,
    )

    if save_dir is not None:
        state_dir = save_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        city_path = state_dir / "city.json"
        _write_json(city_path, city)
        export_city_state_obj(city_path, state_dir / "city.obj")
        return audit_state(city_path, "probe")

    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        city_path = Path(directory) / "city.json"
        _write_json(city_path, city)
        return audit_state(city_path, "probe")


def _save_preview(values: torch.Tensor, path: Path) -> Image.Image:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = render_triptych(values.detach().cpu())
    image.save(path, optimize=True)
    return image


def _contact_sheet(
    images: dict[tuple[int, str], Image.Image],
    tile_ids: list[str],
    levels: list[str],
    path: Path,
) -> None:
    thumbnail_width = 300
    label_height = 34
    tiles: list[list[Image.Image]] = []
    for tile_index, tile_id in enumerate(tile_ids):
        row: list[Image.Image] = []
        for level in levels:
            source = images[(tile_index, level)].copy()
            ratio = thumbnail_width / source.width
            source = source.resize(
                (thumbnail_width, max(1, round(source.height * ratio))),
                Image.Resampling.LANCZOS,
            )
            cell = Image.new("RGB", (thumbnail_width, source.height + label_height), "white")
            cell.paste(source.convert("RGB"), (0, label_height))
            draw = ImageDraw.Draw(cell)
            draw.text((6, 4), f"{tile_id} | {level}", fill="black")
            row.append(cell)
        tiles.append(row)

    cell_height = max(image.height for row in tiles for image in row)
    sheet = Image.new(
        "RGB",
        (thumbnail_width * len(levels), cell_height * len(tiles)),
        "white",
    )
    for row_index, row in enumerate(tiles):
        for column_index, image in enumerate(row):
            sheet.paste(image, (column_index * thumbnail_width, row_index * cell_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_probe(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output: str | Path,
    *,
    tile_count: int = 4,
    timesteps: tuple[int, ...] = DEFAULT_TIMESTEPS,
    seed: int = 9142,
    device_name: str = "cuda",
    overwrite: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    config = load_layered_diffusion_config(config_file)

    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_root = output_path / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    if tile_count <= 0:
        raise ValueError("tile_count must be positive")
    if any(value < 0 or value >= config.diffusion_steps for value in timesteps):
        raise ValueError("Probe timesteps must be inside the training diffusion schedule")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = LayeredBlockDataset(config, config.validation_manifest, augment=False)
    if not dataset:
        raise ValueError("Validation dataset is empty")
    count = min(int(tile_count), len(dataset))
    indexes = sorted(set(np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()))
    samples = [dataset[index] for index in indexes]
    clean = torch.stack([sample["x0"] for sample in samples]).to(device)
    supervision = torch.stack([sample["valid_mask"] for sample in samples]).cpu()
    tile_ids = [str(sample["tile_id"]).split(":", 1)[0] for sample in samples]

    model, checkpoint = _load_model(config, checkpoint_file, device)
    training_scheduler = build_noise_scheduler(config)
    noise = _noise_like(clean, seed, device)

    levels: list[tuple[str, int | None, torch.Tensor]] = [("clean", None, clean.detach())]
    for timestep in timesteps:
        t = torch.full(
            (clean.shape[0],),
            int(timestep),
            device=device,
            dtype=torch.long,
        )
        noised = training_scheduler.add_noise(clean, noise, t)
        print(f"=== DENOISE t={timestep} ===", flush=True)
        reconstructed = _reverse_from_timestep(
            model,
            config,
            noised,
            int(timestep),
            device,
        )
        levels.append((f"t{timestep}", int(timestep), reconstructed.detach()))
        del noised
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pure_timestep = config.diffusion_steps - 1
    print("=== PURE NOISE ===", flush=True)
    pure = _reverse_from_timestep(
        model,
        config,
        noise.clone(),
        pure_timestep,
        device,
    )
    levels.append(("pure_noise", pure_timestep, pure.detach()))

    rows: list[dict[str, Any]] = []
    preview_images: dict[tuple[int, str], Image.Image] = {}
    reference_cpu = clean.detach().cpu()

    alpha = training_scheduler.alphas_cumprod.detach().cpu()
    representative_levels = {"clean", "t100", "t500", "pure_noise"}

    for level_name, timestep, values in levels:
        values_cpu = values.detach().cpu()
        if timestep is None:
            signal_scale = 1.0
            noise_scale = 0.0
        elif level_name == "pure_noise":
            signal_scale = 0.0
            noise_scale = 1.0
        else:
            alpha_t = float(alpha[int(timestep)].item())
            signal_scale = math.sqrt(alpha_t)
            noise_scale = math.sqrt(max(0.0, 1.0 - alpha_t))

        for sample_index, tile_id in enumerate(tile_ids):
            preview_path = (
                output_path
                / "previews"
                / f"tile-{sample_index + 1:02d}"
                / f"{level_name}.png"
            )
            preview_images[(sample_index, level_name)] = _save_preview(
                values_cpu[sample_index],
                preview_path,
            )

            save_dir = None
            if sample_index == 0 and level_name in representative_levels:
                save_dir = output_path / "representative-3d" / level_name

            city = _city_metrics(
                values_cpu[sample_index],
                config,
                temp_root,
                seed=seed + sample_index,
                save_dir=save_dir,
            )
            reconstruction = _reconstruction_metrics(
                reference_cpu[sample_index],
                values_cpu[sample_index],
                supervision[sample_index],
            )

            rows.append(
                {
                    "level": level_name,
                    "kind": "reference" if level_name == "clean" else "reconstruction",
                    "timestep": -1 if timestep is None else int(timestep),
                    "signal_scale": signal_scale,
                    "noise_scale": noise_scale,
                    "tile_index": indexes[sample_index],
                    "tile_id": tile_id,
                    **reconstruction,
                    "road_components": city["road_components"],
                    "road_total_length_m": city["road_total_length_m"],
                    "road_assisted_components": city["road_assisted_components"],
                    "road_assisted_largest_length_fraction": city[
                        "road_assisted_largest_length_fraction"
                    ],
                    "road_assisted_interior_component_length_fraction": city[
                        "road_assisted_interior_component_length_fraction"
                    ],
                    "road_interior_dead_ends": city["road_interior_dead_ends"],
                    "local_length_connected_to_higher_fraction": city[
                        "local_length_connected_to_higher_fraction"
                    ],
                    "road_component_length_serving_buildings_fraction": city[
                        "road_component_length_serving_buildings_fraction"
                    ],
                    "building_count": city["building_count"],
                    "building_area_median_m2": city["building_area_median_m2"],
                    "building_height_median_m": city["building_height_median_m"],
                    "buildings_within_20m_road_fraction": city[
                        "buildings_within_20m_road_fraction"
                    ],
                }
            )

    shutil.rmtree(temp_root, ignore_errors=True)
    _write_csv(output_path / "metrics.csv", rows)

    level_names = [name for name, _timestep, _values in levels]
    _contact_sheet(
        preview_images,
        tile_ids,
        level_names,
        output_path / "overview.png",
    )

    grouped = {
        level: _metric_summary([row for row in rows if row["level"] == level])
        for level in level_names
    }
    result = {
        "analysis_version": 1,
        "purpose": (
            "Test whether the corrected 1 km diffusion model is a local denoiser with "
            "a weak global prior, or whether even mild corruption is reconstructed poorly."
        ),
        "config": str(config_file),
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": _sha256(checkpoint_file),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_validation_loss": float(
            checkpoint.get("best_validation_loss", float("nan"))
        ),
        "validation_manifest": str(config.validation_manifest),
        "tile_indexes": indexes,
        "tile_ids": tile_ids,
        "seed": seed,
        "timesteps": list(timesteps),
        "reverse_steps": "exact 1000-step DDIM schedule, started at each corruption timestep",
        "levels": grouped,
        "notes": {
            "clean_reference": (
                "The clean model-space tile is vectorised with the same generated-output "
                "vectorizer used for every reconstruction."
            ),
            "pure_noise": (
                "Pure Gaussian noise is denoised from the final diffusion timestep using "
                "the same model and reverse process."
            ),
            "signal_scale": "sqrt(alpha_cumprod[t]) for forward-noised levels.",
            "noise_scale": "sqrt(1-alpha_cumprod[t]) for forward-noised levels.",
        },
    }
    _write_json(output_path / "summary.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe denoising strength versus global-prior failure in the 1 km diffusion model"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile-count", type=int, default=4)
    parser.add_argument("--timestep", action="append", type=int)
    parser.add_argument("--seed", type=int, default=9142)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_probe(
            args.config,
            args.checkpoint,
            args.output,
            tile_count=args.tile_count,
            timesteps=tuple(args.timestep or DEFAULT_TIMESTEPS),
            seed=args.seed,
            device_name=args.device,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
