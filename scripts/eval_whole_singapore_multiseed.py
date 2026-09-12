from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from urban_model.whole_city_overfit import (
    OVERVIEW_NAMES,
    PALETTE,
    _build_model,
    _metrics,
    _sample,
    build_overview_target,
    save_class_image,
)


def _contact_sheet(
    target: np.ndarray,
    samples: list[tuple[int, np.ndarray, dict[str, float]]],
    path: Path,
) -> None:
    tile = Image.fromarray(PALETTE[target.astype(np.int64)])
    w, h = tile.size
    header = 34
    cols = 3
    items = [("TARGET", target, None)] + [
        (f"seed {seed}", sample, metrics) for seed, sample, metrics in samples
    ]
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * (h + header)), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (label, classes, metrics) in enumerate(items):
        row, col = divmod(index, cols)
        x = col * w
        y = row * (h + header)
        image = Image.fromarray(PALETTE[classes.astype(np.int64)])
        canvas.paste(image, (x, y + header))
        if metrics is None:
            text = label
        else:
            text = (
                f"{label}  acc={metrics['accuracy']:.4f}  "
                f"mIoU={metrics['mean_iou']:.4f}"
            )
        draw.text((x + 6, y + 9), text, fill="black")

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def evaluate(
    checkpoint: Path,
    city: Path,
    boundary: Path,
    output: Path,
    seeds: list[int],
    *,
    inference_steps: int = 1000,
    device_name: str = "cuda",
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    resolution = int(config.get("resolution", 512))
    diffusion_steps = int(config.get("diffusion_steps", 1000))

    target, target_summary = build_overview_target(
        city,
        boundary,
        resolution=resolution,
    )
    save_class_image(target, output / "target.png")

    model = _build_model(resolution)
    weights = state.get("ema") or state["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    rows: list[dict[str, float | int]] = []
    samples: list[tuple[int, np.ndarray, dict[str, float]]] = []

    for seed in seeds:
        print(f"sampling unseen seed {seed} ...", flush=True)
        generated = _sample(
            model,
            resolution=resolution,
            train_steps=diffusion_steps,
            inference_steps=inference_steps,
            seed=seed,
            device=device,
        )
        classes = (
            generated[0]
            .argmax(dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        metrics = _metrics(target, classes)
        samples.append((seed, classes, metrics))
        row = {"seed": seed, **metrics}
        rows.append(row)
        save_class_image(classes, output / f"seed-{seed}.png")
        print(
            f"seed={seed} "
            f"accuracy={metrics['accuracy']:.6f} "
            f"mean_iou={metrics['mean_iou']:.6f} "
            f"road_iou={metrics['road_iou']:.6f} "
            f"urban_iou={metrics['urban_iou']:.6f}",
            flush=True,
        )
        del generated
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_names = [
        "accuracy",
        "mean_iou",
        "road_iou",
        "urban_iou",
        *[f"iou_{name}" for name in OVERVIEW_NAMES],
    ]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", *metric_names])
        writer.writeheader()
        writer.writerows(rows)

    pairwise = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            seed_a, a, _ = samples[i]
            seed_b, b, _ = samples[j]
            pairwise.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "pixel_agreement": float((a == b).mean()),
                }
            )

    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": int(state.get("step", -1)),
        "checkpoint_prediction_type": config.get("prediction_type"),
        "resolution": resolution,
        "diffusion_steps": diffusion_steps,
        "inference_steps": inference_steps,
        "seeds": seeds,
        "original_training_sample_seed": config.get("sample_seed"),
        "all_seeds_unseen": config.get("sample_seed") not in seeds,
        "target_summary": target_summary,
        "mean_metrics": {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in metric_names
        },
        "min_metrics": {
            name: float(np.min([float(row[name]) for row in rows]))
            for name in metric_names
        },
        "pairwise_sample_agreement": pairwise,
        "mean_pairwise_sample_agreement": (
            float(np.mean([p["pixel_agreement"] for p in pairwise]))
            if pairwise
            else 1.0
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _contact_sheet(target, samples, output / "contact-sheet.png")

    print("\nsummary:", flush=True)
    print(
        f"  min accuracy: {summary['min_metrics']['accuracy']:.6f}",
        flush=True,
    )
    print(
        f"  min mean IoU: {summary['min_metrics']['mean_iou']:.6f}",
        flush=True,
    )
    print(
        "  mean pairwise agreement: "
        f"{summary['mean_pairwise_sample_agreement']:.6f}",
        flush=True,
    )
    print(f"  contact sheet: {output / 'contact-sheet.png'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate whole-Singapore overfit checkpoint on unseen pure-noise seeds"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--city", required=True, type=Path)
    parser.add_argument("--boundary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 123, 2026, 9999, 8675309],
    )
    parser.add_argument("--inference-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    evaluate(
        args.checkpoint.expanduser().resolve(),
        args.city.expanduser().resolve(),
        args.boundary.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.seeds,
        inference_steps=args.inference_steps,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
