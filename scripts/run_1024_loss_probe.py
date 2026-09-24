from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, convolve, label
from skimage.morphology import skeletonize
from torch import nn

from urban_model.config import load_layered_diffusion_config
from urban_model.data import LayeredBlockDataset
from urban_model.model import autocast_context
from urban_model.morphology_control import (
    CONTROLS,
    ControlDataset,
    build_model,
    condition_planes,
    control_stats,
    cuda_setup,
    loader,
    make_optimizer,
    normalise,
    read_controls,
    sample,
    seed_everything,
    validate,
)
from urban_model.surface_distribution_v2 import (
    _EMA,
    _classes,
    _coordinate_grid,
    _direct_x0_loss,
    _sample_timesteps,
    _schedulers,
    _surface,
    _surface_class_weights,
)

PALETTE = np.asarray(
    [
        (226, 221, 209),
        (111, 174, 105),
        (137, 142, 148),
        (215, 58, 48),
        (239, 116, 66),
        (246, 180, 90),
        (85, 176, 194),
        (78, 151, 211),
    ],
    dtype=np.uint8,
)

ARMS = (
    ("base", 0.0, 0.0),
    ("mse", 0.0, 0.0),
    ("edge", 0.20, 0.0),
    ("edge_transport", 0.20, 0.05),
)


def edge_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
) -> torch.Tensor:
    valid_x = supervision[..., :, 1:] * supervision[..., :, :-1]
    valid_y = supervision[..., 1:, :] * supervision[..., :-1, :]

    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]

    loss_x = (pred_x.float() - target_x.float()).abs() * valid_x.float()
    loss_y = (pred_y.float() - target_y.float()).abs() * valid_y.float()
    total = loss_x.sum() + loss_y.sum()
    weight = valid_x.sum() + valid_y.sum()
    return total / weight.clamp_min(1.0)


def transport_margin_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    classes = target.argmax(dim=1)
    valid = supervision[:, 0] > 0
    active = valid & (classes >= 3) & (classes <= 6)
    if not active.any():
        return prediction.new_zeros(())

    scores = prediction[:, 3:7].float()
    transport_class = (classes - 3).clamp(0, 3)
    correct = scores.gather(1, transport_class[:, None]).squeeze(1)

    one_hot = F.one_hot(transport_class, num_classes=4).permute(0, 3, 1, 2).bool()
    wrong = scores.masked_fill(one_hot, -1e4).amax(dim=1)
    margin = F.relu(1.0 - (correct - wrong))

    weights = class_weights[classes] * active.float()
    return (margin * weights).sum() / weights.sum().clamp_min(1.0)


def restore_ema(ema: _EMA, state: dict, device: torch.device) -> None:
    ema.updates = int(state.get("updates", 0))
    ema.shadow = {
        name: value.to(device=device)
        for name, value in state["shadow"].items()
    }


def endpoint_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skeleton = skeletonize(mask)
    neighbours = convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    endpoints = skeleton & (neighbours == 2)
    return endpoints, skeleton


def component_stats(mask: np.ndarray) -> tuple[int, float]:
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return int(count), float(sizes.max() / max(sizes.sum(), 1))


def sample_metrics(classes: np.ndarray) -> dict[str, float | int]:
    road = (classes >= 3) & (classes <= 5)
    rail = classes == 6
    building = classes == 2

    road_endpoints, road_skeleton = endpoint_mask(road)
    rail_endpoints, rail_skeleton = endpoint_mask(rail)
    road_components, road_largest = component_stats(road_skeleton)
    rail_components, rail_largest = component_stats(rail_skeleton)
    building_components, building_largest = component_stats(building)

    interior = np.ones(classes.shape, dtype=bool)
    interior[:4] = False
    interior[-4:] = False
    interior[:, :4] = False
    interior[:, -4:] = False

    rail_interior = rail_endpoints & interior
    rail_near_road = binary_dilation(road, iterations=3)

    building_boundary = building & ~binary_erosion(
        building,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )

    return {
        "building_fraction": float(building.mean()),
        "building_components": building_components,
        "building_largest_component_fraction": building_largest,
        "building_boundary_per_area": float(
            building_boundary.sum() / max(building.sum(), 1)
        ),
        "road_fraction": float(road.mean()),
        "road_components": road_components,
        "road_largest_component_fraction": road_largest,
        "rail_fraction": float(rail.mean()),
        "rail_components": rail_components,
        "rail_largest_component_fraction": rail_largest,
        "rail_interior_endpoints": int(rail_interior.sum()),
        "rail_interior_endpoint_near_road_fraction": float(
            (rail_interior & rail_near_road).sum() / max(rail_interior.sum(), 1)
        ),
    }


def save_class_map(classes: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(PALETTE[classes.astype(np.int64)]).save(path, optimize=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_conditions(stats: dict) -> list[tuple[str, np.ndarray]]:
    median = np.asarray([stats[name]["median"] for name in CONTROLS], dtype=np.float32)
    cases = [("median", median.copy())]

    road_high = median.copy()
    road_high[CONTROLS.index("road_major_share")] = stats["road_major_share"]["p90"]
    cases.append(("road_major_high", road_high))

    building_low = median.copy()
    building_low[CONTROLS.index("building_coverage")] = stats["building_coverage"]["p10"]
    cases.append(("building_low", building_low))
    return cases


def evaluate_arm(
    name: str,
    model: nn.Module,
    config,
    stats: dict,
    output: Path,
    device: torch.device,
    steps: int,
) -> list[dict]:
    rows = []
    sample_dir = output / "samples" / name

    for case_name, raw in evaluation_conditions(stats):
        controls = normalise(raw[None], stats)
        for seed in (101, 202, 303):
            generated = sample(
                model,
                config,
                controls,
                seed=seed,
                steps=steps,
                device=device,
                same_noise=False,
            )
            classes = _classes(generated)[0]
            row = {
                "arm": name,
                "case": case_name,
                "seed": seed,
                **sample_metrics(classes),
            }
            rows.append(row)
            save_class_map(classes, sample_dir / f"{case_name}-{seed}.png")
            del generated

    return rows


def load_arm(
    checkpoint: dict,
    config,
    device: torch.device,
):
    model = build_model(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    optimizer = make_optimizer(model, config, device)
    optimizer.load_state_dict(checkpoint["optimizer"])

    ema = _EMA(model, config.ema_decay)
    restore_ema(ema, checkpoint["ema"], device)
    return model, optimizer, ema


def run_arm(
    name: str,
    edge_weight: float,
    transport_weight: float,
    checkpoint: dict,
    config,
    train_set,
    validation_loader,
    class_weights: torch.Tensor,
    stats: dict,
    output: Path,
    device: torch.device,
    updates: int,
    inference_steps: int,
) -> tuple[dict, list[dict]]:
    seed_everything(config.seed)
    model, optimizer, ema = load_arm(checkpoint, config, device)
    noise_scheduler, _ = _schedulers(config)
    xy = _coordinate_grid(config.resolution[0], device)

    if name != "base":
        train_loader = loader(train_set, config, True)
        model.train()
        completed = 0
        totals = {"mse": 0.0, "edge": 0.0, "transport": 0.0, "loss": 0.0}
        started = time.time()

        while completed < updates:
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
                    reconstruction = _direct_x0_loss(
                        prediction,
                        x0,
                        mask,
                        class_weights,
                        config.channel_loss_weights,
                    )
                    edge = (
                        edge_loss(prediction, x0, mask)
                        if edge_weight > 0
                        else prediction.new_zeros(())
                    )
                    transport = (
                        transport_margin_loss(
                            prediction,
                            x0,
                            mask,
                            class_weights,
                        )
                        if transport_weight > 0
                        else prediction.new_zeros(())
                    )
                    loss = (
                        reconstruction
                        + edge_weight * edge
                        + transport_weight * transport
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss in {name}")

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                ema.update(model)

                totals["mse"] += float(reconstruction.detach())
                totals["edge"] += float(edge.detach())
                totals["transport"] += float(transport.detach())
                totals["loss"] += float(loss.detach())
                completed += 1

                if completed % 250 == 0 or completed == updates:
                    elapsed = (time.time() - started) / 3600.0
                    print(
                        f"{name}: {completed}/{updates} "
                        f"mse={totals['mse'] / completed:.5f} "
                        f"edge={totals['edge'] / completed:.5f} "
                        f"transport={totals['transport'] / completed:.5f} "
                        f"time={elapsed:.2f}h",
                        flush=True,
                    )

                if completed >= updates:
                    break

    eval_model = build_model(config).to(device)
    if device.type == "cuda":
        eval_model = eval_model.to(memory_format=torch.channels_last)
    ema.load_into(eval_model)

    validation_loss, high_noise_loss = validate(
        eval_model,
        validation_loader,
        config,
        noise_scheduler,
        class_weights,
        device,
    )

    rows = evaluate_arm(
        name,
        eval_model,
        config,
        stats,
        output,
        device,
        inference_steps,
    )

    result = {
        "arm": name,
        "updates": 0 if name == "base" else updates,
        "edge_weight": edge_weight,
        "transport_weight": transport_weight,
        "validation_loss": validation_loss,
        "high_noise_loss": high_noise_loss,
    }
    if name != "base":
        result.update(
            {
                "train_mse": totals["mse"] / updates,
                "train_edge": totals["edge"] / updates,
                "train_transport": totals["transport"] / updates,
                "train_total": totals["loss"] / updates,
            }
        )

    del eval_model, model, optimizer, ema
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--inference-steps", type=int, default=125)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = load_layered_diffusion_config(
        source_root / "configs/layered-corpus-v2-1km.yaml"
    )
    config = replace(
        config,
        train_manifest=source_root / "data/manifests/corpus-v2-1024/train.jsonl",
        validation_manifest=source_root / "data/manifests/corpus-v2-1024/validation.jsonl",
        resolution=(1024, 1024),
        crop_size_pixels=1024,
        crop_stride_pixels=1024,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        precision="bf16",
        device="cuda",
        vertical_crop_repeat=1,
    )

    device = torch.device("cuda")
    cuda_setup(device)

    run_root = source_root / "runs/morphology-control-1024-v1"
    train_frame = read_controls(run_root / "train-analysis/tiles.csv")
    validation_frame = read_controls(run_root / "validation-analysis/tiles.csv")
    stats = control_stats(train_frame)

    base_train = LayeredBlockDataset(config, config.train_manifest, augment=config.augment)
    base_train_plain = LayeredBlockDataset(config, config.train_manifest, augment=False)
    base_validation = LayeredBlockDataset(config, config.validation_manifest, augment=False)
    train_set = ControlDataset(base_train, train_frame, stats)
    validation_set = ControlDataset(base_validation, validation_frame, stats)
    validation_loader = loader(validation_set, config, False)

    _counts, weight_values = _surface_class_weights(base_train_plain, config)
    class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)

    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )

    arm_results = []
    sample_rows = []
    for name, edge_weight, transport_weight in ARMS:
        print(f"\n=== {name} ===", flush=True)
        result, rows = run_arm(
            name,
            edge_weight,
            transport_weight,
            checkpoint,
            config,
            train_set,
            validation_loader,
            class_weights,
            stats,
            output,
            device,
            args.updates,
            args.inference_steps,
        )
        arm_results.append(result)
        sample_rows.extend(rows)

    write_csv(output / "arms.csv", arm_results)
    write_csv(output / "samples.csv", sample_rows)

    summary = {
        "base_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "updates_per_finetune_arm": args.updates,
        "inference_steps": args.inference_steps,
        "arms": arm_results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
