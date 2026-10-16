from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from urban_model.grid_city import GridCityConfig, GridCityDenoiser
from urban_model.grid_city_data import GridCityDataset
from urban_model.grid_city_diffusion import corrupt, grid_city_loss
from urban_model.structured_city_data import RELATIONS, SceneTensorConfig, StructuredCityDataset


def region_bucket(value):
    digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "little") % 100


def split_indices(dataset):
    groups = {"train": [], "validation": [], "test": []}
    for index, (row, _payload, _scene) in enumerate(dataset.samples):
        bucket = region_bucket(str(row["parent_region_id"]))
        if bucket < 75:
            groups["train"].append(index)
        elif bucket < 88:
            groups["validation"].append(index)
        else:
            groups["test"].append(index)
    return groups


def move(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_epoch(model, loader, device, optimizer=None, fixed_time=None, time_power=1.0):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    examples = 0
    parts = {}
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            batch = move(batch, device)
            size = batch["context"].shape[0]
            if fixed_time is not None:
                time_values = torch.full((size,), float(fixed_time), device=device)
            else:
                time_values = torch.rand(size, device=device).pow(time_power)
            noisy = corrupt(batch, time_values, model.config)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    noisy,
                    batch["context"],
                    batch["relations"],
                    batch["context_padding"],
                    batch["ports"],
                    batch["port_padding"],
                    time_values,
                )
                loss, current = grid_city_loss(output, batch, noisy, model.config)
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach()) * size
            examples += size
            for name, value in current.items():
                parts[name] = parts.get(name, 0.0) + value * size
    examples = max(examples, 1)
    return {"loss": total / examples, "parts": {name: value / examples for name, value in parts.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--time-power", type=float, default=1.0)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--nodes", type=int, default=448)
    parser.add_argument("--edges", type=int, default=512)
    parser.add_argument("--buildings", type=int, default=512)
    parser.add_argument("--areas", type=int, default=256)
    parser.add_argument("--ports", type=int, default=96)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--model-dimensions", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--position-bands", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(5132)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)

    scene_config = SceneTensorConfig(
        node_slots=args.nodes,
        edge_slots=args.edges,
        building_slots=args.buildings,
        area_slots=args.areas,
        maximum_ports=args.ports,
    )
    base = StructuredCityDataset(
        args.data,
        config=scene_config,
        maximum_samples=args.maximum_samples,
        cache_dir=args.cache_dir,
    )
    splits = split_indices(base)
    if args.overfit:
        everything = list(range(len(base)))
        splits = {"train": everything, "validation": everything, "test": []}

    model_config = GridCityConfig(
        context_dimensions=base.context_dimensions,
        relation_count=len(RELATIONS),
        port_dimensions=base.port_dimensions,
        model_dimensions=args.model_dimensions,
        layers=args.layers,
        dropout=args.dropout,
        position_bands=args.position_bands,
    )
    grid = model_config.grid()
    train_set = GridCityDataset(base, splits["train"], grid, augment=not args.overfit and not args.no_augment)
    validation_set = GridCityDataset(base, splits["validation"], grid)
    summary = {
        "samples": len(base),
        "splits": {name: len(values) for name, values in splits.items()},
        "train_kept": len(train_set),
        "validation_kept": len(validation_set),
        "rejected": dict(train_set.rejected + validation_set.rejected),
        "dropped": dict(train_set.dropped + validation_set.dropped),
    }
    print(json.dumps(summary, indent=2), flush=True)

    model = GridCityDenoiser(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    start = 0
    best = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint["epoch"])
        best = float(checkpoint.get("best", math.inf))
        print(f"resume={args.resume} epoch={start}", flush=True)

    options = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda", "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_set, shuffle=True, drop_last=len(train_set) > args.batch_size, **options)
    validation_loader = DataLoader(validation_set, shuffle=False, **options)

    args.output.mkdir(parents=True, exist_ok=True)
    experiment = {
        **summary,
        "model_config": model_config.to_dict(),
        "scene_config": scene_config.__dict__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "args": {key: str(value) for key, value in vars(args).items()},
    }
    (args.output / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")

    started = time.time()
    for epoch in range(start + 1, args.epochs + 1):
        epoch_started = time.time()
        train = run_epoch(model, train_loader, device, optimizer, time_power=args.time_power)
        middle = run_epoch(model, validation_loader, device, fixed_time=0.5)
        high = run_epoch(model, validation_loader, device, fixed_time=0.9)
        score = (middle["loss"] + high["loss"]) / 2.0
        record = {"epoch": epoch, "train": train, "validation": middle, "validation_high": high, "score": score}
        with (args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config.to_dict(),
            "scene_config": scene_config.__dict__,
            "overfit": args.overfit,
            "maximum_samples": args.maximum_samples,
            "best": min(best, score),
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(checkpoint, args.output / f"epoch-{epoch:03d}.pt")
        if score < best:
            best = score
            torch.save(checkpoint, args.output / "best.pt")
        values = middle["parts"]
        print(
            f"epoch={epoch}/{args.epochs} train={train['loss']:.4f} validation={middle['loss']:.4f} "
            f"high={high['loss']:.4f} node_presence={values['node_presence']:.4f} "
            f"edge_target={values['edge_target']:.4f} node_pos={values['node_pos']:.4f} "
            f"building_presence={values['building_presence']:.4f} building_frame={values['building_frame']:.4f} "
            f"building_shape={values['building_shape']:.4f} area_shape={values['area_shape']:.4f} "
            f"epoch_s={time.time() - epoch_started:.1f} elapsed_min={(time.time() - started) / 60.0:.1f}",
            flush=True,
        )

    (args.output / "summary.json").write_text(json.dumps({"epochs": args.epochs, "best": best, "seconds": time.time() - started}, indent=2) + "\n")


if __name__ == "__main__":
    main()
