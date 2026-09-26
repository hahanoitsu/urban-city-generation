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
from torch.utils.data import DataLoader, Subset

from urban_model.structured_city import StructuredCityConfig, StructuredCityDenoiser
from urban_model.structured_city_data import RELATIONS, SceneTensorConfig, StructuredCityDataset
from urban_model.structured_city_diffusion import corrupt_scene, structured_city_loss


SCENE_FIELDS = (
    "node_position",
    "node_presence",
    "node_z_valid",
    "edge_presence",
    "edge_mode",
    "edge_class",
    "edge_vertical",
    "edge_from",
    "edge_to",
    "edge_width",
    "edge_width_valid",
    "edge_width_weight",
    "edge_shape",
    "edge_z_valid",
    "building_presence",
    "building_kind",
    "building_shape",
    "building_height",
    "building_height_valid",
    "building_height_weight",
    "building_base_z",
    "building_base_z_valid",
    "area_presence",
    "area_kind",
    "area_shape",
)


def region_bucket(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "little") % 100


def split_indices(dataset):
    groups = {"train": [], "validation": [], "test": []}
    for index, (row, _payload) in enumerate(dataset.samples):
        bucket = region_bucket(str(row["parent_region_id"]))
        if bucket < 75:
            groups["train"].append(index)
        elif bucket < 88:
            groups["validation"].append(index)
        else:
            groups["test"].append(index)
    return groups


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(model, loader, dataset, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    batches = 0
    parts = {}
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for batch in loader:
            batch = move_batch(batch, device)
            target = {name: batch[name] for name in SCENE_FIELDS}
            time_values = (
                torch.rand(batch["context"].shape[0], device=device)
                if training
                else torch.full((batch["context"].shape[0],), 0.5, device=device)
            )
            noisy = corrupt_scene(target, time_values)
            relations = batch["relations"]
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    noisy,
                    batch["context"],
                    relations,
                    batch["context_padding"],
                    batch["ports"],
                    batch["port_padding"],
                    time_values,
                )
                loss, current = structured_city_loss(output, target)
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
            for name, value in current.items():
                parts[name] = parts.get(name, 0.0) + value

    return {
        "loss": loss_sum / max(batches, 1),
        "parts": {
            name: value / max(batches, 1)
            for name, value in sorted(parts.items())
        },
        "batches": batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=384)
    parser.add_argument("--edges", type=int, default=640)
    parser.add_argument("--buildings", type=int, default=384)
    parser.add_argument("--areas", type=int, default=96)
    parser.add_argument("--ports", type=int, default=96)
    parser.add_argument("--maximum-samples", type=int)
    args = parser.parse_args()

    torch.manual_seed(5132)
    torch.cuda.manual_seed_all(5132)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    scene_config = SceneTensorConfig(
        node_slots=args.nodes,
        edge_slots=args.edges,
        building_slots=args.buildings,
        area_slots=args.areas,
        maximum_ports=args.ports,
    )
    dataset = StructuredCityDataset(
        args.data,
        config=scene_config,
        maximum_samples=args.maximum_samples,
    )
    splits = split_indices(dataset)
    if not splits["train"] or not splits["validation"]:
        raise RuntimeError("Structured split produced an empty train or validation set")

    device = torch.device("cuda")
    model_config = StructuredCityConfig(
        context_dimensions=dataset.context_dimensions,
        relation_count=len(RELATIONS),
        port_dimensions=dataset.port_dimensions,
        node_slots=scene_config.node_slots,
        edge_slots=scene_config.edge_slots,
        building_slots=scene_config.building_slots,
        area_slots=scene_config.area_slots,
        edge_shape_points=scene_config.edge_shape_points,
        building_points=scene_config.building_points,
        area_points=scene_config.area_points,
    )
    model = StructuredCityDenoiser(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        Subset(dataset, splits["train"]),
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        Subset(dataset, splits["validation"]),
        shuffle=False,
        **loader_options,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scene_config": scene_config.__dict__,
        "model_config": model_config.to_dict(),
        "samples": len(dataset),
        "splits": {name: len(values) for name, values in splits.items()},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "feature_names": dataset.feature_names,
        "feature_mean": dataset.feature_mean.tolist(),
        "feature_std": dataset.feature_std.tolist(),
        "relation_names": list(RELATIONS),
    }
    (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")

    best = math.inf
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, dataset, device, optimizer)
        validation = run_epoch(model, validation_loader, dataset, device)
        record = {
            "epoch": epoch,
            "train": train,
            "validation": validation,
            "seconds": time.time() - started,
        }
        with (args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_config": model_config.to_dict(),
            "scene_config": scene_config.__dict__,
            "best_validation_loss": min(best, validation["loss"]),
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if validation["loss"] < best:
            best = validation["loss"]
            torch.save(checkpoint, args.output / "best.pt")
        print(
            f"epoch={epoch} train={train['loss']:.4f} validation={validation['loss']:.4f}",
            flush=True,
        )

    summary = {
        "epochs": args.epochs,
        "best_validation_loss": best,
        "seconds": time.time() - started,
        "samples": len(dataset),
        "splits": {name: len(values) for name, values in splits.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
