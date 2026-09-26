from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

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
    for index, (row, _payload, _scene) in enumerate(dataset.samples):
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


def distributed_state():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def reduce_epoch(loss_sum, examples, batches, parts, device):
    names = sorted(parts)
    values = torch.tensor(
        [loss_sum, float(examples), float(batches), *[parts[name] for name in names]],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_examples = max(float(values[1].item()), 1.0)
    return {
        "loss": float(values[0].item() / total_examples),
        "parts": {
            name: float(values[index + 3].item() / total_examples)
            for index, name in enumerate(names)
        },
        "examples": int(values[1].item()),
        "batches": int(values[2].item()),
    }


def run_epoch(model, loader, dataset, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    examples = 0
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
                clip_grad_norm_(
                    (
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    1.0,
                    foreach=True,
                )
                optimizer.step()
            batch_examples = int(batch["context"].shape[0])
            loss_sum += float(loss.detach()) * batch_examples
            examples += batch_examples
            batches += 1
            for name, value in current.items():
                parts[name] = parts.get(name, 0.0) + value * batch_examples

    return reduce_epoch(loss_sum, examples, batches, parts, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=448)
    parser.add_argument("--edges", type=int, default=512)
    parser.add_argument("--buildings", type=int, default=512)
    parser.add_argument("--areas", type=int, default=160)
    parser.add_argument("--ports", type=int, default=96)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    rank, local_rank, world_size = distributed_state()
    torch.manual_seed(5132 + rank)
    torch.cuda.manual_seed_all(5132 + rank)
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
        cache_dir=args.cache_dir,
        show_cache_progress=rank == 0,
    )
    splits = split_indices(dataset)
    if not splits["train"] or not splits["validation"]:
        raise RuntimeError("Structured split produced an empty train or validation set")

    if rank == 0:
        print(
            json.dumps(
                {
                    "samples": len(dataset),
                    "source_rows": dataset.total_rows,
                    "accepted_before_limit": dataset.accepted_before_limit,
                    "rejected": dataset.rejected,
                    "cache_hits": dataset.cache_hits,
                    "cache_built": dataset.cache_built,
                    "splits": {name: len(values) for name, values in splits.items()},
                    "world_size": world_size,
                },
                indent=2,
            ),
            flush=True,
        )

    device = torch.device("cuda", local_rank) if world_size > 1 else torch.device("cuda")
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
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=2e-4,
        weight_decay=0.01,
        fused=True,
    )

    start_epoch = 0
    best = math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best = float(checkpoint.get("best_validation_loss", math.inf))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if rank == 0:
            print(
                f"resume={args.resume} epoch={start_epoch} "
                f"optimizer={'loaded' if 'optimizer' in checkpoint else 'fresh'}",
                flush=True,
            )

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    train_subset = Subset(dataset, splits["train"])
    validation_subset = Subset(dataset, splits["validation"])
    train_sampler = (
        DistributedSampler(
            train_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=5132,
        )
        if world_size > 1
        else None
    )
    validation_sampler = (
        DistributedSampler(
            validation_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if world_size > 1
        else None
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        train_subset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_subset,
        shuffle=False,
        sampler=validation_sampler,
        **loader_options,
    )

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scene_config": scene_config.__dict__,
        "model_config": model_config.to_dict(),
        "samples": len(dataset),
        "source_rows": dataset.total_rows,
        "accepted_before_limit": dataset.accepted_before_limit,
        "rejected": dataset.rejected,
        "cache_hits": dataset.cache_hits,
        "cache_built": dataset.cache_built,
        "splits": {name: len(values) for name, values in splits.items()},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "feature_names": dataset.feature_names,
        "feature_mean": dataset.feature_mean.tolist(),
        "feature_std": dataset.feature_std.tolist(),
        "relation_names": list(RELATIONS),
        "world_size": world_size,
        "batch_size_per_gpu": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
    }
    if rank == 0:
        (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")

    started = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_started = time.time()
        train = run_epoch(model, train_loader, dataset, device, optimizer)
        validation = run_epoch(model, validation_loader, dataset, device)
        record = {
            "epoch": epoch,
            "train": train,
            "validation": validation,
            "seconds": time.time() - started,
        }
        values = validation["parts"]
        epoch_seconds = time.time() - epoch_started
        elapsed_seconds = time.time() - started
        examples = len(train_loader.dataset) + len(validation_loader.dataset)

        if rank == 0:
            with (args.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            state_model = model.module if isinstance(model, DistributedDataParallel) else model
            checkpoint = {
                "epoch": epoch,
                "model": state_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "scene_config": scene_config.__dict__,
                "best_validation_loss": min(best, validation["loss"]),
                "world_size": world_size,
            }
            torch.save(checkpoint, args.output / "latest.pt")
            if args.save_every > 0 and epoch % args.save_every == 0:
                torch.save(checkpoint, args.output / f"epoch-{epoch:03d}.pt")
            if validation["loss"] < best:
                best = validation["loss"]
                checkpoint["best_validation_loss"] = best
                torch.save(checkpoint, args.output / "best.pt")
            print(
                f"epoch={epoch}/{args.epochs} train={train['loss']:.4f} "
                f"validation={validation['loss']:.4f} "
                f"node_xy={values['node_xy']:.4f} edge_presence={values['edge_presence']:.4f} "
                f"edge_xy={values['edge_xy']:.4f} building_presence={values['building_presence']:.4f} "
                f"building_shape={values['building_shape']:.4f} "
                f"area_presence={values['area_presence']:.4f} area_shape={values['area_shape']:.4f} "
                f"epoch_s={epoch_seconds:.1f} elapsed_min={elapsed_seconds / 60.0:.1f} "
                f"examples_per_s={examples / max(epoch_seconds, 1e-6):.2f}",
                flush=True,
            )

    if rank == 0:
        summary = {
            "epochs": args.epochs,
            "best_validation_loss": best,
            "seconds": time.time() - started,
            "samples": len(dataset),
            "splits": {name: len(values) for name, values in splits.items()},
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
