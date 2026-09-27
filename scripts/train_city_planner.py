from __future__ import annotations

import argparse
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

from urban_model.city_plan_data import CityPlanConfig, CityPlanDataset
from urban_model.city_plan_loss import city_plan_loss
from urban_model.city_planner import CityPlanner, CityPlannerConfig
from urban_model.spatial_world_data import SpatialTensorConfig


def distributed_state():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


CONDITION_KEYS = (
    "context_cells",
    "context_line_points",
    "context_line_mode",
    "context_line_class",
    "context_line_vertical",
    "context_line_width",
    "context_line_length",
    "context_line_padding",
    "ports",
    "port_mode",
    "port_class",
    "port_vertical",
    "port_padding",
)


def move_batch(batch, device, *, zero_controls, shuffle_context=False):
    values = {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }
    if zero_controls:
        values["controls"] = torch.zeros_like(values["controls"])
    if shuffle_context and values["context_cells"].shape[0] > 1:
        for key in CONDITION_KEYS:
            values[key] = torch.roll(values[key], shifts=1, dims=0)
    return values


def split_indices(dataset, overfit):
    if overfit:
        values = list(range(len(dataset)))
        return {"train": values, "validation": values, "test": values}
    result = {"train": [], "validation": [], "test": []}
    for index, sample in enumerate(dataset.samples):
        result[str(sample["split"])].append(index)
    return result


def reduce_metrics(loss_sum, examples, metric_sums, device):
    names = sorted(metric_sums)
    values = torch.tensor(
        [loss_sum, float(examples), *[metric_sums[name] for name in names]],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = max(float(values[1].item()), 1.0)
    return {
        "loss": float(values[0].item() / count),
        "parts": {
            name: float(values[index + 2].item() / count)
            for index, name in enumerate(names)
        },
        "examples": int(values[1].item()),
    }


def run_epoch(
    model,
    loader,
    dataset,
    device,
    *,
    optimizer=None,
    zero_controls=True,
    shuffle_context=False,
):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    examples = 0
    metric_sums = {}
    context = torch.enable_grad() if training else torch.inference_mode()

    plan_mean = dataset.plan_mean.to(device)
    plan_std = dataset.plan_std.to(device)
    global_mean = dataset.global_mean.to(device)
    global_std = dataset.global_std.to(device)

    with context:
        for batch in loader:
            batch = move_batch(
                batch,
                device,
                zero_controls=zero_controls,
                shuffle_context=shuffle_context,
            )
            batch_size = int(batch["plan_global"].shape[0])

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                output = model(batch)
                loss, metrics = city_plan_loss(
                    output,
                    batch,
                    plan_mean=plan_mean,
                    plan_std=plan_std,
                    global_mean=global_mean,
                    global_std=global_std,
                )

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

            loss_sum += float(loss.detach()) * batch_size
            examples += batch_size
            for name, value in metrics.items():
                metric_sums[name] = (
                    metric_sums.get(name, 0.0)
                    + value * batch_size
                )

    return reduce_metrics(
        loss_sum,
        examples,
        metric_sums,
        device,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--use-target-controls", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    rank, local_rank, world_size = distributed_state()
    torch.manual_seed(5132 + rank)
    torch.cuda.manual_seed_all(5132 + rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    tensor_config = SpatialTensorConfig(
        max_nodes=384,
        max_edges=512,
        max_context_lines=768,
        max_ports=128,
    )
    plan_config = CityPlanConfig(grid_size=args.grid_size)
    dataset = CityPlanDataset(
        args.data,
        tensor_config=tensor_config,
        plan_config=plan_config,
        maximum_samples=args.maximum_samples,
    )
    splits = split_indices(dataset, args.overfit)
    if not splits["train"] or not splits["validation"]:
        raise RuntimeError(
            "City planner split produced an empty train or validation set"
        )

    if rank == 0:
        print(
            json.dumps(
                {
                    "samples": len(dataset),
                    "base_rejected": dataset.base_rejected,
                    "splits": {
                        name: len(values)
                        for name, values in splits.items()
                    },
                    "grid_size": plan_config.grid_size,
                    "plan_channels": dataset.plan_dimensions,
                    "global_channels": dataset.global_dimensions,
                    "world_size": world_size,
                    "overfit": args.overfit,
                    "target_controls": args.use_target_controls,
                },
                indent=2,
            ),
            flush=True,
        )

    device = (
        torch.device("cuda", local_rank)
        if world_size > 1
        else torch.device("cuda")
    )
    model_config = CityPlannerConfig(
        context_dimensions=dataset.context_dimensions,
        style_dimensions=dataset.style_dimensions,
        plan_dimensions=dataset.plan_dimensions,
        global_dimensions=dataset.global_dimensions,
        grid_size=plan_config.grid_size,
        context_line_points=tensor_config.context_line_points,
    )
    model = CityPlanner(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        fused=True,
    )

    start_epoch = 0
    best_validation = math.inf
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        start_epoch = int(checkpoint.get("epoch", 0))
        best_validation = float(
            checkpoint.get(
                "best_validation_loss",
                math.inf,
            )
        )
        if rank == 0:
            print(
                f"resume={args.resume} epoch={start_epoch}",
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
    validation_subset = Subset(
        dataset,
        splits["validation"],
    )
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
    options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        train_subset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **options,
    )
    validation_loader = DataLoader(
        validation_subset,
        shuffle=False,
        sampler=validation_sampler,
        **options,
    )

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "tensor_config": tensor_config.__dict__,
            "plan_config": plan_config.__dict__,
            "model_config": model_config.to_dict(),
            "samples": len(dataset),
            "splits": {
                name: len(values)
                for name, values in splits.items()
            },
            "plan_mean": dataset.plan_mean.tolist(),
            "plan_std": dataset.plan_std.tolist(),
            "global_mean": dataset.global_mean.tolist(),
            "global_std": dataset.global_std.tolist(),
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
            "overfit": args.overfit,
            "target_controls": args.use_target_controls,
            "maximum_samples": args.maximum_samples,
        }
        (args.output / "experiment.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    started = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_started = time.time()
        train = run_epoch(
            model,
            train_loader,
            dataset,
            device,
            optimizer=optimizer,
            zero_controls=not args.use_target_controls,
        )
        validation = run_epoch(
            model,
            validation_loader,
            dataset,
            device,
            zero_controls=not args.use_target_controls,
        )
        shuffled = run_epoch(
            model,
            validation_loader,
            dataset,
            device,
            zero_controls=not args.use_target_controls,
            shuffle_context=True,
        )
        elapsed = time.time() - started
        epoch_seconds = time.time() - epoch_started

        if rank == 0:
            state_model = (
                model.module
                if isinstance(
                    model,
                    DistributedDataParallel,
                )
                else model
            )
            checkpoint = {
                "epoch": epoch,
                "model": state_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "tensor_config": tensor_config.__dict__,
                "plan_config": plan_config.__dict__,
                "plan_mean": dataset.plan_mean,
                "plan_std": dataset.plan_std,
                "global_mean": dataset.global_mean,
                "global_std": dataset.global_std,
                "best_validation_loss": min(
                    best_validation,
                    validation["loss"],
                ),
                "target_controls": args.use_target_controls,
                "maximum_samples": args.maximum_samples,
            }
            torch.save(
                checkpoint,
                args.output / "latest.pt",
            )
            if (
                args.save_every > 0
                and epoch % args.save_every == 0
            ):
                torch.save(
                    checkpoint,
                    args.output / f"epoch-{epoch:03d}.pt",
                )
            if validation["loss"] < best_validation:
                best_validation = validation["loss"]
                checkpoint["best_validation_loss"] = (
                    best_validation
                )
                torch.save(
                    checkpoint,
                    args.output / "best.pt",
                )

            record = {
                "epoch": epoch,
                "train": train,
                "validation": validation,
                "shuffled_context": shuffled,
                "seconds": elapsed,
            }
            with (
                args.output / "metrics.jsonl"
            ).open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record) + "\n"
                )

            parts = validation["parts"]
            print(
                f"epoch={epoch}/{args.epochs} "
                f"train={train['loss']:.4f} "
                f"validation={validation['loss']:.4f} "
                f"node_mae={parts['node_mae']:.2f} "
                f"edge_mae={parts['edge_mae']:.2f} "
                f"component_mae={parts['component_mae']:.2f} "
                f"occupancy_iou={parts['occupancy_iou']:.3f} "
                f"plan_mae={parts['raw_plan_mae']:.3f} "
                f"context_gap={shuffled['loss'] - validation['loss']:.4f} "
                f"epoch_s={epoch_seconds:.1f} "
                f"elapsed_min={elapsed / 60.0:.1f}",
                flush=True,
            )

    if rank == 0:
        summary = {
            "epochs": args.epochs,
            "best_validation_loss": best_validation,
            "seconds": time.time() - started,
            "samples": len(dataset),
            "splits": {
                name: len(values)
                for name, values in splits.items()
            },
            "world_size": world_size,
            "global_batch_size": args.batch_size
            * world_size,
            "overfit": args.overfit,
            "target_controls": args.use_target_controls,
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
