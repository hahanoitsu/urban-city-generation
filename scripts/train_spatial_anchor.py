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

from urban_model.spatial_anchor import SpatialAnchorArchitect, SpatialAnchorModelConfig
from urban_model.spatial_anchor_data import AnchoredSpatialWorldDataset, SpatialAnchorConfig
from urban_model.spatial_anchor_loss import spatial_anchor_loss
from urban_model.spatial_world_data import SpatialTensorConfig


def distributed_state():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def split_indices(dataset):
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
    device,
    *,
    optimizer=None,
    kl_weight=0.0,
    use_posterior=True,
    control_dropout=0.0,
):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    examples = 0
    metric_sums = {}
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for batch in loader:
            batch = move_batch(batch, device)
            batch_size = int(batch["context_cells"].shape[0])
            if training and control_dropout > 0:
                keep = (
                    torch.rand(batch_size, 1, device=device) >= control_dropout
                ).to(batch["controls"].dtype)
                batch["controls"] = batch["controls"] * keep

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    batch,
                    use_posterior=use_posterior,
                    sample_latent=training,
                )
                loss, metrics = spatial_anchor_loss(
                    output,
                    batch,
                    kl_weight=kl_weight,
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
                metric_sums[name] = metric_sums.get(name, 0.0) + value * batch_size

    return reduce_metrics(loss_sum, examples, metric_sums, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--kl-warmup", type=int, default=8)
    parser.add_argument("--control-dropout", type=float, default=0.5)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--save-every", type=int, default=5)
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
    anchor_config = SpatialAnchorConfig(
        grid_size=32,
        slots_per_cell=12,
        max_active_nodes=384,
        max_edges=512,
    )
    dataset = AnchoredSpatialWorldDataset(
        args.data,
        tensor_config=tensor_config,
        anchor_config=anchor_config,
        maximum_samples=args.maximum_samples,
    )
    splits = split_indices(dataset)
    if not splits["train"] or not splits["validation"]:
        raise RuntimeError("Anchored spatial split produced an empty train or validation set")

    if rank == 0:
        print(
            json.dumps(
                {
                    "samples": len(dataset),
                    "base_rejected": dataset.base_rejected,
                    "anchor_rejected": dataset.rejected,
                    "duplicate_edges": dataset.duplicate_edges,
                    "splits": {name: len(values) for name, values in splits.items()},
                    "world_size": world_size,
                },
                indent=2,
            ),
            flush=True,
        )

    device = torch.device("cuda", local_rank) if world_size > 1 else torch.device("cuda")
    model_config = SpatialAnchorModelConfig(
        context_dimensions=dataset.context_dimensions,
        style_dimensions=dataset.style_dimensions,
        grid_size=anchor_config.grid_size,
        slots_per_cell=anchor_config.slots_per_cell,
        max_active_nodes=anchor_config.max_active_nodes,
        context_line_points=tensor_config.context_line_points,
        edge_shape_points=tensor_config.edge_shape_points,
    )
    model = SpatialAnchorArchitect(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        fused=True,
    )

    start_epoch = 0
    best_prior = math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        start_epoch = int(checkpoint.get("epoch", 0))
        best_prior = float(checkpoint.get("best_prior_loss", math.inf))
        if rank == 0:
            print(f"resume={args.resume} epoch={start_epoch}", flush=True)

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
            "anchor_config": anchor_config.__dict__,
            "model_config": model_config.to_dict(),
            "samples": len(dataset),
            "base_rejected": dataset.base_rejected,
            "anchor_rejected": dataset.rejected,
            "duplicate_edges": dataset.duplicate_edges,
            "splits": {name: len(values) for name, values in splits.items()},
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
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
        kl_weight = args.kl_weight * min(1.0, epoch / max(args.kl_warmup, 1))

        train = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            kl_weight=kl_weight,
            use_posterior=True,
            control_dropout=args.control_dropout,
        )
        validation = run_epoch(
            model,
            validation_loader,
            device,
            kl_weight=kl_weight,
            use_posterior=True,
        )
        prior = run_epoch(
            model,
            validation_loader,
            device,
            kl_weight=0.0,
            use_posterior=False,
        )

        epoch_seconds = time.time() - epoch_started
        elapsed = time.time() - started
        if rank == 0:
            record = {
                "epoch": epoch,
                "kl_weight": kl_weight,
                "train": train,
                "validation": validation,
                "prior": prior,
                "seconds": elapsed,
            }
            with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            state_model = model.module if isinstance(model, DistributedDataParallel) else model
            checkpoint = {
                "epoch": epoch,
                "model": state_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config.to_dict(),
                "tensor_config": tensor_config.__dict__,
                "anchor_config": anchor_config.__dict__,
                "best_prior_loss": min(best_prior, prior["loss"]),
            }
            torch.save(checkpoint, args.output / "latest.pt")
            if args.save_every > 0 and epoch % args.save_every == 0:
                torch.save(checkpoint, args.output / f"epoch-{epoch:03d}.pt")
            if prior["loss"] < best_prior:
                best_prior = prior["loss"]
                checkpoint["best_prior_loss"] = best_prior
                torch.save(checkpoint, args.output / "best.pt")

            parts = prior["parts"]
            print(
                f"epoch={epoch}/{args.epochs} "
                f"train={train['loss']:.4f} val={validation['loss']:.4f} "
                f"prior={prior['loss']:.4f} kl={validation['parts']['kl']:.4f} "
                f"cell_count={parts['cell_count']:.4f} "
                f"slot={parts['slot_score']:.4f} offset={parts['node_offset']:.4f} "
                f"edge={parts['edge_relation']:.4f} shape={parts['edge_shape']:.4f} "
                f"epoch_s={epoch_seconds:.1f} elapsed_min={elapsed / 60.0:.1f}",
                flush=True,
            )

    if rank == 0:
        summary = {
            "epochs": args.epochs,
            "best_prior_loss": best_prior,
            "seconds": time.time() - started,
            "samples": len(dataset),
            "splits": {name: len(values) for name, values in splits.items()},
            "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
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
