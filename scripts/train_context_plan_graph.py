from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from urban_model.city_plan_data import CityPlanConfig, CityPlanDataset
from urban_model.city_plan_loss import city_plan_loss
from urban_model.city_planner import CityPlannerConfig
from urban_model.context_plan_graph import ContextPlanGraph, context_inputs
from urban_model.plan_cell_graph import PlanCellGraphConfig
from urban_model.plan_cell_graph_loss import plan_cell_graph_loss
from urban_model.spatial_world_data import SpatialTensorConfig


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def run_epoch(model, loader, device, tensor_config, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    examples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move_batch(batch, device)
            context = context_inputs(batch, use_context=not args.no_context)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                plan, graph = model(context, batch)
                graph_loss, graph_metrics, _ = plan_cell_graph_loss(
                    graph,
                    batch,
                    target_size_m=tensor_config.target_size_m,
                    grid_size=args.grid_size,
                )
                plan_loss, plan_metrics = city_plan_loss(
                    plan,
                    batch,
                    presence_pos_weight=model.presence_pos_weight,
                    global_mean=model.global_mean,
                    global_std=model.global_std,
                )
            occupied = batch["plan_counts"][..., 0] > 0
            rate = (torch.expm1(plan["plan_log_count"].float()[..., 0].clamp_max(8)) - 1).clamp_min(
                1e-4
            )
            count_loss = (
                F.poisson_nll_loss(
                    rate[occupied], batch["plan_counts"][..., 0][occupied] - 1, log_input=False
                )
                if occupied.any()
                else rate.sum() * 0
            )
            loss = graph_loss + 0.5 * plan_loss + 0.05 * count_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for {batch['sample_id']}")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            metrics = {
                "loss": float(loss.detach()),
                **graph_metrics,
                **{f"plan_{key}": value for key, value in plan_metrics.items()},
            }
            count = len(batch["sample_id"])
            examples += count
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * count
    return {name: value / max(examples, 1) for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=384)
    parser.add_argument("--max-edges", type=int, default=512)
    parser.add_argument("--seed", type=int, default=5132)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Output is not empty: {args.output}. Use a new run directory.")
    if args.dimensions % 8:
        raise ValueError("Dimensions must be divisible by 8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    tensor_config = SpatialTensorConfig.from_dataset(
        args.data, simple_graph=True, max_nodes=args.max_nodes, max_edges=args.max_edges
    )
    plan_config = CityPlanConfig(grid_size=args.grid_size)
    split_strategy = "legacy" if args.overfit else "buffered"
    dataset = CityPlanDataset(
        args.data,
        tensor_config=tensor_config,
        plan_config=plan_config,
        maximum_samples=args.maximum_samples,
        normalization_split="all" if args.overfit else "train",
        split_strategy=split_strategy,
    )
    if args.overfit:
        splits = {name: list(range(len(dataset))) for name in ("train", "validation")}
    else:
        splits = {
            name: [i for i, sample in enumerate(dataset.samples) if sample["split"] == name]
            for name in ("train", "validation", "test", "buffer")
        }
    if not splits["train"] or not splits["validation"]:
        raise RuntimeError(
            "Empty training or validation split after the context buffer. Use more samples."
        )
    if any(int(sample["node_count"]) < 2 for sample in dataset.samples):
        raise RuntimeError("Dataset contains a transport graph with fewer than two nodes")
    if any(int(sample["plan_counts"][:, 0].max()) > 64 for sample in dataset.samples):
        raise RuntimeError("A cell exceeds 64 nodes. Increase --grid-size.")
    shared = dict(
        model_dimensions=args.dimensions,
        heads=8,
        feedforward_dimensions=args.dimensions * 4,
        dropout=0.0 if args.overfit else 0.1,
    )
    planner_config = CityPlannerConfig(
        context_dimensions=dataset.context_dimensions,
        style_dimensions=dataset.style_dimensions,
        plan_dimensions=dataset.plan_dimensions,
        orientation_dimensions=dataset.orientation_dimensions,
        global_dimensions=dataset.global_dimensions,
        grid_size=args.grid_size,
        context_line_points=tensor_config.context_line_points,
        planner_layers=args.layers,
        **shared,
    )
    graph_config = PlanCellGraphConfig(
        plan_dimensions=dataset.plan_dimensions,
        orientation_dimensions=dataset.orientation_dimensions,
        global_dimensions=dataset.global_dimensions,
        plan_grid_size=args.grid_size,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        plan_layers=args.layers,
        node_layers=args.layers,
        edge_shape_points=tensor_config.edge_shape_points,
        **shared,
    )
    model = ContextPlanGraph(
        planner_config.to_dict(), graph_config.to_dict(), dataset.normalization
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loaders = {
        name: DataLoader(
            Subset(dataset, splits[name]),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name in ("train", "validation")
    }
    checkpoint_info = {
        "format": "context-plan-graph-v1",
        "planner_config": planner_config.to_dict(),
        "graph_config": graph_config.to_dict(),
        "tensor_config": asdict(tensor_config),
        "plan_config": asdict(plan_config),
        "normalization": dataset.normalization,
        "overfit": args.overfit,
        "no_context": args.no_context,
        "controls": "disabled",
        "split_strategy": split_strategy,
        "maximum_samples": args.maximum_samples,
        "sample_ids": {
            name: [dataset.samples[i]["sample_id"] for i in values]
            for name, values in splits.items()
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "base_rejected": dataset.base_rejected,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "experiment.json").write_text(json.dumps(checkpoint_info, indent=2) + "\n")
    print(
        json.dumps(
            {
                "samples": len(dataset),
                "rejected": dataset.base_rejected,
                "splits": {key: len(value) for key, value in splits.items()},
                "overfit": args.overfit,
                "target_size_m": tensor_config.target_size_m,
            }
        ),
        flush=True,
    )
    best = float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], device, tensor_config, args, optimizer)
        validation = run_epoch(model, loaders["validation"], device, tensor_config, args)
        record = {
            "epoch": epoch,
            "train": train,
            "validation": validation,
            "validation_kind": "training reconstruction"
            if args.overfit
            else "held-out target-plan reconstruction",
            "seconds": time.time() - started,
        }
        with (args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            **checkpoint_info,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "validation": validation,
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if validation["loss"] < best:
            best = validation["loss"]
            torch.save(checkpoint, args.output / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            torch.save(checkpoint, args.output / f"epoch-{epoch:03d}.pt")
        print(
            f"epoch={epoch}/{args.epochs} loss={validation['loss']:.4f} "
            f"node_m={validation['node_position_mae_m']:.2f} edge_recall={validation['edge_recall']:.3f} "
            f"plan_nodes_mae={validation['plan_node_mae']:.2f} elapsed_min={(time.time() - started) / 60:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
