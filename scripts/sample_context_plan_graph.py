from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sample_plan_cell_graph import (
    comparison_stats,
    generated_graph,
    graph_stats,
    move_sample,
    render,
    target_graph,
)
from urban_model.city_plan_data import CityPlanConfig, CityPlanDataset
from urban_model.context_plan_graph import ContextPlanGraph, context_inputs
from urban_model.spatial_world_data import SpatialTensorConfig


def save_predictions(path, output, plan, planner):
    count = int(plan["node_count"])
    arrays = {}
    for name, value in output.items():
        if name == "node_hidden":
            continue
        value = value[0, :count]
        if name.startswith("edge_"):
            value = value[:, :count]
        arrays[name] = value.detach().cpu().numpy()
    for name in (
        "plan_counts",
        "plan_presence",
        "plan_orientation",
        "plan_global_raw",
        "node_count",
    ):
        arrays[name] = plan[name].detach().cpu().numpy()
    for name, value in planner.items():
        arrays[f"planner_{name}"] = value[0].detach().cpu().numpy()
    np.savez_compressed(path, **arrays)


def plan_stats(target, predicted):
    actual = target["plan_counts"][:, 0].detach().cpu()
    counts = predicted["plan_counts"][:, 0].detach().cpu()
    return {
        "node_count_error": int(counts.sum() - actual.sum()),
        "cell_count_l1": int((counts - actual).abs().sum()),
        "changed_cells": int((counts != actual).sum()),
        "exact_cell_count_fraction": float((counts == actual).float().mean()),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[],
        help="Add independent count-noise samples after the deterministic result",
    )
    parser.add_argument("--compare-decoders", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tensor_config = SpatialTensorConfig(**checkpoint["tensor_config"])
    plan_config = CityPlanConfig(**checkpoint["plan_config"])
    dataset = CityPlanDataset(
        args.data,
        tensor_config=tensor_config,
        plan_config=plan_config,
        maximum_samples=checkpoint["maximum_samples"],
        normalization=checkpoint["normalization"],
        split_strategy=checkpoint["split_strategy"],
    )
    split = args.split or ("train" if checkpoint["overfit"] else "validation")
    wanted = set(checkpoint["sample_ids"][split])
    samples = [sample for sample in dataset.samples if sample["sample_id"] in wanted][
        : args.samples
    ]
    if not samples:
        raise ValueError(f"No saved {split} samples found")
    device = torch.device(args.device)
    model = ContextPlanGraph(
        checkpoint["planner_config"], checkpoint["graph_config"], checkpoint["normalization"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, sample in enumerate(samples):
        batch = move_sample(sample, device)
        context = context_inputs(batch, use_context=not checkpoint["no_context"])
        target = target_graph(sample, tensor_config, plan_config.grid_size)
        predicted, output = model(context, batch)
        reconstructed = generated_graph(output, sample, tensor_config, strategy="raw")
        record = {
            "sample_id": sample["sample_id"],
            "split": split,
            "overfit": checkpoint["overfit"],
            "coordinate_system": {
                "units": "metres",
                "target_size_m": tensor_config.target_size_m,
                "metric_elevation": "unavailable",
            },
            "target": target,
            "reconstruction": {
                "graph": reconstructed,
                "uses_target_plan": True,
                "metrics": comparison_stats(target, reconstructed),
                "statistics": graph_stats(reconstructed),
            },
            "generations": [],
        }
        panels = [
            render(target, sample, tensor_config.target_size_m, size=512),
            render(reconstructed, sample, tensor_config.target_size_m, size=512),
        ]
        labels = ["target", "target plan: old edge selection"]
        stem = f"{index:02d}-{sample['sample_id']}"
        if args.save_predictions:
            save_predictions(args.output / f"{stem}-reconstruction.npz", output, sample, predicted)
        if args.compare_decoders:
            decoded = generated_graph(output, sample, tensor_config, strategy="learned_degree")
            record["degree_reconstruction"] = {
                "graph": decoded,
                "uses_target_plan": True,
                "metrics": comparison_stats(target, decoded),
                "statistics": graph_stats(decoded),
                "solver": decoded["decoder"],
            }
            panels.append(render(decoded, sample, tensor_config.target_size_m, size=512))
            labels.append("target plan: learned degree selection")
        strategies = ["compatible", "learned_degree"] if args.compare_decoders else ["compatible"]
        for seed in [None, *args.seeds]:
            generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
            plan, generated = model.generate(
                context, stochastic=seed is not None, generator=generator
            )
            single_plan = {key: value[0] for key, value in plan.items()}
            tag = "deterministic" if seed is None else f"seed-{seed}"
            if args.save_predictions:
                save_predictions(
                    args.output / f"{stem}-{tag}.npz", generated, single_plan, predicted
                )
            for strategy in strategies:
                graph = generated_graph(generated, single_plan, tensor_config, strategy=strategy)
                record["generations"].append(
                    {
                        "seed": seed,
                        "sampling": "deterministic" if seed is None else "independent count noise",
                        "decoder": strategy,
                        "uses_target_plan": False,
                        "controls": "disabled",
                        "graph": graph,
                        "statistics": graph_stats(graph),
                        "solver": graph.get("decoder"),
                        "plan_metrics": plan_stats(sample, single_plan),
                        "predicted_nodes": int(single_plan["node_count"]),
                        "predicted_edges": int(single_plan["plan_global_raw"][1]),
                    }
                )
                panels.append(render(graph, single_plan, tensor_config.target_size_m, size=512))
                selection = (
                    "old edge selection" if strategy == "compatible" else "learned degree selection"
                )
                labels.append(f"{tag}: {selection}")
        image = Image.new("RGB", (512 * len(panels), 540), "white")
        draw = ImageDraw.Draw(image)
        for column, (panel, label) in enumerate(zip(panels, labels, strict=True)):
            image.paste(panel, (column * 512, 28))
            draw.text((column * 512 + 5, 7), label, fill="black")
        image.save(args.output / f"{stem}.png")
        (args.output / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
        summaries.append(
            {
                "sample_id": sample["sample_id"],
                "target": graph_stats(target),
                "reconstruction": record["reconstruction"]["metrics"],
                "reconstruction_statistics": record["reconstruction"]["statistics"],
                "degree_reconstruction": {
                    key: value
                    for key, value in record.get("degree_reconstruction", {}).items()
                    if key != "graph"
                },
                "generations": [
                    {key: value for key, value in result.items() if key != "graph"}
                    for result in record["generations"]
                ],
            }
        )
        print(f"Saved {sample['sample_id']} ({index + 1}/{len(samples)})", flush=True)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "epoch": checkpoint["epoch"],
                "split": split,
                "overfit": checkpoint["overfit"],
                "sampling": "deterministic plan; optional seeds add independent cell count noise",
                "compare_decoders": args.compare_decoders,
                "saved_predictions": args.save_predictions,
                "decoder": "top-count selection, plus learned degree selection when compare_decoders is true",
                "degree_decoder": "experimental; predicted degree capacities and estimated BCE calibration",
                "precision": "float32",
                "source_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=Path(__file__).resolve().parents[1],
                ).stdout.strip(),
                "tensor_config": checkpoint["tensor_config"],
                "graph_config": checkpoint["graph_config"],
                "plan_config": checkpoint["plan_config"],
                "training_source_commit": checkpoint.get("source_commit"),
                "samples": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(samples)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
