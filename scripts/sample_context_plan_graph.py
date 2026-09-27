from __future__ import annotations

import argparse
import json
from pathlib import Path

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


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 37])
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
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
        _, output = model(context, batch)
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
            },
            "generations": [],
        }
        panels = [
            render(target, sample, tensor_config.target_size_m, size=512),
            render(reconstructed, sample, tensor_config.target_size_m, size=512),
        ]
        labels = ["target", "reconstruction: target plan supplied"]
        for seed in args.seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            plan, generated = model.generate(context, stochastic=True, generator=generator)
            single_plan = {key: value[0] for key, value in plan.items()}
            graph = generated_graph(generated, single_plan, tensor_config, strategy="compatible")
            record["generations"].append(
                {
                    "seed": seed,
                    "uses_target_plan": False,
                    "controls": "disabled",
                    "graph": graph,
                    "statistics": graph_stats(graph),
                    "predicted_nodes": int(single_plan["node_count"]),
                    "predicted_edges": int(single_plan["plan_global_raw"][1]),
                }
            )
            panels.append(render(graph, single_plan, tensor_config.target_size_m, size=512))
            labels.append(f"context generation, seed {seed}")
        image = Image.new("RGB", (512 * len(panels), 540), "white")
        draw = ImageDraw.Draw(image)
        for column, (panel, label) in enumerate(zip(panels, labels, strict=True)):
            image.paste(panel, (column * 512, 28))
            draw.text((column * 512 + 5, 7), label, fill="black")
        stem = f"{index:02d}-{sample['sample_id']}"
        image.save(args.output / f"{stem}.png")
        (args.output / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
        summaries.append(
            {
                "sample_id": sample["sample_id"],
                "target": graph_stats(target),
                "reconstruction": record["reconstruction"]["metrics"],
                "generations": [
                    {key: value for key, value in result.items() if key != "graph"}
                    for result in record["generations"]
                ],
            }
        )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "epoch": checkpoint["epoch"],
                "split": split,
                "overfit": checkpoint["overfit"],
                "sampling": "independent cell occupancy and shifted Poisson node counts",
                "decoder": "highest scoring same-mode pairs, predicted edge budget; no component repair",
                "samples": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(samples)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
