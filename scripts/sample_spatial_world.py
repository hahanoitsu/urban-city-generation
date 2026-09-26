from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from urban_model.spatial_world import SpatialWorldArchitect, SpatialWorldModelConfig
from urban_model.spatial_world_data import (
    RAIL_CLASSES,
    ROAD_CLASSES,
    SpatialTensorConfig,
    SpatialWorldDataset,
    TRANSPORT_CLASSES,
    VERTICAL_MODES,
)


def move_sample(sample, device):
    return {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


def count_from_logits(logits):
    return int(logits.argmax(dim=-1)[0].item())


def decode_generation(output, config):
    node_count = min(count_from_logits(output["node_count"]), config.max_nodes)
    edge_count = min(count_from_logits(output["edge_count"]), config.max_edges)
    node_xy = output["node_xy"][0, :node_count].detach().cpu()
    node_mode = output["node_mode"][0, :node_count].argmax(dim=-1).detach().cpu()
    node_vertical = output["node_vertical"][0, :node_count].argmax(dim=-1).detach().cpu()
    boundary = torch.sigmoid(output["node_boundary"][0, :node_count]).detach().cpu()

    graph = {
        "nodes": [],
        "edges": [],
    }
    for index in range(node_count):
        xy = (node_xy[index] + 1.0) * 0.5 * config.target_size_m
        graph["nodes"].append(
            {
                "id": index,
                "position_local_m": [float(xy[0]), float(xy[1]), None],
                "mode": "road" if int(node_mode[index]) == 0 else "rail",
                "vertical_mode": VERTICAL_MODES[int(node_vertical[index])],
                "boundary_score": float(boundary[index]),
            }
        )

    if node_count < 2:
        return graph

    valid_nodes = torch.arange(config.max_nodes, device=output["edge_from"].device)
    invalid = valid_nodes >= node_count
    for index in range(edge_count):
        from_logits = output["edge_from"][0, index].clone()
        to_logits = output["edge_to"][0, index].clone()
        from_logits[invalid] = float("-inf")
        to_logits[invalid] = float("-inf")
        left = int(from_logits.argmax().item())
        to_logits[left] = float("-inf")
        right = int(to_logits.argmax().item())
        start = node_xy[left]
        end = node_xy[right]
        internal = []
        for point_index in range(config.edge_shape_points):
            fraction = (point_index + 1) / (config.edge_shape_points + 1)
            base = start + (end - start) * fraction
            internal.append(base + output["edge_shape"][0, index, point_index].detach().cpu())
        values = [start, *internal, end]
        geometry = [
            [
                float((point[0] + 1.0) * 0.5 * config.target_size_m),
                float((point[1] + 1.0) * 0.5 * config.target_size_m),
                None,
            ]
            for point in values
        ]
        mode_index = int(output["edge_mode"][0, index].argmax().item())
        class_index = int(output["edge_class"][0, index].argmax().item())
        vertical_index = int(output["edge_vertical"][0, index].argmax().item())
        width = float(
            output["edge_width"][0, index, 0].detach().cpu()
            * config.width_scale_m
        )
        graph["edges"].append(
            {
                "id": index,
                "from_node": left,
                "to_node": right,
                "mode": "road" if mode_index == 0 else "rail",
                "class": TRANSPORT_CLASSES[class_index],
                "vertical_mode": VERTICAL_MODES[vertical_index],
                "width_m": max(width, 0.0),
                "geometry_local_m": geometry,
            }
        )
    return graph


def decode_target(sample, config):
    node_count = int(sample["node_count"])
    edge_count = int(sample["edge_count"])
    nodes = []
    for index in range(node_count):
        xy = (sample["node_xy"][index] + 1.0) * 0.5 * config.target_size_m
        nodes.append(
            {
                "id": index,
                "position_local_m": [float(xy[0]), float(xy[1]), None],
                "mode": "road" if int(sample["node_mode"][index]) == 0 else "rail",
                "vertical_mode": VERTICAL_MODES[int(sample["node_vertical"][index])],
            }
        )

    edges = []
    for index in range(edge_count):
        left = int(sample["edge_from"][index])
        right = int(sample["edge_to"][index])
        start = sample["node_xy"][left]
        end = sample["node_xy"][right]
        internal = []
        for point_index in range(config.edge_shape_points):
            fraction = (point_index + 1) / (config.edge_shape_points + 1)
            base = start + (end - start) * fraction
            internal.append(base + sample["edge_shape"][index, point_index])
        values = [start, *internal, end]
        geometry = [
            [
                float((point[0] + 1.0) * 0.5 * config.target_size_m),
                float((point[1] + 1.0) * 0.5 * config.target_size_m),
                None,
            ]
            for point in values
        ]
        class_index = int(sample["edge_class"][index])
        edges.append(
            {
                "id": index,
                "from_node": left,
                "to_node": right,
                "mode": "road" if int(sample["edge_mode"][index]) == 0 else "rail",
                "class": TRANSPORT_CLASSES[class_index],
                "vertical_mode": VERTICAL_MODES[int(sample["edge_vertical"][index])],
                "width_m": float(sample["edge_width"][index, 0] * config.width_scale_m),
                "geometry_local_m": geometry,
            }
        )
    return {"nodes": nodes, "edges": edges}


def render(graph, size=720):
    image = Image.new("RGB", (size, size), (247, 246, 242))
    draw = ImageDraw.Draw(image)

    def point(value):
        return (
            int(round(value[0] / 1024.0 * (size - 1))),
            int(round((1.0 - value[1] / 1024.0) * (size - 1))),
        )

    for edge in graph["edges"]:
        values = [point(value) for value in edge["geometry_local_m"]]
        colour = (205, 75, 55) if edge["mode"] == "road" else (55, 125, 185)
        draw.line(values, fill=colour, width=3, joint="curve")

    for node in graph["nodes"]:
        x, y = point(node["position_local_m"])
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(30, 30, 30))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = SpatialWorldModelConfig.from_dict(checkpoint["model_config"])
    tensor_config = SpatialTensorConfig(**checkpoint["tensor_config"])
    dataset = SpatialWorldDataset(args.data, config=tensor_config)

    candidates = [
        index
        for index, sample in enumerate(dataset.samples)
        if sample["split"] == "test"
    ]
    candidates.sort(
        key=lambda index: hashlib.sha1(
            str(dataset.samples[index]["sample_id"]).encode("utf-8"),
            usedforsecurity=False,
        ).digest()
    )
    indexes = candidates[: args.samples]
    if not indexes:
        raise RuntimeError("No held-out spatial world samples found")

    device = torch.device("cuda")
    model = SpatialWorldArchitect(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    panels = []
    for sample_index, dataset_index in enumerate(indexes):
        sample = dataset[dataset_index]
        batch = move_sample(sample, device)
        target = decode_target(sample, tensor_config)
        target_image = render(target)

        images = [target_image]
        sample_record = {
            "sample_id": sample["sample_id"],
            "target": {
                "nodes": len(target["nodes"]),
                "edges": len(target["edges"]),
            },
            "generations": [],
        }

        for seed_index in range(args.seeds):
            torch.manual_seed(1000 + sample_index * 100 + seed_index)
            torch.cuda.manual_seed_all(1000 + sample_index * 100 + seed_index)
            output = model.generate(batch, temperature=args.temperature)
            generated = decode_generation(output, tensor_config)
            images.append(render(generated))
            sample_record["generations"].append(
                {
                    "seed": seed_index,
                    "nodes": len(generated["nodes"]),
                    "edges": len(generated["edges"]),
                }
            )
            path = args.output / (
                f"{sample_index:02d}-{sample['sample_id']}-seed{seed_index}.json"
            )
            path.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")

        panel = Image.new("RGB", (720 * len(images), 750), "white")
        draw = ImageDraw.Draw(panel)
        for image_index, image in enumerate(images):
            panel.paste(image, (image_index * 720, 30))
            title = "target" if image_index == 0 else f"seed {image_index - 1}"
            draw.text((image_index * 720 + 8, 8), title, fill=(0, 0, 0))
        panel_path = args.output / f"{sample_index:02d}-{sample['sample_id']}.png"
        panel.save(panel_path)
        panels.append(panel)
        records.append(sample_record)

    sheet = Image.new(
        "RGB",
        (720 * (args.seeds + 1), 750 * len(panels)),
        "white",
    )
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 750))
    sheet.save(args.output / "generations.png")
    (args.output / "summary.json").write_text(
        json.dumps({"samples": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"samples": records}, indent=2))


if __name__ == "__main__":
    main()
