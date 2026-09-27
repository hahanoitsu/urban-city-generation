from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from urban_model.spatial_anchor import SpatialAnchorArchitect, SpatialAnchorModelConfig
from urban_model.spatial_anchor_data import AnchoredSpatialWorldDataset, SpatialAnchorConfig
from urban_model.spatial_world_data import (
    ROAD_CLASSES,
    TRANSPORT_CLASSES,
    VERTICAL_MODES,
    SpatialTensorConfig,
)


def move_sample(sample, device):
    return {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


def to_metres(value):
    return [
        float((value[0] + 1.0) * 512.0),
        float((value[1] + 1.0) * 512.0),
    ]


def curve_points(start, end, curve):
    chord = end - start
    length = torch.linalg.vector_norm(chord).clamp_min(1e-6)
    normal = torch.stack([-chord[1], chord[0]]) / length
    values = [start]
    for point_index in range(curve.shape[0]):
        fraction = (point_index + 1) / (curve.shape[0] + 1)
        base = start + chord * fraction
        values.append(base + normal * curve[point_index] * length)
    values.append(end)
    return values


def target_graph(model, batch):
    positions = model.node_positions(batch["node_offset"])[0]
    flat_positions = positions.reshape(-1, 2)
    flat_mode = batch["node_mode"][0].reshape(-1)
    flat_vertical = batch["node_vertical"][0].reshape(-1)
    count = int(batch["active_count"][0])
    ids = batch["active_anchor_ids"][0, :count]
    nodes = []
    for index, anchor_id in enumerate(ids):
        anchor = int(anchor_id)
        nodes.append(
            {
                "id": index,
                "anchor_id": anchor,
                "position_local_m": to_metres(flat_positions[anchor]),
                "mode": "road" if int(flat_mode[anchor]) == 0 else "rail",
                "vertical_mode": VERTICAL_MODES[int(flat_vertical[anchor])],
                "boundary": bool(
                    batch["node_boundary"][0].reshape(-1)[anchor] > 0.5
                ),
            }
        )

    edge_count = int(batch["edge_count"][0])
    edges = []
    for index in range(edge_count):
        left = int(batch["edge_pairs"][0, index, 0])
        right = int(batch["edge_pairs"][0, index, 1])
        start = flat_positions[int(ids[left])]
        end = flat_positions[int(ids[right])]
        values = curve_points(
            start,
            end,
            batch["edge_curve"][0, index],
        )
        class_index = int(batch["edge_class"][0, index])
        transport_class = TRANSPORT_CLASSES[class_index]
        edges.append(
            {
                "id": index,
                "from_node": left,
                "to_node": right,
                "class": transport_class,
                "mode": "road" if transport_class in ROAD_CLASSES else "rail",
                "vertical_mode": VERTICAL_MODES[
                    int(batch["edge_vertical"][0, index])
                ],
                "width_m": float(batch["edge_width"][0, index, 0] * 32.0),
                "geometry_local_m": [to_metres(value) for value in values],
            }
        )
    return {"nodes": nodes, "edges": edges}


def generated_graph(model, output):
    count = int(output["active_count"][0])
    ids = output["active_anchor_ids"][0, :count]
    positions = output["active_positions"][0, :count]
    flat_mode = output["node_mode"][0].reshape(-1, 2)
    flat_vertical = output["node_vertical"][0].reshape(-1, 4)

    boundary_ids = {
        int(output["boundary_anchor_ids"][0, index])
        for index in range(int(output["boundary_count"][0]))
    }
    nodes = []
    for index, anchor_id in enumerate(ids):
        anchor = int(anchor_id)
        nodes.append(
            {
                "id": index,
                "anchor_id": anchor,
                "position_local_m": to_metres(positions[index]),
                "mode": "road" if int(flat_mode[anchor].argmax()) == 0 else "rail",
                "vertical_mode": VERTICAL_MODES[
                    int(flat_vertical[anchor].argmax())
                ],
                "boundary": anchor in boundary_ids,
            }
        )

    edges = []
    if count >= 2:
        pairs = torch.triu_indices(
            count,
            count,
            offset=1,
            device=output["edge_exists"].device,
        )
        scores = output["edge_exists"][0, pairs[0], pairs[1]]
        requested = min(
            int(output["predicted_edge_count"][0]),
            int(scores.numel()),
        )
        degree_target = output["node_degree"][0, :count].argmax(dim=-1)
        degree = torch.zeros(
            count,
            dtype=torch.long,
            device=scores.device,
        )
        order = torch.argsort(scores, descending=True)
        chosen = []
        chosen_set = set()

        for strict in (True, False):
            for pair_index in order:
                if len(chosen) >= requested:
                    break
                index = int(pair_index)
                if index in chosen_set:
                    continue
                left = int(pairs[0, index])
                right = int(pairs[1, index])
                left_need = int(degree[left]) < int(degree_target[left])
                right_need = int(degree[right]) < int(degree_target[right])
                if strict and not (left_need and right_need):
                    continue
                if not strict and not (left_need or right_need):
                    continue
                chosen.append(index)
                chosen_set.add(index)
                degree[left] += 1
                degree[right] += 1

        if len(chosen) < requested:
            for pair_index in order:
                if len(chosen) >= requested:
                    break
                index = int(pair_index)
                if index in chosen_set:
                    continue
                chosen.append(index)
                chosen_set.add(index)
                left = int(pairs[0, index])
                right = int(pairs[1, index])
                degree[left] += 1
                degree[right] += 1

        for pair_index in chosen:
            left = int(pairs[0, pair_index])
            right = int(pairs[1, pair_index])
            class_index = int(
                output["edge_class"][0, left, right].argmax()
            )
            transport_class = TRANSPORT_CLASSES[class_index]
            start = positions[left]
            end = positions[right]
            values = curve_points(
                start,
                end,
                output["edge_curve"][0, left, right],
            )
            edges.append(
                {
                    "id": len(edges),
                    "from_node": left,
                    "to_node": right,
                    "class": transport_class,
                    "mode": "road" if transport_class in ROAD_CLASSES else "rail",
                    "vertical_mode": VERTICAL_MODES[
                        int(output["edge_vertical"][0, left, right].argmax())
                    ],
                    "width_m": max(
                        0.0,
                        float(
                            output["edge_width"][0, left, right, 0]
                            * 32.0
                        ),
                    ),
                    "geometry_local_m": [
                        to_metres(value)
                        for value in values
                    ],
                }
            )
    return {"nodes": nodes, "edges": edges}


def graph_stats(graph):
    nodes = len(graph["nodes"])
    adjacency = [set() for _ in range(nodes)]
    for edge in graph["edges"]:
        left = int(edge["from_node"])
        right = int(edge["to_node"])
        if 0 <= left < nodes and 0 <= right < nodes and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)

    isolated = sum(not values for values in adjacency)
    boundary_nodes = [
        index
        for index, node in enumerate(graph["nodes"])
        if bool(node.get("boundary", False))
    ]
    connected_boundary = sum(bool(adjacency[index]) for index in boundary_nodes)
    seen = set()
    components = []
    for start in range(nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(size)

    return {
        "nodes": nodes,
        "edges": len(graph["edges"]),
        "isolated_nodes": isolated,
        "isolated_fraction": isolated / max(nodes, 1),
        "components": len(components),
        "largest_component_fraction": max(components, default=0) / max(nodes, 1),
        "road_edges": sum(edge["mode"] == "road" for edge in graph["edges"]),
        "rail_edges": sum(edge["mode"] == "rail" for edge in graph["edges"]),
        "boundary_nodes": len(boundary_nodes),
        "connected_boundary_nodes": connected_boundary,
        "boundary_connected_fraction": (
            connected_boundary / len(boundary_nodes)
            if boundary_nodes
            else 1.0
        ),
    }


def render(graph, sample, tensor_config, size=720):
    image = Image.new("RGB", (size, size), (247, 246, 242))
    draw = ImageDraw.Draw(image)
    local_size = tensor_config.local_vector_size_m
    target_size = tensor_config.target_size_m
    margin = (local_size - target_size) / 2.0

    def point(value):
        x = float(value[0])
        y = float(value[1])
        return (
            int(round((x + margin) / local_size * (size - 1))),
            int(
                round(
                    (1.0 - (y + margin) / local_size)
                    * (size - 1)
                )
            ),
        )

    padding = sample["context_line_padding"]
    points = sample["context_line_points"]
    modes = sample["context_line_mode"]
    for index in range(points.shape[0]):
        if bool(padding[index]):
            continue
        values = (
            points[index] * (local_size / 2.0)
            + target_size / 2.0
        )
        projected = [point(value) for value in values]
        colour = (
            (194, 194, 194)
            if int(modes[index]) == 0
            else (170, 198, 218)
        )
        draw.line(projected, fill=colour, width=1)

    target_left, target_bottom = point([0.0, 0.0])
    target_right, target_top = point([target_size, target_size])
    draw.rectangle(
        [target_left, target_top, target_right, target_bottom],
        outline=(80, 80, 80),
        width=2,
    )

    port_padding = sample["port_padding"]
    ports = sample["ports"]
    for index in range(ports.shape[0]):
        if bool(port_padding[index]):
            continue
        x = float((ports[index, 0] + 1.0) * 0.5 * target_size)
        y = float((ports[index, 1] + 1.0) * 0.5 * target_size)
        px, py = point([x, y])
        draw.ellipse(
            [px - 3, py - 3, px + 3, py + 3],
            fill=(20, 20, 20),
        )

    for edge in graph["edges"]:
        values = [point(value) for value in edge["geometry_local_m"]]
        colour = (205, 75, 55) if edge["mode"] == "road" else (55, 125, 185)
        draw.line(values, fill=colour, width=2, joint="curve")

    for node in graph["nodes"]:
        x, y = point(node["position_local_m"])
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(25, 25, 25))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--drop-controls", action="store_true")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tensor_config = SpatialTensorConfig(**checkpoint["tensor_config"])
    anchor_config = SpatialAnchorConfig(**checkpoint["anchor_config"])
    model_config = SpatialAnchorModelConfig.from_dict(checkpoint["model_config"])
    dataset = AnchoredSpatialWorldDataset(
        args.data,
        tensor_config=tensor_config,
        anchor_config=anchor_config,
    )

    candidates = [
        index
        for index, sample in enumerate(dataset.samples)
        if sample["split"] == args.split
    ]
    candidates.sort(
        key=lambda index: hashlib.sha1(
            str(dataset.samples[index]["sample_id"]).encode("utf-8"),
            usedforsecurity=False,
        ).digest()
    )
    indexes = candidates[: args.samples]
    if not indexes:
        raise RuntimeError("No held-out anchored samples found")

    device = torch.device("cuda")
    model = SpatialAnchorArchitect(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    panels = []
    for sample_index, dataset_index in enumerate(indexes):
        sample = dataset[dataset_index]
        batch = move_sample(sample, device)
        if args.drop_controls:
            batch["controls"] = torch.zeros_like(batch["controls"])
        target = target_graph(model, batch)
        images = [render(target, sample, tensor_config)]
        record = {
            "sample_id": sample["sample_id"],
            "target": graph_stats(target),
            "generations": [],
        }

        for seed_index in range(args.seeds):
            seed = 1000 + sample_index * 100 + seed_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            output = model.generate(batch, temperature=args.temperature)
            generated = generated_graph(model, output)
            images.append(render(generated, sample, tensor_config))
            stats = graph_stats(generated)
            stats["seed"] = seed_index
            record["generations"].append(stats)
            (args.output / f"{sample_index:02d}-{sample['sample_id']}-seed{seed_index}.json").write_text(
                json.dumps(generated, indent=2) + "\n",
                encoding="utf-8",
            )

        panel = Image.new("RGB", (720 * len(images), 750), "white")
        draw = ImageDraw.Draw(panel)
        for image_index, image in enumerate(images):
            panel.paste(image, (image_index * 720, 30))
            title = "target" if image_index == 0 else f"seed {image_index - 1}"
            draw.text((image_index * 720 + 8, 8), title, fill=(0, 0, 0))
        panel.save(args.output / f"{sample_index:02d}-{sample['sample_id']}.png")
        panels.append(panel)
        records.append(record)

    sheet = Image.new(
        "RGB",
        (720 * (args.seeds + 1), 750 * len(panels)),
        "white",
    )
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 750))
    sheet.save(args.output / "generations.png")
    summary = {"samples": records}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
