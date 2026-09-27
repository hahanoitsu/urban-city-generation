from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from urban_model.city_plan_data import CityPlanConfig, CityPlanDataset
from urban_model.plan_set_graph import (
    PlanSetGraphArchitect,
    PlanSetGraphConfig,
)
from urban_model.spatial_world_data import (
    ROAD_CLASSES,
    TRANSPORT_CLASSES,
    VERTICAL_MODES,
    SpatialTensorConfig,
)


def move_sample(sample, device):
    return {
        key: value.unsqueeze(0).to(device)
        if torch.is_tensor(value)
        else value
        for key, value in sample.items()
    }


def to_metres(value, target_size_m):
    return [
        float((value[0] + 1.0) * 0.5 * target_size_m),
        float((value[1] + 1.0) * 0.5 * target_size_m),
    ]


def curve_points(start, end, curve):
    chord = end - start
    length = torch.linalg.vector_norm(chord).clamp_min(1e-6)
    normal = torch.stack([-chord[1], chord[0]]) / length
    values = [start]
    for index in range(curve.shape[0]):
        fraction = (index + 1) / (curve.shape[0] + 1)
        base = start + chord * fraction
        values.append(base + normal * curve[index] * length)
    values.append(end)
    return values


def target_graph(sample, tensor_config):
    node_count = int(sample["node_count"])
    edge_count = int(sample["edge_count"])
    positions = sample["node_xy"][:node_count]
    nodes = []
    for index in range(node_count):
        nodes.append(
            {
                "id": index,
                "position_norm": [
                    float(positions[index, 0]),
                    float(positions[index, 1]),
                ],
                "position_local_m": to_metres(
                    positions[index],
                    tensor_config.target_size_m,
                ),
                "mode": "road"
                if int(sample["node_mode"][index]) == 0
                else "rail",
                "vertical_mode": VERTICAL_MODES[
                    int(sample["node_vertical"][index])
                ],
            }
        )

    edges = []
    for index in range(edge_count):
        left = int(sample["edge_from"][index])
        right = int(sample["edge_to"][index])
        if left == right or left >= node_count or right >= node_count:
            continue
        start = positions[left]
        end = positions[right]
        straight = torch.stack(
            [
                start
                + (end - start)
                * (point + 1)
                / (tensor_config.edge_shape_points + 1)
                for point in range(tensor_config.edge_shape_points)
            ]
        )
        internal = straight + sample["edge_shape"][index]
        values = [start, *internal, end]
        transport_class = TRANSPORT_CLASSES[
            int(sample["edge_class"][index])
        ]
        edges.append(
            {
                "id": len(edges),
                "from_node": left,
                "to_node": right,
                "class": transport_class,
                "mode": "road"
                if transport_class in ROAD_CLASSES
                else "rail",
                "vertical_mode": VERTICAL_MODES[
                    int(sample["edge_vertical"][index])
                ],
                "geometry_local_m": [
                    to_metres(value, tensor_config.target_size_m)
                    for value in values
                ],
            }
        )
    return {"nodes": nodes, "edges": edges}


def generated_graph(output, sample, tensor_config):
    node_count = max(
        1,
        min(
            int(round(float(sample["plan_global_raw"][0]))),
            output["node_xy"].shape[1],
        ),
    )
    edge_count = max(
        0,
        int(round(float(sample["plan_global_raw"][1]))),
    )
    selected = torch.arange(
        node_count,
        device=output["node_xy"].device,
    )
    positions = output["node_xy"][
        0,
        :node_count,
    ]
    nodes = []
    for index, query in enumerate(selected):
        query_index = int(query)
        nodes.append(
            {
                "id": index,
                "query_id": query_index,
                "position_norm": [
                    float(positions[index, 0]),
                    float(positions[index, 1]),
                ],
                "position_local_m": to_metres(
                    positions[index],
                    tensor_config.target_size_m,
                ),
                "mode": "road"
                if int(
                    output["node_mode"][0, query_index].argmax()
                )
                == 0
                else "rail",
                "vertical_mode": VERTICAL_MODES[
                    int(
                        output["node_vertical"][
                            0,
                            query_index,
                        ].argmax()
                    )
                ],
            }
        )

    edges = []
    if node_count >= 2 and edge_count > 0:
        pairs = torch.triu_indices(
            node_count,
            node_count,
            offset=1,
            device=positions.device,
        )
        query_left = selected[pairs[0]]
        query_right = selected[pairs[1]]
        scores = output["edge_exists"][
            0,
            query_left,
            query_right,
        ]
        requested = min(
            edge_count,
            int(scores.numel()),
        )
        chosen = torch.topk(
            scores,
            k=requested,
        ).indices
        for pair_index in chosen:
            pair_index = int(pair_index)
            left = int(pairs[0, pair_index])
            right = int(pairs[1, pair_index])
            q_left = int(query_left[pair_index])
            q_right = int(query_right[pair_index])
            class_index = int(
                output["edge_class"][
                    0,
                    q_left,
                    q_right,
                ].argmax()
            )
            transport_class = TRANSPORT_CLASSES[class_index]
            values = curve_points(
                positions[left],
                positions[right],
                output["edge_curve"][
                    0,
                    q_left,
                    q_right,
                ],
            )
            edges.append(
                {
                    "id": len(edges),
                    "from_node": left,
                    "to_node": right,
                    "class": transport_class,
                    "mode": "road"
                    if transport_class in ROAD_CLASSES
                    else "rail",
                    "vertical_mode": VERTICAL_MODES[
                        int(
                            output["edge_vertical"][
                                0,
                                q_left,
                                q_right,
                            ].argmax()
                        )
                    ],
                    "geometry_local_m": [
                        to_metres(
                            value,
                            tensor_config.target_size_m,
                        )
                        for value in values
                    ],
                }
            )
    return {"nodes": nodes, "edges": edges}


def graph_stats(graph):
    node_count = len(graph["nodes"])
    adjacency = [set() for _ in range(node_count)]
    for edge in graph["edges"]:
        left = int(edge["from_node"])
        right = int(edge["to_node"])
        if 0 <= left < node_count and 0 <= right < node_count:
            adjacency[left].add(right)
            adjacency[right].add(left)

    seen = set()
    components = []
    for start in range(node_count):
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

    ratios = []
    deviations = []
    for edge in graph["edges"]:
        points = edge["geometry_local_m"]
        chord = math.dist(points[0], points[-1])
        path = sum(
            math.dist(points[index], points[index + 1])
            for index in range(len(points) - 1)
        )
        ratios.append(path / max(chord, 1e-6))
        x1, y1 = points[0]
        x2, y2 = points[-1]
        dx = x2 - x1
        dy = y2 - y1
        denominator = max(math.hypot(dx, dy), 1e-6)
        deviations.append(
            max(
                (
                    abs(
                        dy * x
                        - dx * y
                        + x2 * y1
                        - y2 * x1
                    )
                    / denominator
                    for x, y in points[1:-1]
                ),
                default=0.0,
            )
        )

    ratios.sort()
    deviations.sort()
    return {
        "nodes": node_count,
        "edges": len(graph["edges"]),
        "components": len(components),
        "largest_component_fraction": (
            max(components, default=0) / max(node_count, 1)
        ),
        "isolated_fraction": (
            sum(not values for values in adjacency) / max(node_count, 1)
        ),
        "road_edges": sum(
            edge["mode"] == "road" for edge in graph["edges"]
        ),
        "rail_edges": sum(
            edge["mode"] == "rail" for edge in graph["edges"]
        ),
        "path_chord_p50": (
            ratios[len(ratios) // 2] if ratios else 1.0
        ),
        "path_chord_p90": (
            ratios[int(0.9 * (len(ratios) - 1))]
            if ratios
            else 1.0
        ),
        "curve_deviation_p50_m": (
            deviations[len(deviations) // 2]
            if deviations
            else 0.0
        ),
        "curve_deviation_p90_m": (
            deviations[int(0.9 * (len(deviations) - 1))]
            if deviations
            else 0.0
        ),
    }


def greedy_match(target, generated):
    if not target["nodes"] or not generated["nodes"]:
        return {}
    target_xy = torch.tensor(
        [node["position_norm"] for node in target["nodes"]]
    )
    generated_xy = torch.tensor(
        [node["position_norm"] for node in generated["nodes"]]
    )
    distances = torch.cdist(generated_xy, target_xy)
    proposals = []
    for generated_index in range(generated_xy.shape[0]):
        for target_index in range(target_xy.shape[0]):
            proposals.append(
                (
                    float(distances[generated_index, target_index]),
                    generated_index,
                    target_index,
                )
            )
    proposals.sort(key=lambda value: value[0])
    result = {}
    used_targets = set()
    for _distance, generated_index, target_index in proposals:
        if generated_index in result or target_index in used_targets:
            continue
        result[generated_index] = target_index
        used_targets.add(target_index)
        if len(result) == min(
            len(target["nodes"]),
            len(generated["nodes"]),
        ):
            break
    return result


def comparison_stats(target, generated, sample, plan_config):
    mapping = greedy_match(target, generated)
    distances = []
    for generated_index, target_index in mapping.items():
        distances.append(
            math.dist(
                generated["nodes"][generated_index]["position_local_m"],
                target["nodes"][target_index]["position_local_m"],
            )
        )

    target_pairs = {
        tuple(sorted((int(edge["from_node"]), int(edge["to_node"]))))
        for edge in target["edges"]
    }
    generated_pairs = set()
    for edge in generated["edges"]:
        left = mapping.get(int(edge["from_node"]))
        right = mapping.get(int(edge["to_node"]))
        if left is None or right is None or left == right:
            continue
        generated_pairs.add(tuple(sorted((left, right))))
    hits = target_pairs & generated_pairs

    grid = plan_config.grid_size
    target_counts = sample["plan_counts"][:, 0].reshape(grid, grid)
    generated_counts = torch.zeros(grid, grid)
    for node in generated["nodes"]:
        x, y = node["position_norm"]
        column = min(grid - 1, max(0, int((x + 1.0) * 0.5 * grid)))
        row = min(grid - 1, max(0, int((y + 1.0) * 0.5 * grid)))
        generated_counts[row, column] += 1.0
    target_presence = target_counts > 0
    generated_presence = generated_counts > 0
    intersection = int((target_presence & generated_presence).sum())
    union = int((target_presence | generated_presence).sum())

    return {
        "node_match_mean_m": (
            sum(distances) / len(distances) if distances else None
        ),
        "node_match_p90_m": (
            sorted(distances)[int(0.9 * (len(distances) - 1))]
            if distances
            else None
        ),
        "node_cell_occupancy_iou": intersection / max(union, 1),
        "node_cell_count_mae": float(
            (generated_counts - target_counts).abs().mean()
        ),
        "edge_pair_recall": len(hits) / max(len(target_pairs), 1),
        "edge_pair_precision": len(hits) / max(len(generated_pairs), 1),
    }


def render_plan_background(draw, sample, size):
    grid = int(round(sample["plan_presence"].shape[0] ** 0.5))
    cell = size / grid
    corridor = sample["plan_presence"][:, 3:7].any(dim=-1)
    for index in range(corridor.shape[0]):
        if not bool(corridor[index]):
            continue
        row = index // grid
        column = index % grid
        x0 = column * cell
        y0 = (grid - 1 - row) * cell
        draw.rectangle(
            [x0, y0, x0 + cell, y0 + cell],
            fill=(238, 238, 238),
        )


def render(graph, sample, target_size_m, size=720):
    image = Image.new("RGB", (size, size), (250, 249, 246))
    draw = ImageDraw.Draw(image)
    render_plan_background(draw, sample, size)

    def point(value):
        return (
            int(round(value[0] / target_size_m * (size - 1))),
            int(
                round(
                    (1.0 - value[1] / target_size_m)
                    * (size - 1)
                )
            ),
        )

    for edge in graph["edges"]:
        colour = (
            (205, 75, 55)
            if edge["mode"] == "road"
            else (55, 125, 185)
        )
        draw.line(
            [point(value) for value in edge["geometry_local_m"]],
            fill=colour,
            width=2,
            joint="curve",
        )
    for node in graph["nodes"]:
        x, y = point(node["position_local_m"])
        draw.ellipse(
            [x - 2, y - 2, x + 2, y + 2],
            fill=(25, 25, 25),
        )
    return image


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    tensor_config = SpatialTensorConfig(
        **checkpoint["tensor_config"]
    )
    plan_config = CityPlanConfig(
        **checkpoint["plan_config"]
    )
    dataset = CityPlanDataset(
        args.data,
        tensor_config=tensor_config,
        plan_config=plan_config,
        maximum_samples=checkpoint.get("maximum_samples"),
    )
    model_config = PlanSetGraphConfig.from_dict(
        checkpoint["model_config"]
    )
    device = torch.device("cuda")
    model = PlanSetGraphArchitect(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    indexes = list(range(len(dataset)))
    indexes.sort(
        key=lambda index: hashlib.sha1(
            str(dataset.samples[index]["sample_id"]).encode("utf-8"),
            usedforsecurity=False,
        ).digest()
    )
    indexes = indexes[: args.samples]

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    panels = []
    for order, index in enumerate(indexes):
        sample = dataset[index]
        batch = move_sample(sample, device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            output = model(batch)
        target = target_graph(sample, tensor_config)
        generated = generated_graph(output, sample, tensor_config)
        target_stats = graph_stats(target)
        generated_stats = graph_stats(generated)
        comparison = comparison_stats(
            target,
            generated,
            sample,
            plan_config,
        )

        target_image = render(
            target,
            sample,
            tensor_config.target_size_m,
        )
        generated_image = render(
            generated,
            sample,
            tensor_config.target_size_m,
        )
        panel = Image.new("RGB", (1440, 750), "white")
        panel.paste(target_image, (0, 30))
        panel.paste(generated_image, (720, 30))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), "target", fill=(0, 0, 0))
        draw.text((728, 8), "generated", fill=(0, 0, 0))
        panel.save(
            args.output / f"{order:02d}-{sample['sample_id']}.png"
        )
        panels.append(panel)

        record = {
            "sample_id": sample["sample_id"],
            "target": target_stats,
            "generated": generated_stats,
            "comparison": comparison,
        }
        records.append(record)
        (
            args.output / f"{order:02d}-{sample['sample_id']}.json"
        ).write_text(
            json.dumps(
                {
                    "target": target,
                    "generated": generated,
                    "statistics": record,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    sheet = Image.new(
        "RGB",
        (1440, 750 * len(panels)),
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
