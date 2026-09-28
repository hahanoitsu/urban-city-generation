from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from urban_model.city_plan_data import CityPlanConfig, CityPlanDataset
from urban_model.plan_cell_graph import PlanCellGraphArchitect, PlanCellGraphConfig
from urban_model.plan_cell_graph_loss import canonical_cell_order
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


def to_metres(value, target_size_m):
    value = value.float()
    return [
        float((value[0] + 1.0) * 0.5 * target_size_m),
        float((value[1] + 1.0) * 0.5 * target_size_m),
    ]


def curve_points(start, end, curve):
    start, end, curve = start.float(), end.float(), curve.float()
    chord = end - start
    length = torch.linalg.vector_norm(chord).clamp_min(1e-6)
    normal = torch.stack([-chord[1], chord[0]]) / length
    values = [start]
    for index in range(curve.shape[0]):
        fraction = (index + 1) / (curve.shape[0] + 1)
        base = start + chord * fraction
        residual = curve[index] if curve.ndim == 2 else normal * curve[index] * length
        values.append(base + residual)
    values.append(end)
    return values


def target_graph(sample, tensor_config, grid_size):
    node_count = int(sample["node_count"])
    positions = sample["node_xy"][:node_count]
    order = canonical_cell_order(positions, grid_size)
    inverse = torch.empty(node_count, dtype=torch.long)
    inverse[order] = torch.arange(node_count)

    nodes = []
    ordered_positions = positions[order]
    ordered_mode = sample["node_mode"][:node_count][order]
    ordered_vertical = sample["node_vertical"][:node_count][order]
    for index in range(node_count):
        nodes.append(
            {
                "id": index,
                "position_local_m": to_metres(
                    ordered_positions[index], tensor_config.target_size_m
                ),
                "mode": "road" if int(ordered_mode[index]) == 0 else "rail",
                "vertical_mode": VERTICAL_MODES[int(ordered_vertical[index])],
            }
        )

    edges = []
    edge_count = int(sample["edge_count"])
    for source_index in range(edge_count):
        source_left = int(sample["edge_from"][source_index])
        source_right = int(sample["edge_to"][source_index])
        if source_left == source_right or source_left >= node_count or source_right >= node_count:
            continue
        left = int(inverse[source_left])
        right = int(inverse[source_right])
        shape = sample["edge_shape"][source_index]
        start = positions[source_left]
        end = positions[source_right]
        if left > right:
            left, right = right, left
            start, end = end, start
            shape = torch.flip(shape, dims=[0])

        straight = torch.stack(
            [
                start + (end - start) * (point + 1) / (tensor_config.edge_shape_points + 1)
                for point in range(tensor_config.edge_shape_points)
            ]
        )
        internal = straight + shape
        values = [start, *internal, end]
        transport_class = TRANSPORT_CLASSES[int(sample["edge_class"][source_index])]
        edges.append(
            {
                "id": len(edges),
                "from_node": left,
                "to_node": right,
                "class": transport_class,
                "mode": "road" if transport_class in ROAD_CLASSES else "rail",
                "vertical_mode": VERTICAL_MODES[int(sample["edge_vertical"][source_index])],
                "width_m": float(sample["edge_width"][source_index, 0])
                * tensor_config.width_scale_m,
                "geometry_local_m": [
                    to_metres(value, tensor_config.target_size_m) for value in values
                ],
            }
        )
    return {"nodes": nodes, "edges": edges}


def edge_candidates(output, node_count):
    pairs = torch.triu_indices(
        node_count, node_count, offset=1, device=output["edge_exists"].device
    )
    scores = output["edge_exists"][0, pairs[0], pairs[1]]
    return pairs, scores


def choose_edges_raw(output, node_count, edge_count):
    if node_count < 2 or edge_count <= 0:
        return []
    pairs, scores = edge_candidates(output, node_count)
    requested = min(edge_count, int(scores.numel()))
    chosen = torch.topk(scores, k=requested).indices
    return [(int(pairs[0, index]), int(pairs[1, index])) for index in chosen]


def choose_edges_component(output, node_count, edge_count, component_count):
    if node_count < 2 or edge_count <= 0:
        return []
    pairs, scores = edge_candidates(output, node_count)
    order = torch.argsort(scores, descending=True)
    mode = output["node_mode"][0, :node_count].argmax(dim=-1)
    parent = list(range(node_count))
    rank = [0] * node_count

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return False
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1
        return True

    target_components = max(1, min(int(component_count), node_count))
    current_components = node_count
    selected = []
    selected_set = set()

    for candidate_value in order:
        if current_components <= target_components:
            break
        candidate = int(candidate_value)
        left = int(pairs[0, candidate])
        right = int(pairs[1, candidate])
        if int(mode[left]) != int(mode[right]):
            continue
        if union(left, right):
            selected.append(candidate)
            selected_set.add(candidate)
            current_components -= 1

    for candidate_value in order:
        if len(selected) >= edge_count:
            break
        candidate = int(candidate_value)
        if candidate in selected_set:
            continue
        left = int(pairs[0, candidate])
        right = int(pairs[1, candidate])
        if int(mode[left]) != int(mode[right]):
            continue
        selected.append(candidate)
        selected_set.add(candidate)

    if len(selected) < edge_count:
        for candidate_value in order:
            if len(selected) >= edge_count:
                break
            candidate = int(candidate_value)
            if candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)

    return [(int(pairs[0, index]), int(pairs[1, index])) for index in selected[:edge_count]]


def choose_edges(output, node_count, edge_count):
    if node_count < 2 or edge_count <= 0:
        return []
    pairs = torch.triu_indices(
        node_count, node_count, offset=1, device=output["edge_exists"].device
    )
    scores = output["edge_exists"][0, pairs[0], pairs[1]]
    degree_target = output["node_degree"][0, :node_count].argmax(dim=-1)
    degree = torch.zeros(node_count, dtype=torch.long, device=scores.device)
    selected = []
    selected_set = set()

    node_order = torch.argsort(degree_target, descending=True)
    for node_value in node_order:
        node = int(node_value)
        if int(degree_target[node]) <= 0:
            continue
        incident = (pairs[0] == node) | (pairs[1] == node)
        candidates = torch.where(incident)[0]
        if candidates.numel() == 0:
            continue
        candidates = candidates[torch.argsort(scores[candidates], descending=True)]
        for candidate_value in candidates:
            candidate = int(candidate_value)
            if candidate in selected_set:
                break
            left = int(pairs[0, candidate])
            right = int(pairs[1, candidate])
            if int(degree[left]) >= max(int(degree_target[left]), 1) or int(degree[right]) >= max(
                int(degree_target[right]), 1
            ):
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            degree[left] += 1
            degree[right] += 1
            break
        if len(selected) >= edge_count:
            break

    order = torch.argsort(scores, descending=True)
    for candidate_value in order:
        if len(selected) >= edge_count:
            break
        candidate = int(candidate_value)
        if candidate in selected_set:
            continue
        left = int(pairs[0, candidate])
        right = int(pairs[1, candidate])
        left_room = int(degree[left]) < max(int(degree_target[left]), 1)
        right_room = int(degree[right]) < max(int(degree_target[right]), 1)
        if not (left_room or right_room):
            continue
        selected.append(candidate)
        selected_set.add(candidate)
        degree[left] += 1
        degree[right] += 1

    if len(selected) < edge_count:
        for candidate_value in order:
            if len(selected) >= edge_count:
                break
            candidate = int(candidate_value)
            if candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)

    return [(int(pairs[0, index]), int(pairs[1, index])) for index in selected[:edge_count]]


def generated_graph(output, sample, tensor_config, *, strategy="degree"):
    node_count = int(sample["node_count"])
    edge_count = int(round(float(sample["plan_global_raw"][1])))
    positions = output["node_xy"][0, :node_count]
    nodes = []
    for index in range(node_count):
        nodes.append(
            {
                "id": index,
                "position_local_m": to_metres(positions[index], tensor_config.target_size_m),
                "mode": "road" if int(output["node_mode"][0, index].argmax()) == 0 else "rail",
                "vertical_mode": VERTICAL_MODES[int(output["node_vertical"][0, index].argmax())],
            }
        )

    decoder = None
    if strategy == "learned_degree":
        from urban_model.graph_decode import learned_degree_edges

        pairs, decoder = learned_degree_edges(output, node_count)
    elif strategy == "compatible":
        pairs, scores = edge_candidates(output, node_count)
        modes = output["node_mode"][0, :node_count].argmax(dim=-1)
        valid = torch.where(modes[pairs[0]] == modes[pairs[1]])[0]
        chosen = valid[torch.argsort(scores[valid], descending=True)[: max(edge_count, 0)]]
        pairs = [(int(pairs[0, i]), int(pairs[1, i])) for i in chosen]
    elif strategy == "raw":
        pairs = choose_edges_raw(output, node_count, edge_count)
    elif strategy == "component":
        pairs = choose_edges_component(
            output, node_count, edge_count, int(round(float(sample["plan_global_raw"][2])))
        )
    else:
        pairs = choose_edges(output, node_count, edge_count)

    edges = []
    for left, right in pairs:
        class_index = int(output["edge_class"][0, left, right].argmax())
        if strategy in ("compatible", "learned_degree"):
            mode = int(output["node_mode"][0, left].argmax())
            logits = output["edge_class"][0, left, right]
            class_index = int(logits[:3].argmax()) if mode == 0 else int(logits[3:].argmax()) + 3
        transport_class = TRANSPORT_CLASSES[class_index]
        values = curve_points(
            positions[left], positions[right], output["edge_curve"][0, left, right]
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
                "width_m": max(0.0, float(output["edge_width"][0, left, right, 0]))
                * tensor_config.width_scale_m,
                "geometry_local_m": [
                    to_metres(value, tensor_config.target_size_m) for value in values
                ],
            }
        )
    graph = {"nodes": nodes, "edges": edges}
    if decoder is not None:
        graph["decoder"] = decoder
    return graph


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
        path = sum(math.dist(points[index], points[index + 1]) for index in range(len(points) - 1))
        ratios.append(path / max(chord, 1e-6))
        x1, y1 = points[0]
        x2, y2 = points[-1]
        dx = x2 - x1
        dy = y2 - y1
        denominator = max(math.hypot(dx, dy), 1e-6)
        deviations.append(
            max(
                (abs(dy * x - dx * y + x2 * y1 - y2 * x1) / denominator for x, y in points[1:-1]),
                default=0.0,
            )
        )

    ratios.sort()
    deviations.sort()
    return {
        "nodes": node_count,
        "edges": len(graph["edges"]),
        "components": len(components),
        "largest_component_fraction": (max(components, default=0) / max(node_count, 1)),
        "isolated_fraction": (sum(not values for values in adjacency) / max(node_count, 1)),
        "max_degree": max((len(values) for values in adjacency), default=0),
        "triangles": sum(
            len(adjacency[a] & adjacency[b])
            for a in range(node_count)
            for b in adjacency[a]
            if a < b
        )
        // 3,
        "road_edges": sum(edge["mode"] == "road" for edge in graph["edges"]),
        "rail_edges": sum(edge["mode"] == "rail" for edge in graph["edges"]),
        "path_chord_p50": (ratios[len(ratios) // 2] if ratios else 1.0),
        "path_chord_p90": (ratios[int(0.9 * (len(ratios) - 1))] if ratios else 1.0),
        "curve_deviation_p50_m": (deviations[len(deviations) // 2] if deviations else 0.0),
        "curve_deviation_p90_m": (
            deviations[int(0.9 * (len(deviations) - 1))] if deviations else 0.0
        ),
    }


def comparison_stats(target, generated):
    node_distances = [
        math.dist(
            target["nodes"][index]["position_local_m"],
            generated["nodes"][index]["position_local_m"],
        )
        for index in range(min(len(target["nodes"]), len(generated["nodes"])))
    ]
    target_pairs = {
        tuple(sorted((int(edge["from_node"]), int(edge["to_node"])))) for edge in target["edges"]
    }
    generated_pairs = {
        tuple(sorted((int(edge["from_node"]), int(edge["to_node"])))) for edge in generated["edges"]
    }
    hits = target_pairs & generated_pairs
    return {
        "node_mean_m": (sum(node_distances) / len(node_distances) if node_distances else None),
        "node_p90_m": (
            sorted(node_distances)[int(0.9 * (len(node_distances) - 1))] if node_distances else None
        ),
        "edge_pair_recall": (len(hits) / max(len(target_pairs), 1)),
        "edge_pair_precision": (len(hits) / max(len(generated_pairs), 1)),
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
        draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=(238, 238, 238))


def render(graph, sample, target_size_m, size=720):
    image = Image.new("RGB", (size, size), (250, 249, 246))
    draw = ImageDraw.Draw(image)
    render_plan_background(draw, sample, size)

    def point(value):
        return (
            int(round(value[0] / target_size_m * (size - 1))),
            int(round((1.0 - value[1] / target_size_m) * (size - 1))),
        )

    for edge in graph["edges"]:
        colour = (205, 75, 55) if edge["mode"] == "road" else (55, 125, 185)
        draw.line(
            [point(value) for value in edge["geometry_local_m"]],
            fill=colour,
            width=2,
            joint="curve",
        )
    for node in graph["nodes"]:
        x, y = point(node["position_local_m"])
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(25, 25, 25))
    return image


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tensor_config = SpatialTensorConfig(**checkpoint["tensor_config"])
    plan_config = CityPlanConfig(**checkpoint["plan_config"])
    dataset = CityPlanDataset(
        args.data,
        tensor_config=tensor_config,
        plan_config=plan_config,
        maximum_samples=checkpoint.get("maximum_samples"),
        normalization=checkpoint.get("normalization"),
    )
    model_config = PlanCellGraphConfig.from_dict(checkpoint["model_config"])
    device = torch.device("cuda")
    model = PlanCellGraphArchitect(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    indexes = list(range(len(dataset)))
    indexes.sort(
        key=lambda index: hashlib.sha1(
            str(dataset.samples[index]["sample_id"]).encode("utf-8"), usedforsecurity=False
        ).digest()
    )
    indexes = indexes[: args.samples]

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    panels = []
    for order, index in enumerate(indexes):
        sample = dataset[index]
        batch = move_sample(sample, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
        target = target_graph(sample, tensor_config, plan_config.grid_size)
        raw_generated = generated_graph(output, sample, tensor_config, strategy="raw")
        degree_generated = generated_graph(output, sample, tensor_config, strategy="degree")
        component_generated = generated_graph(output, sample, tensor_config, strategy="component")
        target_stats = graph_stats(target)
        decoder_graphs = {
            "raw": raw_generated,
            "degree": degree_generated,
            "component": component_generated,
        }
        decoder_stats = {
            name: {"graph": graph_stats(graph), "comparison": comparison_stats(target, graph)}
            for name, graph in decoder_graphs.items()
        }

        target_image = render(target, sample, tensor_config.target_size_m)
        raw_image = render(raw_generated, sample, tensor_config.target_size_m)
        degree_image = render(degree_generated, sample, tensor_config.target_size_m)
        component_image = render(component_generated, sample, tensor_config.target_size_m)
        panel = Image.new("RGB", (2880, 750), "white")
        panel.paste(target_image, (0, 30))
        panel.paste(raw_image, (720, 30))
        panel.paste(degree_image, (1440, 30))
        panel.paste(component_image, (2160, 30))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), "target", fill=(0, 0, 0))
        draw.text((728, 8), "raw top-E", fill=(0, 0, 0))
        draw.text((1448, 8), "degree decode", fill=(0, 0, 0))
        draw.text((2168, 8), "component decode", fill=(0, 0, 0))
        panel.save(args.output / f"{order:02d}-{sample['sample_id']}.png")
        panels.append(panel)

        record = {
            "sample_id": sample["sample_id"],
            "target": target_stats,
            "decoders": decoder_stats,
        }
        records.append(record)
        (args.output / f"{order:02d}-{sample['sample_id']}.json").write_text(
            json.dumps(
                {"target": target, "generated": decoder_graphs, "statistics": record}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )

    sheet = Image.new("RGB", (2880, 750 * len(panels)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 750))
    sheet.save(args.output / "generations.png")
    summary = {"samples": records}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
