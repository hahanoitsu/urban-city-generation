from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from urban_model.frontier_architect import FrontierArchitect, FrontierArchitectConfig
from urban_model.frontier_data import (
    OP_BOS,
    OP_CLOSE,
    OP_EOS,
    OP_GROW,
    OP_LINK,
    OP_ROOT,
    FrontierProgramConfig,
    FrontierProgramDataset,
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


def categorical(logits, temperature):
    if temperature <= 0:
        return int(logits.argmax().item())
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probabilities, 1).item())


def normal(mean, logstd, temperature):
    if temperature <= 0:
        return mean
    noise = torch.randn_like(mean)
    return mean + torch.exp(logstd) * noise * temperature


def curve_points(start, end, curve):
    chord = end - start
    length = torch.linalg.vector_norm(chord).clamp_min(1e-6)
    normal = torch.stack([-chord[1], chord[0]]) / length
    values = [start]
    for index in range(curve.shape[0]):
        fraction = (index + 1) / (curve.shape[0] + 1)
        base = start + chord * fraction
        values.append(base + normal * curve[index])
    values.append(end)
    return values


def to_metres(value, target_size):
    return [
        float((value[0] + 1.0) * 0.5 * target_size),
        float((value[1] + 1.0) * 0.5 * target_size),
    ]


def target_graph(sample, tensor_config):
    node_count = int(sample["node_count"])
    edge_count = int(sample["edge_count"])
    positions = sample["node_xy"][:node_count]
    nodes = []
    for index in range(node_count):
        nodes.append(
            {
                "id": index,
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
                "boundary": bool(sample["node_boundary"][index] > 0.5),
            }
        )

    edges = []
    for index in range(edge_count):
        left = int(sample["edge_from"][index])
        right = int(sample["edge_to"][index])
        start = positions[left]
        end = positions[right]
        straight = torch.stack(
            [
                start + (end - start) * (point + 1)
                / (tensor_config.edge_shape_points + 1)
                for point in range(tensor_config.edge_shape_points)
            ]
        )
        internal = straight + sample["edge_shape"][index]
        values = [start, *internal, end]
        class_index = int(sample["edge_class"][index])
        transport_class = TRANSPORT_CLASSES[class_index]
        edges.append(
            {
                "id": index,
                "from_node": left,
                "to_node": right,
                "class": transport_class,
                "mode": "road"
                if transport_class in ROAD_CLASSES
                else "rail",
                "vertical_mode": VERTICAL_MODES[
                    int(sample["edge_vertical"][index])
                ],
                "width_m": float(
                    sample["edge_width"][index, 0]
                    * tensor_config.width_scale_m
                ),
                "geometry_local_m": [
                    to_metres(value, tensor_config.target_size_m)
                    for value in values
                ],
            }
        )
    return {"nodes": nodes, "edges": edges}


def empty_program(batch):
    names = (
        "program_op",
        "program_xy",
        "program_node_mode",
        "program_node_vertical",
        "program_node_boundary",
        "program_edge_class",
        "program_edge_vertical",
        "program_edge_width",
        "program_curve",
        "program_pointer",
        "program_active_node",
        "program_active_xy",
    )
    for name in names:
        batch[name] = torch.zeros_like(batch[name])
    batch["program_active_node"].fill_(-1)
    batch["program_op"][0, 0] = OP_BOS
    batch["program_length"][0] = 2


def write_event(
    batch,
    index,
    *,
    op,
    xy=None,
    node_mode=0,
    node_vertical=0,
    node_boundary=0.0,
    edge_class=0,
    edge_vertical=0,
    width=0.0,
    curve=None,
    pointer=0,
    active=-1,
    active_xy=None,
):
    batch["program_op"][0, index] = op
    if xy is not None:
        batch["program_xy"][0, index] = xy
    batch["program_node_mode"][0, index] = node_mode
    batch["program_node_vertical"][0, index] = node_vertical
    batch["program_node_boundary"][0, index] = node_boundary
    batch["program_edge_class"][0, index] = edge_class
    batch["program_edge_vertical"][0, index] = edge_vertical
    batch["program_edge_width"][0, index, 0] = width
    if curve is not None:
        batch["program_curve"][0, index] = curve
    batch["program_pointer"][0, index] = pointer
    batch["program_active_node"][0, index] = active
    if active_xy is not None:
        batch["program_active_xy"][0, index] = active_xy


def legal_op_logits(logits, active, node_count, edge_count, max_nodes, max_edges):
    result = torch.full_like(logits, float("-inf"))
    if active is None:
        if node_count < max_nodes:
            result[OP_ROOT] = logits[OP_ROOT]
        result[OP_EOS] = logits[OP_EOS]
    else:
        result[OP_CLOSE] = logits[OP_CLOSE]
        if node_count < max_nodes and edge_count < max_edges:
            result[OP_GROW] = logits[OP_GROW]
        if node_count >= 2 and edge_count < max_edges:
            result[OP_LINK] = logits[OP_LINK]
    return result


def rollout(model, sample, tensor_config, program_config, device, temperature):
    batch = move_sample(sample, device)
    empty_program(batch)
    memory, memory_padding = model.encode_context(batch)

    nodes = []
    edges = []
    queue = []
    active = None
    edge_pairs = set()
    roots = 0
    clamped_nodes = 0
    prefix = 1

    while prefix < program_config.max_steps:
        batch["program_length"][0] = prefix + 1
        output = model.decode_program(
            batch,
            memory,
            memory_padding,
            input_length=prefix,
        )
        last = prefix - 1
        op_logits = legal_op_logits(
            output["op"][0, last],
            active,
            len(nodes),
            len(edges),
            program_config.max_nodes,
            tensor_config.max_edges,
        )
        op = categorical(op_logits, temperature)

        if op == OP_EOS:
            write_event(batch, prefix, op=OP_EOS)
            prefix += 1
            break

        if op == OP_ROOT:
            value = normal(
                output["xy_mean"][0, last],
                output["xy_logstd"][0, last],
                temperature,
            ).clamp(-1.0, 1.0)
            node_id = len(nodes)
            node_mode = categorical(
                output["node_mode"][0, last],
                temperature,
            )
            node_vertical = categorical(
                output["node_vertical"][0, last],
                temperature,
            )
            boundary = float(
                torch.sigmoid(
                    output["node_boundary"][0, last]
                ).item()
            )
            nodes.append(
                {
                    "id": node_id,
                    "position": value.detach().clone(),
                    "mode": node_mode,
                    "vertical": node_vertical,
                    "boundary": boundary,
                }
            )
            queue = [node_id]
            active = node_id
            roots += 1
            write_event(
                batch,
                prefix,
                op=OP_ROOT,
                xy=value,
                node_mode=node_mode,
                node_vertical=node_vertical,
                node_boundary=boundary,
                active=node_id,
                active_xy=value,
            )

        elif op == OP_GROW:
            delta = normal(
                output["xy_mean"][0, last],
                output["xy_logstd"][0, last],
                temperature,
            ).clamp(-1.0, 1.0)
            raw_position = nodes[active]["position"] + delta * 2.0
            position = raw_position.clamp(-1.0, 1.0)
            if not torch.equal(raw_position, position):
                clamped_nodes += 1
            child = len(nodes)
            node_mode = categorical(
                output["node_mode"][0, last],
                temperature,
            )
            node_vertical = categorical(
                output["node_vertical"][0, last],
                temperature,
            )
            boundary = float(
                torch.sigmoid(
                    output["node_boundary"][0, last]
                ).item()
            )
            edge_class = categorical(
                output["edge_class"][0, last],
                temperature,
            )
            edge_vertical = categorical(
                output["edge_vertical"][0, last],
                temperature,
            )
            width = float(
                normal(
                    output["width_mean"][0, last],
                    output["width_logstd"][0, last],
                    temperature,
                )[0].clamp_min(0.0).item()
            )
            curve = normal(
                output["curve_mean"][0, last],
                output["curve_logstd"][0, last],
                temperature,
            ).clamp(-1.5, 1.5)
            nodes.append(
                {
                    "id": child,
                    "position": position.detach().clone(),
                    "mode": node_mode,
                    "vertical": node_vertical,
                    "boundary": boundary,
                }
            )
            edges.append(
                {
                    "from_node": active,
                    "to_node": child,
                    "class": edge_class,
                    "vertical": edge_vertical,
                    "width": width,
                    "curve": curve.detach().clone(),
                }
            )
            edge_pairs.add(tuple(sorted((active, child))))
            queue.append(child)
            write_event(
                batch,
                prefix,
                op=OP_GROW,
                xy=delta,
                node_mode=node_mode,
                node_vertical=node_vertical,
                node_boundary=boundary,
                edge_class=edge_class,
                edge_vertical=edge_vertical,
                width=width,
                curve=curve,
                active=active,
                active_xy=nodes[active]["position"],
            )

        elif op == OP_LINK:
            logits = output["pointer"][0, last].clone()
            valid = torch.arange(
                program_config.max_nodes,
                device=logits.device,
            ) < len(nodes)
            valid[active] = False
            for left, right in edge_pairs:
                if left == active:
                    valid[right] = False
                elif right == active:
                    valid[left] = False
            logits[~valid] = float("-inf")
            if not bool(valid.any()):
                op = OP_CLOSE
            else:
                pointer = categorical(logits, temperature)
                edge_class = categorical(
                    output["edge_class"][0, last],
                    temperature,
                )
                edge_vertical = categorical(
                    output["edge_vertical"][0, last],
                    temperature,
                )
                width = float(
                    normal(
                        output["width_mean"][0, last],
                        output["width_logstd"][0, last],
                        temperature,
                    )[0].clamp_min(0.0).item()
                )
                curve = normal(
                    output["curve_mean"][0, last],
                    output["curve_logstd"][0, last],
                    temperature,
                ).clamp(-1.5, 1.5)
                edges.append(
                    {
                        "from_node": active,
                        "to_node": pointer,
                        "class": edge_class,
                        "vertical": edge_vertical,
                        "width": width,
                        "curve": curve.detach().clone(),
                    }
                )
                edge_pairs.add(tuple(sorted((active, pointer))))
                write_event(
                    batch,
                    prefix,
                    op=OP_LINK,
                    edge_class=edge_class,
                    edge_vertical=edge_vertical,
                    width=width,
                    curve=curve,
                    pointer=pointer,
                    active=active,
                    active_xy=nodes[active]["position"],
                )

        if op == OP_CLOSE:
            if queue:
                queue.pop(0)
            active = queue[0] if queue else None
            write_event(
                batch,
                prefix,
                op=OP_CLOSE,
                active=active if active is not None else -1,
                active_xy=(
                    nodes[active]["position"]
                    if active is not None
                    else None
                ),
            )

        prefix += 1

    graph_nodes = []
    for node in nodes:
        graph_nodes.append(
            {
                "id": node["id"],
                "position_local_m": to_metres(
                    node["position"],
                    tensor_config.target_size_m,
                ),
                "mode": "road" if node["mode"] == 0 else "rail",
                "vertical_mode": VERTICAL_MODES[node["vertical"]],
                "boundary_score": node["boundary"],
            }
        )

    graph_edges = []
    for index, edge in enumerate(edges):
        transport_class = TRANSPORT_CLASSES[edge["class"]]
        values = curve_points(
            nodes[edge["from_node"]]["position"],
            nodes[edge["to_node"]]["position"],
            edge["curve"],
        )
        graph_edges.append(
            {
                "id": index,
                "from_node": edge["from_node"],
                "to_node": edge["to_node"],
                "class": transport_class,
                "mode": "road"
                if transport_class in ROAD_CLASSES
                else "rail",
                "vertical_mode": VERTICAL_MODES[edge["vertical"]],
                "width_m": edge["width"] * tensor_config.width_scale_m,
                "geometry_local_m": [
                    to_metres(value, tensor_config.target_size_m)
                    for value in values
                ],
            }
        )

    return {
        "nodes": graph_nodes,
        "edges": graph_edges,
        "program_steps": prefix,
        "roots": roots,
        "clamped_nodes": clamped_nodes,
    }


def graph_stats(graph, sample, tensor_config):
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

    ports = sample["ports"][~sample["port_padding"]]
    satisfied = 0
    if len(ports) and node_count:
        generated = torch.tensor(
            [
                [
                    node["position_local_m"][0]
                    / tensor_config.target_size_m
                    * 2.0
                    - 1.0,
                    node["position_local_m"][1]
                    / tensor_config.target_size_m
                    * 2.0
                    - 1.0,
                ]
                for node in graph["nodes"]
            ],
            dtype=torch.float32,
        )
        for port in ports:
            distance = torch.linalg.vector_norm(
                generated - port[:2],
                dim=-1,
            ).min()
            metres = float(
                distance
                * tensor_config.target_size_m
                / 2.0
            )
            if metres <= 40.0:
                satisfied += 1

    isolated = sum(not values for values in adjacency)
    return {
        "nodes": node_count,
        "edges": len(graph["edges"]),
        "components": len(components),
        "largest_component_fraction": (
            max(components, default=0) / max(node_count, 1)
        ),
        "isolated_nodes": isolated,
        "isolated_fraction": isolated / max(node_count, 1),
        "road_edges": sum(
            edge["mode"] == "road"
            for edge in graph["edges"]
        ),
        "rail_edges": sum(
            edge["mode"] == "rail"
            for edge in graph["edges"]
        ),
        "ports": int(len(ports)),
        "port_satisfaction_fraction": (
            satisfied / len(ports)
            if len(ports)
            else 1.0
        ),
        "roots": int(graph.get("roots", 0)),
        "program_steps": int(graph.get("program_steps", 0)),
        "clamped_nodes": int(graph.get("clamped_nodes", 0)),
    }


def render(graph, sample, config, size=720):
    image = Image.new("RGB", (size, size), (247, 246, 242))
    draw = ImageDraw.Draw(image)
    local_size = config.local_vector_size_m
    target_size = config.target_size_m
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

    for index in range(sample["context_line_points"].shape[0]):
        if bool(sample["context_line_padding"][index]):
            continue
        values = (
            sample["context_line_points"][index]
            * (local_size / 2.0)
            + target_size / 2.0
        )
        colour = (
            (195, 195, 195)
            if int(sample["context_line_mode"][index]) == 0
            else (170, 198, 218)
        )
        draw.line(
            [point(value) for value in values],
            fill=colour,
            width=1,
        )

    left, bottom = point([0.0, 0.0])
    right, top = point([target_size, target_size])
    draw.rectangle(
        [left, top, right, bottom],
        outline=(80, 80, 80),
        width=2,
    )

    ports = sample["ports"][~sample["port_padding"]]
    for port in ports:
        x = float((port[0] + 1.0) * 0.5 * target_size)
        y = float((port[1] + 1.0) * 0.5 * target_size)
        px, py = point([x, y])
        draw.ellipse(
            [px - 3, py - 3, px + 3, py + 3],
            fill=(20, 20, 20),
        )

    for edge in graph["edges"]:
        colour = (
            (205, 75, 55)
            if edge["mode"] == "road"
            else (55, 125, 185)
        )
        draw.line(
            [
                point(value)
                for value in edge["geometry_local_m"]
            ],
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="test",
    )
    parser.add_argument("--drop-controls", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    tensor_config = SpatialTensorConfig(**checkpoint["tensor_config"])
    program_config = FrontierProgramConfig(**checkpoint["program_config"])
    model_config = FrontierArchitectConfig.from_dict(
        checkpoint["model_config"]
    )
    dataset = FrontierProgramDataset(
        args.data,
        tensor_config=tensor_config,
        program_config=program_config,
    )

    candidates = [
        index
        for index, sample in enumerate(dataset.samples)
        if args.split == "all" or sample["split"] == args.split
    ]
    candidates.sort(
        key=lambda index: hashlib.sha1(
            str(dataset.samples[index]["sample_id"]).encode("utf-8"),
            usedforsecurity=False,
        ).digest()
    )
    indexes = candidates[: args.samples]
    if not indexes:
        raise RuntimeError("No frontier samples found for requested split")

    device = torch.device("cuda")
    model = FrontierArchitect(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    panels = []
    for sample_index, dataset_index in enumerate(indexes):
        sample = dataset[dataset_index]
        target = target_graph(sample, tensor_config)
        images = [render(target, sample, tensor_config)]
        record = {
            "sample_id": sample["sample_id"],
            "target": graph_stats(
                target,
                sample,
                tensor_config,
            ),
            "generations": [],
        }

        for seed_index in range(args.seeds):
            seed = 1000 + sample_index * 100 + seed_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generation_sample = dict(sample)
            if args.drop_controls:
                generation_sample["controls"] = torch.zeros_like(
                    generation_sample["controls"]
                )
            generated = rollout(
                model,
                generation_sample,
                tensor_config,
                program_config,
                device,
                args.temperature,
            )
            images.append(
                render(
                    generated,
                    sample,
                    tensor_config,
                )
            )
            stats = graph_stats(
                generated,
                sample,
                tensor_config,
            )
            stats["seed"] = seed_index
            record["generations"].append(stats)
            path = args.output / (
                f"{sample_index:02d}-{sample['sample_id']}"
                f"-seed{seed_index}.json"
            )
            path.write_text(
                json.dumps(
                    {
                        "nodes": generated["nodes"],
                        "edges": generated["edges"],
                        "statistics": stats,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        panel = Image.new(
            "RGB",
            (720 * len(images), 750),
            "white",
        )
        draw = ImageDraw.Draw(panel)
        for image_index, image in enumerate(images):
            panel.paste(image, (image_index * 720, 30))
            title = (
                "target"
                if image_index == 0
                else f"seed {image_index - 1}"
            )
            draw.text(
                (image_index * 720 + 8, 8),
                title,
                fill=(0, 0, 0),
            )
        panel.save(
            args.output
            / f"{sample_index:02d}-{sample['sample_id']}.png"
        )
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
