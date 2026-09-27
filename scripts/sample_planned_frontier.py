from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from urban_model.city_plan_data import CityPlanConfig
from urban_model.frontier_data import (
    OP_BOS,
    OP_CLOSE,
    OP_EOS,
    OP_GROW,
    OP_LINK,
    OP_ROOT,
    FrontierProgramConfig,
)
from urban_model.planned_frontier import (
    PlannedFrontierArchitect,
    PlannedFrontierConfig,
)
from urban_model.planned_frontier_data import (
    PROGRESS_CHANNELS,
    PlannedFrontierDataset,
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
    probabilities = torch.softmax(
        logits / temperature,
        dim=-1,
    )
    return int(
        torch.multinomial(probabilities, 1).item()
    )


def normal(mean, logstd, temperature):
    if temperature <= 0:
        return mean
    return (
        mean
        + torch.exp(logstd)
        * torch.randn_like(mean)
        * temperature
    )


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
                "position": positions[index].clone(),
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
        start = positions[left]
        end = positions[right]
        straight = torch.stack(
            [
                start
                + (end - start)
                * (point + 1)
                / (tensor_config.edge_shape_points + 1)
                for point in range(
                    tensor_config.edge_shape_points
                )
            ]
        )
        internal = straight + sample["edge_shape"][index]
        values = [start, *internal, end]
        transport_class = TRANSPORT_CLASSES[
            int(sample["edge_class"][index])
        ]
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


def clear_program(batch):
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
        "program_progress",
    )
    for name in names:
        batch[name] = torch.zeros_like(batch[name])
    batch["program_active_node"].fill_(-1)
    batch["program_op"][0, 0] = OP_BOS
    batch["program_length"][0] = 2


def budgets(plan_global_raw):
    nodes = max(1, int(round(float(plan_global_raw[0]))))
    edges = max(0, int(round(float(plan_global_raw[1]))))
    components = max(
        1,
        int(round(float(plan_global_raw[2]))),
    )
    edge_total = max(float(plan_global_raw[1]), 1.0)
    classes = [
        max(float(plan_global_raw[4]) * edge_total, 1.0),
        max(float(plan_global_raw[5]) * edge_total, 1.0),
        max(float(plan_global_raw[6]) * edge_total, 1.0),
        max(float(plan_global_raw[3]) * edge_total, 1.0),
    ]
    return nodes, edges, components, classes


def progress_vector(state, plan_global_raw):
    node_budget, edge_budget, component_budget, class_budgets = budgets(
        plan_global_raw
    )
    values = [
        state["nodes"] / node_budget,
        state["edges"] / max(edge_budget, 1),
        state["components"] / component_budget,
    ]
    values.extend(
        state["classes"][index] / class_budgets[index]
        for index in range(4)
    )
    return torch.tensor(
        values,
        dtype=torch.float32,
        device=plan_global_raw.device,
    ).clamp(0.0, 2.0)


def write_event(
    batch,
    index,
    state,
    plan_global_raw,
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
    active=None,
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
    batch["program_active_node"][0, index] = (
        active if active is not None else -1
    )
    if active_xy is not None:
        batch["program_active_xy"][0, index] = active_xy
    batch["program_progress"][0, index] = progress_vector(
        state,
        plan_global_raw,
    )


def legal_op_logits(
    logits,
    *,
    active,
    node_count,
    edge_count,
    root_count,
    node_budget,
    edge_budget,
    component_budget,
):
    result = torch.full_like(logits, float("-inf"))
    if active is None:
        if node_count == 0:
            result[OP_ROOT] = logits[OP_ROOT]
            return result
        if (
            root_count < component_budget
            and node_count < node_budget
        ):
            result[OP_ROOT] = logits[OP_ROOT]
        result[OP_EOS] = logits[OP_EOS]
        return result

    result[OP_CLOSE] = logits[OP_CLOSE]
    if node_count < node_budget and edge_count < edge_budget:
        result[OP_GROW] = logits[OP_GROW]
    if node_count >= 2 and edge_count < edge_budget:
        result[OP_LINK] = logits[OP_LINK]
    return result


def nearest_link_target(
    nodes,
    active,
    expected,
    edge_pairs,
):
    candidates = []
    for node in nodes:
        node_id = int(node["id"])
        if node_id == active:
            continue
        pair = tuple(sorted((active, node_id)))
        if pair in edge_pairs:
            continue
        distance = float(
            torch.linalg.vector_norm(
                node["position"] - expected
            )
        )
        candidates.append((distance, node_id))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


@torch.inference_mode()
def rollout(
    model,
    sample,
    tensor_config,
    program_config,
    device,
    temperature,
):
    batch = move_sample(sample, device)
    clear_program(batch)
    memory, memory_padding = model.encode_plan(batch)
    plan_global_raw = batch["plan_global_raw"][0]
    (
        node_budget,
        edge_budget,
        component_budget,
        _class_budgets,
    ) = budgets(plan_global_raw)

    nodes = []
    edges = []
    queue = []
    active = None
    edge_pairs = set()
    state = {
        "nodes": 0,
        "edges": 0,
        "components": 0,
        "classes": [0, 0, 0, 0],
    }
    clamped_nodes = 0
    prefix = 1
    terminated = False

    while prefix < program_config.max_steps:
        batch["program_length"][0] = prefix + 1
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model.decode_program(
                batch,
                memory,
                memory_padding,
                input_length=prefix,
                last_only=True,
            )
        op_logits = legal_op_logits(
            output["op"][0, 0],
            active=active,
            node_count=len(nodes),
            edge_count=len(edges),
            root_count=state["components"],
            node_budget=node_budget,
            edge_budget=edge_budget,
            component_budget=component_budget,
        )
        op = categorical(op_logits, temperature)

        if op == OP_EOS:
            write_event(
                batch,
                prefix,
                state,
                plan_global_raw,
                op=OP_EOS,
            )
            prefix += 1
            terminated = True
            break

        if op == OP_ROOT:
            position = normal(
                output["xy_mean"][0, 0],
                output["xy_logstd"][0, 0],
                temperature,
            ).clamp(-1.0, 1.0)
            node_id = len(nodes)
            node_mode = categorical(
                output["node_mode"][0, 0],
                temperature,
            )
            node_vertical = categorical(
                output["node_vertical"][0, 0],
                temperature,
            )
            boundary = float(
                torch.sigmoid(
                    output["node_boundary"][0, 0]
                )
            )
            nodes.append(
                {
                    "id": node_id,
                    "position": position.detach().clone(),
                    "mode": node_mode,
                    "vertical": node_vertical,
                }
            )
            queue.append(node_id)
            active = node_id
            state["nodes"] += 1
            state["components"] += 1
            write_event(
                batch,
                prefix,
                state,
                plan_global_raw,
                op=OP_ROOT,
                xy=position,
                node_mode=node_mode,
                node_vertical=node_vertical,
                node_boundary=boundary,
                active=node_id,
                active_xy=position,
            )

        elif op == OP_GROW:
            delta = normal(
                output["xy_mean"][0, 0],
                output["xy_logstd"][0, 0],
                temperature,
            ).clamp(-1.0, 1.0)
            raw_position = (
                nodes[active]["position"] + delta * 2.0
            )
            position = raw_position.clamp(-1.0, 1.0)
            if not torch.equal(raw_position, position):
                clamped_nodes += 1
            child = len(nodes)
            node_mode = categorical(
                output["node_mode"][0, 0],
                temperature,
            )
            node_vertical = categorical(
                output["node_vertical"][0, 0],
                temperature,
            )
            boundary = float(
                torch.sigmoid(
                    output["node_boundary"][0, 0]
                )
            )
            edge_class = categorical(
                output["edge_class"][0, 0],
                temperature,
            )
            edge_vertical = categorical(
                output["edge_vertical"][0, 0],
                temperature,
            )
            width = float(
                normal(
                    output["width_mean"][0, 0],
                    output["width_logstd"][0, 0],
                    temperature,
                )[0]
                .clamp_min(0.0)
            )
            curve = normal(
                output["curve_mean"][0, 0],
                output["curve_logstd"][0, 0],
                temperature,
            ).clamp(-1.0, 1.0)
            nodes.append(
                {
                    "id": child,
                    "position": position.detach().clone(),
                    "mode": node_mode,
                    "vertical": node_vertical,
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
            edge_pairs.add(
                tuple(sorted((active, child)))
            )
            queue.append(child)
            state["nodes"] += 1
            state["edges"] += 1
            group = edge_class if edge_class < 3 else 3
            state["classes"][group] += 1
            write_event(
                batch,
                prefix,
                state,
                plan_global_raw,
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
            delta = normal(
                output["xy_mean"][0, 0],
                output["xy_logstd"][0, 0],
                temperature,
            ).clamp(-1.0, 1.0)
            expected = (
                nodes[active]["position"] + delta * 2.0
            ).clamp(-1.0, 1.0)
            target = nearest_link_target(
                nodes,
                active,
                expected,
                edge_pairs,
            )
            if target is None:
                op = OP_CLOSE
            else:
                edge_class = categorical(
                    output["edge_class"][0, 0],
                    temperature,
                )
                edge_vertical = categorical(
                    output["edge_vertical"][0, 0],
                    temperature,
                )
                width = float(
                    normal(
                        output["width_mean"][0, 0],
                        output["width_logstd"][0, 0],
                        temperature,
                    )[0]
                    .clamp_min(0.0)
                )
                curve = normal(
                    output["curve_mean"][0, 0],
                    output["curve_logstd"][0, 0],
                    temperature,
                ).clamp(-1.0, 1.0)
                actual_delta = (
                    nodes[target]["position"]
                    - nodes[active]["position"]
                ) / 2.0
                edges.append(
                    {
                        "from_node": active,
                        "to_node": target,
                        "class": edge_class,
                        "vertical": edge_vertical,
                        "width": width,
                        "curve": curve.detach().clone(),
                    }
                )
                edge_pairs.add(
                    tuple(sorted((active, target)))
                )
                state["edges"] += 1
                group = (
                    edge_class
                    if edge_class < 3
                    else 3
                )
                state["classes"][group] += 1
                write_event(
                    batch,
                    prefix,
                    state,
                    plan_global_raw,
                    op=OP_LINK,
                    xy=actual_delta,
                    edge_class=edge_class,
                    edge_vertical=edge_vertical,
                    width=width,
                    curve=curve,
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
                state,
                plan_global_raw,
                op=OP_CLOSE,
                active=active,
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
                "mode": "road"
                if node["mode"] == 0
                else "rail",
                "vertical_mode": VERTICAL_MODES[
                    node["vertical"]
                ],
            }
        )

    graph_edges = []
    for index, edge in enumerate(edges):
        transport_class = TRANSPORT_CLASSES[
            edge["class"]
        ]
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
                "vertical_mode": VERTICAL_MODES[
                    edge["vertical"]
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

    return {
        "nodes": graph_nodes,
        "edges": graph_edges,
        "roots": state["components"],
        "program_steps": prefix,
        "clamped_nodes": clamped_nodes,
        "terminated_eos": terminated,
        "node_budget": node_budget,
        "edge_budget": edge_budget,
        "component_budget": component_budget,
    }


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

    result = {
        "nodes": node_count,
        "edges": len(graph["edges"]),
        "components": len(components),
        "largest_component_fraction": (
            max(components, default=0) / max(node_count, 1)
        ),
        "isolated_fraction": (
            sum(not values for values in adjacency)
            / max(node_count, 1)
        ),
        "road_edges": sum(
            edge["mode"] == "road"
            for edge in graph["edges"]
        ),
        "rail_edges": sum(
            edge["mode"] == "rail"
            for edge in graph["edges"]
        ),
    }
    for name in (
        "roots",
        "program_steps",
        "clamped_nodes",
        "terminated_eos",
        "node_budget",
        "edge_budget",
        "component_budget",
    ):
        if name in graph:
            result[name] = graph[name]
    if "node_budget" in graph:
        result["node_budget_error"] = (
            node_count - graph["node_budget"]
        )
        result["edge_budget_error"] = (
            len(graph["edges"]) - graph["edge_budget"]
        )
        result["component_budget_error"] = (
            len(components) - graph["component_budget"]
        )
    return result


def render_plan_background(draw, sample, size):
    grid = int(
        round(
            sample["plan_presence"].shape[0] ** 0.5
        )
    )
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


def render(graph, sample, target_size, size=720):
    image = Image.new(
        "RGB",
        (size, size),
        (250, 249, 246),
    )
    draw = ImageDraw.Draw(image)
    render_plan_background(draw, sample, size)

    def point(value):
        return (
            int(round(value[0] / target_size * (size - 1))),
            int(
                round(
                    (1.0 - value[1] / target_size)
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
        default="all",
    )
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
    program_config = FrontierProgramConfig(
        **checkpoint["program_config"]
    )
    maximum_samples = checkpoint.get("maximum_samples")
    dataset = PlannedFrontierDataset(
        args.data,
        tensor_config=tensor_config,
        plan_config=plan_config,
        program_config=program_config,
        maximum_samples=maximum_samples,
    )
    model_config = PlannedFrontierConfig.from_dict(
        checkpoint["model_config"]
    )

    device = torch.device("cuda")
    model = PlannedFrontierArchitect(
        model_config
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    candidates = [
        index
        for index, sample in enumerate(dataset.samples)
        if args.split == "all" or sample["split"] == args.split
    ]
    candidates.sort(
        key=lambda index: hashlib.sha1(
            str(dataset.samples[index]["sample_id"]).encode(
                "utf-8"
            ),
            usedforsecurity=False,
        ).digest()
    )
    indexes = candidates[: args.samples]
    if not indexes:
        raise RuntimeError("No planned frontier samples found")

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    panels = []
    for sample_order, dataset_index in enumerate(indexes):
        sample = dataset[dataset_index]
        target = target_graph(
            sample,
            tensor_config,
        )
        images = [
            render(
                target,
                sample,
                tensor_config.target_size_m,
            )
        ]
        record = {
            "sample_id": sample["sample_id"],
            "target": graph_stats(target),
            "generations": [],
        }

        for seed_index in range(args.seeds):
            seed = (
                9000
                + sample_order * 100
                + seed_index
            )
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            generated = rollout(
                model,
                sample,
                tensor_config,
                program_config,
                device,
                args.temperature,
            )
            images.append(
                render(
                    generated,
                    sample,
                    tensor_config.target_size_m,
                )
            )
            stats = graph_stats(generated)
            stats["seed"] = seed_index
            record["generations"].append(stats)
            (args.output / (
                f"{sample_order:02d}-{sample['sample_id']}"
                f"-seed{seed_index}.json"
            )).write_text(
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
            panel.paste(
                image,
                (image_index * 720, 30),
            )
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
            / f"{sample_order:02d}-{sample['sample_id']}.png"
        )
        panels.append(panel)
        records.append(record)

    sheet = Image.new(
        "RGB",
        (
            720 * (args.seeds + 1),
            750 * len(panels),
        ),
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
