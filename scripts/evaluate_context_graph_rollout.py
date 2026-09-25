from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from urban_ai.codec import (
    FIELDS,
    OP_ADD,
    OP_BOS,
    OP_CONNECT,
    OP_EOS,
    OP_ROOT,
    classes_for_mode,
    empty_encoded_command,
    mode_name,
)
from urban_model.context_graph import ContextGraphModelConfig, ContextGraphProgramModel
from urban_model.context_graph_data import ContextGraphProgramDataset


def stack(commands, device):
    return {
        field: torch.tensor(
            [[command[field] for command in commands]],
            dtype=torch.long,
            device=device,
        )
        for field in FIELDS
    }


def choose(logits, allowed=None):
    if allowed is None:
        return int(torch.argmax(logits).item())
    indexes = torch.tensor(list(allowed), dtype=torch.long, device=logits.device)
    return int(indexes[torch.argmax(logits[indexes])].item())


def rollout(model, sample, dataset, device):
    encoded = [empty_encoded_command(OP_BOS)]
    nodes = []
    edges = set()
    context = sample["context"].unsqueeze(0).to(device)
    ports = sample["ports"].unsqueeze(0).to(device)
    padding = sample["port_padding"].unsqueeze(0).to(device)
    relations = dataset.relations.unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        for _ in range(model.config.maximum_sequence_length - 1):
            logits = model(stack(encoded, device), context, relations, ports, padding)
            last = {name: values[0, -1] for name, values in logits.items()}
            node_count = len(nodes)

            if node_count == 0:
                op = OP_ROOT
            else:
                allowed = [OP_EOS, OP_ROOT, OP_ADD]
                if node_count >= 2:
                    allowed.append(OP_CONNECT)
                op = choose(last["op"], allowed)

            command = empty_encoded_command(op)
            if op == OP_EOS:
                encoded.append(command)
                break

            if op == OP_ROOT:
                x = choose(last["x"])
                y = choose(last["y"])
                mode = choose(last["mode"])
                vertical = choose(last["vertical"])
                layer = choose(last["layer"])
                command["x"] = x + 1
                command["y"] = y + 1
                command["mode"] = mode + 1
                command["vertical"] = vertical + 1
                command["layer"] = layer + 1
                nodes.append((x, y))
            elif op == OP_ADD:
                parent = choose(last["id1"], range(node_count))
                x = choose(last["x"])
                y = choose(last["y"])
                mode = choose(last["mode"])
                edge_class = choose(last["class"], classes_for_mode(mode_name(mode)))
                width = choose(last["width"])
                vertical = choose(last["vertical"])
                layer = choose(last["layer"])
                command["x"] = x + 1
                command["y"] = y + 1
                command["id1"] = parent + 1
                command["mode"] = mode + 1
                command["class"] = edge_class + 1
                command["width"] = width + 1
                command["vertical"] = vertical + 1
                command["layer"] = layer + 1
                new_node = len(nodes)
                nodes.append((x, y))
                edges.add((min(parent, new_node), max(parent, new_node)))
            elif op == OP_CONNECT:
                left = choose(last["id1"], range(node_count))
                right_candidates = [
                    node
                    for node in range(node_count)
                    if node != left
                    and (min(left, node), max(left, node)) not in edges
                ]
                if not right_candidates:
                    command = empty_encoded_command(OP_EOS)
                    encoded.append(command)
                    break
                right = choose(last["id2"], right_candidates)
                mode = choose(last["mode"])
                edge_class = choose(last["class"], classes_for_mode(mode_name(mode)))
                width = choose(last["width"])
                vertical = choose(last["vertical"])
                layer = choose(last["layer"])
                command["id1"] = left + 1
                command["id2"] = right + 1
                command["mode"] = mode + 1
                command["class"] = edge_class + 1
                command["width"] = width + 1
                command["vertical"] = vertical + 1
                command["layer"] = layer + 1
                edges.add((min(left, right), max(left, right)))
            encoded.append(command)
        else:
            encoded.append(empty_encoded_command(OP_EOS))

    return encoded


def target_commands(sample):
    values = []
    length = int(sample["commands"])
    for index in range(length):
        values.append({field: int(sample[field][index]) for field in FIELDS})
    return values


def graph_from_commands(commands, bins):
    nodes = []
    edges = []
    for command in commands:
        op = command["op"]
        if op == OP_ROOT:
            nodes.append((command["x"] - 1, command["y"] - 1))
        elif op == OP_ADD:
            parent = command["id1"] - 1
            new_node = len(nodes)
            nodes.append((command["x"] - 1, command["y"] - 1))
            if 0 <= parent < new_node:
                edges.append((parent, new_node, command["mode"]))
        elif op == OP_CONNECT:
            left = command["id1"] - 1
            right = command["id2"] - 1
            if 0 <= left < len(nodes) and 0 <= right < len(nodes) and left != right:
                edges.append((left, right, command["mode"]))
    return nodes, edges


def render(commands, ports, bins, size=512):
    image = Image.new("RGB", (size, size), (245, 243, 235))
    draw = ImageDraw.Draw(image)
    nodes, edges = graph_from_commands(commands, bins)

    def point(node):
        x, y = node
        return (
            int(round(x / max(bins - 1, 1) * (size - 1))),
            int(round((1.0 - y / max(bins - 1, 1)) * (size - 1))),
        )

    for left, right, mode in edges:
        colour = (205, 75, 55) if mode == 1 else (65, 145, 185)
        draw.line([point(nodes[left]), point(nodes[right])], fill=colour, width=2)

    for node in nodes:
        x, y = point(node)
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(30, 30, 30))

    for port in ports:
        x, y = port["position_local_m"]
        px = int(round(float(x) / 512.0 * (size - 1)))
        py = int(round((1.0 - float(y) / 512.0) * (size - 1)))
        colour = (65, 145, 185) if port["mode"] == "rail" else (20, 20, 20)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], outline=colour, width=2)

    return image


def compare(target, generated):
    relevant = {
        "x": {OP_ROOT, OP_ADD},
        "y": {OP_ROOT, OP_ADD},
        "id1": {OP_ADD, OP_CONNECT},
        "id2": {OP_CONNECT},
        "mode": {OP_ROOT, OP_ADD, OP_CONNECT},
        "class": {OP_ADD, OP_CONNECT},
        "width": {OP_ADD, OP_CONNECT},
        "vertical": {OP_ROOT, OP_ADD, OP_CONNECT},
        "layer": {OP_ROOT, OP_ADD, OP_CONNECT},
    }
    length = min(len(target), len(generated))
    result = {}
    op_correct = sum(target[index]["op"] == generated[index]["op"] for index in range(length))
    result["op_accuracy"] = op_correct / max(len(target), len(generated), 1)
    for field, operations in relevant.items():
        total = 0
        correct = 0
        for index, expected in enumerate(target):
            if expected["op"] not in operations:
                continue
            total += 1
            if index < len(generated):
                actual = generated[index]
                if actual["op"] == expected["op"] and actual[field] == expected[field]:
                    correct += 1
        result[f"{field}_accuracy"] = correct / max(total, 1)

    prefix = 0
    for expected, actual in zip(target, generated):
        if expected != actual:
            break
        prefix += 1
    result["target_length"] = len(target)
    result["generated_length"] = len(generated)
    result["exact_prefix"] = prefix
    result["exact_program"] = target == generated
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ContextGraphModelConfig.from_dict(checkpoint["config"])
    dataset = ContextGraphProgramDataset(
        args.data,
        maximum_samples=args.samples,
        maximum_commands=config.maximum_sequence_length + 1,
        maximum_ports=96,
    )
    device = torch.device("cuda")
    model = ContextGraphProgramModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    panels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        target = target_commands(sample)
        generated = rollout(model, sample, dataset, device)
        metrics = compare(target, generated)
        metrics["sample_id"] = sample["sample_id"]
        records.append(metrics)

        payload = dataset.samples[index][1]
        target_image = render(
            target,
            payload["input"]["boundary_ports"],
            dataset.program_config.coordinate_bins,
        )
        generated_image = render(
            generated,
            payload["input"]["boundary_ports"],
            dataset.program_config.coordinate_bins,
        )
        panel = Image.new("RGB", (1024, 544), "white")
        panel.paste(target_image, (0, 32))
        panel.paste(generated_image, (512, 32))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), f"{sample['sample_id']} target", fill="black")
        draw.text((520, 8), "free rollout", fill="black")
        panel.save(args.output / f"{index:02d}-{sample['sample_id']}.png")
        panels.append(panel)

    sheet = Image.new("RGB", (1024, 544 * len(panels)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 544))
    sheet.save(args.output / "rollouts.png")

    summary = {
        "samples": len(records),
        "exact_programs": sum(record["exact_program"] for record in records),
        "mean_op_accuracy": sum(record["op_accuracy"] for record in records) / len(records),
        "mean_x_accuracy": sum(record["x_accuracy"] for record in records) / len(records),
        "mean_y_accuracy": sum(record["y_accuracy"] for record in records) / len(records),
        "mean_id1_accuracy": sum(record["id1_accuracy"] for record in records) / len(records),
        "mean_id2_accuracy": sum(record["id2_accuracy"] for record in records) / len(records),
        "records": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
