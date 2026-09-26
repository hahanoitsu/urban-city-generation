from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from urban_model.structured_city import StructuredCityConfig, StructuredCityDenoiser
from urban_model.structured_city_data import (
    AREA_KINDS,
    BUILDING_KINDS,
    CLASSES,
    SceneTensorConfig,
    StructuredCityDataset,
    VERTICAL,
)
from urban_model.structured_city_diffusion import CATEGORY_MASKS, CONTINUOUS_FIELDS, noise_coefficients


def region_bucket(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "little") % 100


def initial_scene(config: SceneTensorConfig, device: torch.device):
    return {
        "node_position": torch.randn(1, config.node_slots, 3, device=device),
        "node_presence": torch.full((1, config.node_slots), 2, dtype=torch.long, device=device),
        "edge_width": torch.randn(1, config.edge_slots, 1, device=device),
        "edge_shape": torch.randn(
            1,
            config.edge_slots,
            config.edge_shape_points,
            3,
            device=device,
        ),
        "edge_presence": torch.full((1, config.edge_slots), 2, dtype=torch.long, device=device),
        "edge_mode": torch.full((1, config.edge_slots), 2, dtype=torch.long, device=device),
        "edge_class": torch.full((1, config.edge_slots), 7, dtype=torch.long, device=device),
        "edge_vertical": torch.full((1, config.edge_slots), 4, dtype=torch.long, device=device),
        "building_shape": torch.randn(
            1,
            config.building_slots,
            config.building_points,
            2,
            device=device,
        ),
        "building_height": torch.randn(1, config.building_slots, 1, device=device),
        "building_base_z": torch.randn(1, config.building_slots, 1, device=device),
        "building_presence": torch.full(
            (1, config.building_slots),
            2,
            dtype=torch.long,
            device=device,
        ),
        "building_kind": torch.full(
            (1, config.building_slots),
            len(BUILDING_KINDS),
            dtype=torch.long,
            device=device,
        ),
        "area_shape": torch.randn(
            1,
            config.area_slots,
            config.area_points,
            2,
            device=device,
        ),
        "area_presence": torch.full((1, config.area_slots), 2, dtype=torch.long, device=device),
        "area_kind": torch.full(
            (1, config.area_slots),
            len(AREA_KINDS),
            dtype=torch.long,
            device=device,
        ),
    }


def update_categories(scene, output, next_fraction):
    for name, mask_index in CATEGORY_MASKS.items():
        logits = output[name]
        probability = torch.softmax(logits, dim=-1)
        confidence, prediction = probability.max(dim=-1)
        values = prediction.clone()
        count = values.shape[1]
        masked = int(round(next_fraction * count))
        if masked > 0:
            indexes = torch.topk(confidence, k=masked, dim=1, largest=False).indices
            values.scatter_(1, indexes, mask_index)
        scene[name] = values


def update_continuous(current, prediction, time, next_time):
    alpha, sigma = noise_coefficients(time)
    next_alpha, next_sigma = noise_coefficients(next_time)
    shape = [current.shape[0]] + [1] * (current.ndim - 1)
    alpha = alpha.reshape(shape)
    sigma = sigma.reshape(shape).clamp_min(1e-5)
    next_alpha = next_alpha.reshape(shape)
    next_sigma = next_sigma.reshape(shape)
    epsilon = (current - alpha * prediction) / sigma
    return next_alpha * prediction + next_sigma * epsilon


@torch.inference_mode()
def generate(model, context, relations, context_padding, ports, padding, scene_config, steps):
    device = context.device
    scene = initial_scene(scene_config, device)
    for index in range(steps, 0, -1):
        value = index / steps
        next_value = (index - 1) / steps
        time = torch.full((1,), value, device=device)
        next_time = torch.full((1,), next_value, device=device)
        output = model(
            scene,
            context,
            relations,
            context_padding,
            ports,
            padding,
            time,
        )
        for name in CONTINUOUS_FIELDS:
            scene[name] = update_continuous(scene[name], output[name], time, next_time)
        update_categories(scene, output, next_value)
    output = model(
        scene,
        context,
        relations,
        context_padding,
        ports,
        padding,
        torch.zeros(1, device=device),
    )
    for name in CONTINUOUS_FIELDS:
        scene[name] = output[name]
    for name in CATEGORY_MASKS:
        scene[name] = output[name].argmax(dim=-1)
    scene["edge_from"] = output["edge_from"].argmax(dim=-1)
    scene["edge_to"] = output["edge_to"].argmax(dim=-1)
    return {name: value[0].detach().cpu() for name, value in scene.items()}


def target_scene(sample):
    names = (
        "node_position",
        "node_presence",
        "edge_width",
        "edge_shape",
        "edge_presence",
        "edge_mode",
        "edge_class",
        "edge_vertical",
        "edge_from",
        "edge_to",
        "building_shape",
        "building_height",
        "building_base_z",
        "building_presence",
        "building_kind",
        "area_shape",
        "area_presence",
        "area_kind",
    )
    return {name: sample[name] for name in names}


def denormalise(points, size):
    return (points + 1.0) * 0.5 * size


def render(scene, config, size=768):
    image = Image.new("RGB", (size, size), (245, 243, 235))
    draw = ImageDraw.Draw(image)

    def point(value):
        x = float((value[0] + 1.0) * 0.5 * (size - 1))
        y = float((1.0 - (value[1] + 1.0) * 0.5) * (size - 1))
        return (int(round(x)), int(round(y)))

    area_colours = {
        0: (170, 205, 160),
        1: (150, 195, 220),
        2: (238, 225, 197),
        3: (225, 207, 190),
        4: (211, 207, 197),
        5: (219, 213, 190),
    }
    for index in range(config.area_slots):
        if int(scene["area_presence"][index]) != 1:
            continue
        kind = int(scene["area_kind"][index])
        if kind not in area_colours:
            continue
        points = [point(value) for value in scene["area_shape"][index]]
        if len(points) >= 3:
            draw.polygon(points, fill=area_colours[kind])

    for index in range(config.building_slots):
        if int(scene["building_presence"][index]) != 1:
            continue
        points = [point(value) for value in scene["building_shape"][index]]
        if len(points) >= 3:
            draw.polygon(points, fill=(150, 150, 150), outline=(100, 100, 100))

    nodes = scene["node_position"]
    present_nodes = scene["node_presence"].eq(1)
    for index in range(config.edge_slots):
        if int(scene["edge_presence"][index]) != 1:
            continue
        left = int(scene["edge_from"][index])
        right = int(scene["edge_to"][index])
        if (
            left < 0
            or right < 0
            or left >= config.node_slots
            or right >= config.node_slots
            or left == right
            or not bool(present_nodes[left])
            or not bool(present_nodes[right])
        ):
            continue
        start = nodes[left, :2]
        end = nodes[right, :2]
        straight = torch.stack(
            [
                start + (end - start) * fraction
                for fraction in torch.linspace(0.0, 1.0, config.edge_shape_points)
            ]
        )
        line = straight + scene["edge_shape"][index, :, :2]
        points = [point(value) for value in line]
        mode = int(scene["edge_mode"][index])
        colour = (205, 75, 55) if mode == 0 else (65, 145, 185)
        width = max(1, int(round(abs(float(scene["edge_width"][index, 0])) * 8)))
        draw.line(points, fill=colour, width=width, joint="curve")

    for index in range(config.node_slots):
        if not bool(present_nodes[index]):
            continue
        x, y = point(nodes[index, :2])
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(30, 30, 30))

    return image


def export_scene(scene, config):
    nodes = []
    node_map = {}
    for index in range(config.node_slots):
        if int(scene["node_presence"][index]) != 1:
            continue
        node_map[index] = len(nodes)
        x, y = denormalise(scene["node_position"][index, :2], config.target_size_m)
        nodes.append(
            {
                "id": len(nodes),
                "xyz_m": [float(x), float(y), None],
            }
        )

    edges = []
    for index in range(config.edge_slots):
        if int(scene["edge_presence"][index]) != 1:
            continue
        left = int(scene["edge_from"][index])
        right = int(scene["edge_to"][index])
        if left not in node_map or right not in node_map or left == right:
            continue
        start = scene["node_position"][left, :2]
        end = scene["node_position"][right, :2]
        straight = torch.stack(
            [
                start + (end - start) * fraction
                for fraction in torch.linspace(0.0, 1.0, config.edge_shape_points)
            ]
        )
        line = straight + scene["edge_shape"][index, :, :2]
        geometry = denormalise(line, config.target_size_m)
        mode = int(scene["edge_mode"][index])
        edge_class = int(scene["edge_class"][index])
        vertical = int(scene["edge_vertical"][index])
        edges.append(
            {
                "from": node_map[left],
                "to": node_map[right],
                "mode": "road" if mode == 0 else "rail",
                "class": CLASSES[edge_class] if 0 <= edge_class < len(CLASSES) else "unknown",
                "vertical_mode": VERTICAL[vertical] if 0 <= vertical < len(VERTICAL) else "unknown",
                "width_m": float(scene["edge_width"][index, 0] * config.width_scale_m),
                "spline_xyz_m": [[float(x), float(y), None] for x, y in geometry],
            }
        )

    buildings = []
    for index in range(config.building_slots):
        if int(scene["building_presence"][index]) != 1:
            continue
        footprint = denormalise(scene["building_shape"][index], config.target_size_m)
        kind = int(scene["building_kind"][index])
        buildings.append(
            {
                "footprint_xy_m": [[float(x), float(y)] for x, y in footprint],
                "base_z_m": None,
                "height_m": float(scene["building_height"][index, 0] * config.height_scale_m),
                "type": BUILDING_KINDS[kind] if 0 <= kind < len(BUILDING_KINDS) else "generic",
            }
        )

    areas = []
    for index in range(config.area_slots):
        if int(scene["area_presence"][index]) != 1:
            continue
        kind = int(scene["area_kind"][index])
        if not 0 <= kind < len(AREA_KINDS):
            continue
        polygon = denormalise(scene["area_shape"][index], config.target_size_m)
        areas.append(
            {
                "kind": AREA_KINDS[kind],
                "polygon_xy_m": [[float(x), float(y)] for x, y in polygon],
            }
        )
    return {"nodes": nodes, "edges": edges, "buildings": buildings, "areas": areas}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = StructuredCityConfig.from_dict(checkpoint["model_config"])
    scene_config = SceneTensorConfig(**checkpoint["scene_config"])
    dataset = StructuredCityDataset(args.data, config=scene_config)
    indexes = [
        index
        for index, (row, _payload) in enumerate(dataset.samples)
        if region_bucket(str(row["parent_region_id"])) >= 88
    ][: args.samples]
    if not indexes:
        raise RuntimeError("No held-out test samples available")

    device = torch.device("cuda")
    model = StructuredCityDenoiser(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    panels = []
    records = []
    for output_index, dataset_index in enumerate(indexes):
        sample = dataset[dataset_index]
        context = sample["context"].unsqueeze(0).to(device)
        context_padding = sample["context_padding"].unsqueeze(0).to(device)
        relations = sample["relations"].unsqueeze(0).to(device)
        ports = sample["ports"].unsqueeze(0).to(device)
        padding = sample["port_padding"].unsqueeze(0).to(device)
        generated = generate(
            model,
            context,
            relations,
            context_padding,
            ports,
            padding,
            scene_config,
            args.steps,
        )
        target = target_scene(sample)
        target_image = render(target, scene_config)
        generated_image = render(generated, scene_config)
        panel = Image.new("RGB", (1536, 800), "white")
        panel.paste(target_image, (0, 32))
        panel.paste(generated_image, (768, 32))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), f"{sample['sample_id']} target", fill="black")
        draw.text((776, 8), "generated", fill="black")
        panel_path = args.output / f"{output_index:02d}-{sample['sample_id']}.png"
        panel.save(panel_path)
        panels.append(panel)

        payload = export_scene(generated, scene_config)
        json_path = args.output / f"{output_index:02d}-{sample['sample_id']}.json"
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        records.append(
            {
                "sample_id": sample["sample_id"],
                "nodes": len(payload["nodes"]),
                "edges": len(payload["edges"]),
                "buildings": len(payload["buildings"]),
                "areas": len(payload["areas"]),
            }
        )

    sheet = Image.new("RGB", (1536, 800 * len(panels)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 800))
    sheet.save(args.output / "previews.png")
    (args.output / "summary.json").write_text(json.dumps({"samples": records}, indent=2) + "\n")
    print(json.dumps({"samples": records}, indent=2))


if __name__ == "__main__":
    main()
