from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from urban_model.grid_city import GridCityConfig, GridCityDenoiser
from urban_model.grid_city_data import (
    GridCityDataset,
    decode_grid,
    encode_grid,
    json_from_plain,
    merge_chains,
)
from urban_model.grid_city_diffusion import generate
from urban_model.structured_city_data import SceneTensorConfig, StructuredCityDataset


AREA_COLOURS = {
    "green": (170, 205, 160),
    "water": (150, 195, 220),
    "residential": (238, 225, 197),
    "commercial_mixed": (225, 207, 190),
    "industrial": (211, 207, 197),
    "civic": (219, 213, 190),
}
ROAD_WIDTH = {"major": 4, "secondary": 3, "local": 2}


def region_bucket(value):
    digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "little") % 100


def render(payload, size=768, extent=512.0):
    image = Image.new("RGB", (size, size), (245, 243, 235))
    draw = ImageDraw.Draw(image)

    def point(value):
        return (value[0] / extent * (size - 1), (1.0 - value[1] / extent) * (size - 1))

    for area in payload["areas"]:
        points = [point(value) for value in area["polygon_xy_m"]]
        if len(points) >= 3:
            draw.polygon(points, fill=AREA_COLOURS.get(area["kind"], (220, 220, 220)))
    for building in payload["buildings"]:
        points = [point(value) for value in building["footprint_xy_m"]]
        if len(points) >= 3:
            draw.polygon(points, fill=(150, 150, 150), outline=(100, 100, 100))
    for edge in payload["edges"]:
        points = [point(value) for value in edge["spline_xyz_m"]]
        if edge["mode"] == "rail":
            draw.line(points, fill=(65, 145, 185), width=2)
        else:
            draw.line(points, fill=(205, 75, 55), width=ROAD_WIDTH.get(edge["class"], 2))
    for node in payload["nodes"]:
        x, y = point(node["xyz_m"])
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(30, 30, 30))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--split", choices=["all", "train", "validation", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = GridCityConfig.from_dict(checkpoint["model_config"])
    scene_config = SceneTensorConfig(**checkpoint["scene_config"])
    base = StructuredCityDataset(
        args.data,
        config=scene_config,
        maximum_samples=checkpoint.get("maximum_samples"),
        cache_dir=args.cache_dir,
    )
    split = args.split or ("all" if checkpoint.get("overfit") else "test")

    def wanted(row):
        if split == "all":
            return True
        bucket = region_bucket(str(row["parent_region_id"]))
        if split == "train":
            return bucket < 75
        if split == "validation":
            return 75 <= bucket < 88
        return bucket >= 88

    indexes = [index for index, (row, _payload, _scene) in enumerate(base.samples) if wanted(row)]
    indexes.sort(key=lambda index: hashlib.sha1(str(base.samples[index][0]["id"]).encode("utf-8"), usedforsecurity=False).digest())
    grid = model_config.grid()
    dataset = GridCityDataset(base, indexes[: args.samples * 2], grid, show_progress=False)
    if len(dataset) == 0:
        raise RuntimeError(f"no {split} samples")

    device = torch.device(args.device)
    model = GridCityDenoiser(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "targets").mkdir(exist_ok=True)
    panels = []
    records = []
    for output_index in range(min(args.samples, len(dataset))):
        sample = dataset[output_index]
        inputs = {
            name: sample[name][None].to(device)
            for name in ("context", "relations", "context_padding", "ports", "port_padding")
        }
        scene = generate(model, inputs, steps=args.steps, generator=generator, greedy=args.greedy)
        arrays = {name: value[0].float().cpu().numpy() if value.is_floating_point() else value[0].cpu().numpy() for name, value in scene.items()}
        generated = json_from_plain(merge_chains(decode_grid(arrays, grid)))
        _base_index, plain = dataset.items[output_index]
        target_arrays, _dropped = encode_grid(plain, grid)
        target = json_from_plain(merge_chains(decode_grid(target_arrays, grid)))

        name = f"{output_index:02d}-{sample['sample_id']}"
        (args.output / f"{name}.json").write_text(json.dumps(generated) + "\n")
        (args.output / "targets" / f"{name}.json").write_text(json.dumps(target) + "\n")
        panel = Image.new("RGB", (1536, 800), "white")
        panel.paste(render(target), (0, 32))
        panel.paste(render(generated), (768, 32))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), f"{sample['sample_id']} target", fill="black")
        draw.text((776, 8), "generated", fill="black")
        panel.save(args.output / f"{name}.png")
        panels.append(panel)
        records.append(
            {
                "sample_id": sample["sample_id"],
                "target": {key: len(target[key]) for key in ("nodes", "edges", "buildings", "areas")},
                "generated": {key: len(generated[key]) for key in ("nodes", "edges", "buildings", "areas")},
            }
        )
        print(json.dumps(records[-1]), flush=True)

    sheet = Image.new("RGB", (1536, 800 * len(panels)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 800))
    sheet.save(args.output / "previews.png")
    (args.output / "summary.json").write_text(json.dumps({"split": split, "samples": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
