from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from urban_model.city_plan_data import (
    GLOBAL_CHANNELS,
    PLAN_CHANNELS,
    CityPlanConfig,
    CityPlanDataset,
)
from urban_model.city_planner import CityPlanner, CityPlannerConfig
from urban_model.spatial_world_data import SpatialTensorConfig


def move_sample(sample, device, zero_controls):
    batch = {
        key: value.unsqueeze(0).to(device)
        if torch.is_tensor(value)
        else value
        for key, value in sample.items()
    }
    if zero_controls:
        batch["controls"] = torch.zeros_like(batch["controls"])
    return batch


def render_channel(values, grid_size, *, scale=None, size=384):
    tensor = values.reshape(grid_size, grid_size).detach().cpu()
    maximum = float(tensor.max()) if scale is None else float(scale)
    maximum = max(maximum, 1e-6)
    cell = max(size // grid_size, 1)
    image = Image.new(
        "RGB",
        (cell * grid_size, cell * grid_size),
        "white",
    )
    draw = ImageDraw.Draw(image)
    for row in range(grid_size):
        for column in range(grid_size):
            value = float(tensor[row, column])
            level = int(
                round(
                    255.0
                    * min(max(value / maximum, 0.0), 1.0)
                )
            )
            colour = (255 - level, 255 - level, 255 - level)
            x0 = column * cell
            y0 = (grid_size - 1 - row) * cell
            draw.rectangle(
                [x0, y0, x0 + cell - 1, y0 + cell - 1],
                fill=colour,
            )
    return image


def compose_sample(
    target,
    predicted,
    grid_size,
    channels,
):
    tiles = []
    labels = []
    for channel in channels:
        index = PLAN_CHANNELS.index(channel)
        maximum = max(
            float(target[:, index].max()),
            float(predicted[:, index].max()),
            1.0,
        )
        tiles.append(
            render_channel(
                target[:, index],
                grid_size,
                scale=maximum,
            )
        )
        labels.append(f"target {channel}")
        tiles.append(
            render_channel(
                predicted[:, index],
                grid_size,
                scale=maximum,
            )
        )
        labels.append(f"pred {channel}")

    tile_width = tiles[0].width
    tile_height = tiles[0].height
    panel = Image.new(
        "RGB",
        (tile_width * len(tiles), tile_height + 28),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    for index, tile in enumerate(tiles):
        x = index * tile_width
        panel.paste(tile, (x, 28))
        draw.text((x + 5, 7), labels[index], fill=(0, 0, 0))
    return panel


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
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
    maximum_samples = checkpoint.get("maximum_samples")
    dataset = CityPlanDataset(
        args.data,
        tensor_config=tensor_config,
        plan_config=plan_config,
        maximum_samples=maximum_samples,
    )
    model_config = CityPlannerConfig.from_dict(
        checkpoint["model_config"]
    )
    device = torch.device("cuda")
    model = CityPlanner(model_config).to(device)
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
        raise RuntimeError("No city planner samples found")

    plan_mean = checkpoint["plan_mean"].to(device)
    plan_std = checkpoint["plan_std"].to(device)
    global_mean = checkpoint["global_mean"].to(device)
    global_std = checkpoint["global_std"].to(device)
    zero_controls = not bool(
        checkpoint.get("target_controls", False)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    panels = []
    channels = (
        "junctions",
        "major_edges",
        "local_edges",
        "rail_edges",
    )

    for order, index in enumerate(indexes):
        sample = dataset[index]
        batch = move_sample(
            sample,
            device,
            zero_controls,
        )
        output = model(batch)
        predicted_plan = (
            output["plan_grid"][0] * plan_std[None]
            + plan_mean[None]
        ).clamp_min(0.0).detach().cpu()
        predicted_global = (
            output["plan_global"][0] * global_std
            + global_mean
        ).detach().cpu()
        target_plan = sample["plan_grid_raw"]
        target_global = sample["plan_global_raw"]

        panel = compose_sample(
            target_plan,
            predicted_plan,
            plan_config.grid_size,
            channels,
        )
        panel.save(
            args.output
            / f"{order:02d}-{sample['sample_id']}.png"
        )
        panels.append(panel)

        record = {
            "sample_id": sample["sample_id"],
            "target_global": {
                name: float(target_global[position])
                for position, name in enumerate(GLOBAL_CHANNELS)
            },
            "predicted_global": {
                name: float(predicted_global[position])
                for position, name in enumerate(GLOBAL_CHANNELS)
            },
            "plan_channel_mae": {
                name: float(
                    (
                        predicted_plan[:, position]
                        - target_plan[:, position]
                    )
                    .abs()
                    .mean()
                )
                for position, name in enumerate(PLAN_CHANNELS)
            },
        }
        records.append(record)

    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(args.output / "planner.png")

    summary = {"samples": records}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
