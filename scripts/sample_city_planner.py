from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from urban_model.city_plan_data import (
    GLOBAL_CHANNELS,
    ORIENTATION_CHANNELS,
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


def render_channel(
    values,
    presence,
    grid_size,
    *,
    orientation=None,
    size=384,
):
    tensor = values.reshape(grid_size, grid_size).detach().cpu()
    occupied = presence.reshape(
        grid_size,
        grid_size,
    ).detach().cpu()
    maximum = max(float(tensor.max()), 1.0)
    cell = max(size // grid_size, 1)
    image = Image.new(
        "RGB",
        (cell * grid_size, cell * grid_size),
        "white",
    )
    draw = ImageDraw.Draw(image)
    for row in range(grid_size):
        for column in range(grid_size):
            probability = float(occupied[row, column])
            value = float(tensor[row, column])
            density = min(max(value / maximum, 0.0), 1.0)
            level = int(
                round(
                    255.0
                    * min(
                        max(
                            probability * (0.35 + 0.65 * density),
                            0.0,
                        ),
                        1.0,
                    )
                )
            )
            colour = (255 - level, 255 - level, 255 - level)
            x0 = column * cell
            y0 = (grid_size - 1 - row) * cell
            draw.rectangle(
                [x0, y0, x0 + cell - 1, y0 + cell - 1],
                fill=colour,
            )

    if orientation is not None:
        vectors = orientation.reshape(
            grid_size,
            grid_size,
            2,
        ).detach().cpu()
        for row in range(grid_size):
            for column in range(grid_size):
                if float(occupied[row, column]) < 0.5:
                    continue
                vector = vectors[row, column]
                magnitude = float(torch.linalg.vector_norm(vector))
                if magnitude < 0.1:
                    continue
                angle = 0.5 * math.atan2(
                    float(vector[1]),
                    float(vector[0]),
                )
                cx = column * cell + cell * 0.5
                cy = (grid_size - 1 - row) * cell + cell * 0.5
                radius = cell * 0.35
                dx = math.cos(angle) * radius
                dy = -math.sin(angle) * radius
                draw.line(
                    [cx - dx, cy - dy, cx + dx, cy + dy],
                    fill=(180, 45, 45),
                    width=max(1, cell // 10),
                )
    return image


def compose_sample(
    target_counts,
    target_presence,
    target_orientation,
    predicted_counts,
    predicted_probability,
    predicted_orientation,
    grid_size,
    channels,
):
    tiles = []
    labels = []
    for channel in channels:
        index = PLAN_CHANNELS.index(channel)
        orientation_index = (
            ORIENTATION_CHANNELS.index(channel)
            if channel in ORIENTATION_CHANNELS
            else None
        )
        target_vector = (
            target_orientation[:, orientation_index]
            if orientation_index is not None
            else None
        )
        predicted_vector = (
            predicted_orientation[:, orientation_index]
            if orientation_index is not None
            else None
        )
        tiles.append(
            render_channel(
                target_counts[:, index],
                target_presence[:, index],
                grid_size,
                orientation=target_vector,
            )
        )
        labels.append(f"target {channel}")
        tiles.append(
            render_channel(
                predicted_counts[:, index],
                predicted_probability[:, index],
                grid_size,
                orientation=predicted_vector,
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


def channel_iou(predicted, target):
    predicted = predicted > 0.5
    target = target > 0.5
    intersection = int((predicted & target).sum())
    union = int((predicted | target).sum())
    return intersection / max(union, 1)


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
        "major_corridor",
        "local_corridor",
        "rail_corridor",
    )

    for order, index in enumerate(indexes):
        sample = dataset[index]
        batch = move_sample(
            sample,
            device,
            zero_controls,
        )
        output = model(batch)
        predicted_probability = torch.sigmoid(
            output["plan_presence"][0]
        ).detach().cpu()
        predicted_counts = (
            torch.expm1(output["plan_log_count"][0])
            .clamp_min(0.0)
            .detach()
            .cpu()
        )
        predicted_orientation = (
            output["plan_orientation"][0]
            .detach()
            .cpu()
        )
        predicted_global = (
            output["plan_global"][0] * global_std
            + global_mean
        ).detach().cpu()

        target_counts = sample["plan_counts"]
        target_presence = sample["plan_presence"]
        target_orientation = sample["plan_orientation"]
        target_global = sample["plan_global_raw"]

        panel = compose_sample(
            target_counts,
            target_presence,
            target_orientation,
            predicted_counts,
            predicted_probability,
            predicted_orientation,
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
            "presence_iou": {
                name: channel_iou(
                    predicted_probability[:, position],
                    target_presence[:, position],
                )
                for position, name in enumerate(PLAN_CHANNELS)
            },
            "count_mae": {
                name: float(
                    (
                        predicted_counts[:, position]
                        - target_counts[:, position]
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
