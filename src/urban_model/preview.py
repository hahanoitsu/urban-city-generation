from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import model_space_to_layers

SURFACE_PALETTE = np.asarray(
    [
        (226, 221, 209),
        (111, 174, 105),
        (137, 142, 148),
        (215, 58, 48),
        (239, 132, 57),
        (246, 224, 125),
        (85, 176, 194),
        (78, 151, 211),
    ],
    dtype=np.uint8,
)
ROAD_COLOUR = np.asarray((246, 224, 125), dtype=np.float32)
RAIL_COLOUR = np.asarray((85, 176, 194), dtype=np.float32)
BACKGROUND = np.asarray((17, 23, 26), dtype=np.uint8)


def _surface_image(surface: torch.Tensor, valid_mask: torch.Tensor | None = None) -> Image.Image:
    array = surface.detach().cpu().numpy().astype(np.int64)
    image = SURFACE_PALETTE[array]
    if valid_mask is not None:
        valid = valid_mask.detach().cpu().numpy() > 0.5
        image[~valid] = BACKGROUND
    return Image.fromarray(image)


def _apply_profile_colour(
    image: np.ndarray,
    mask: np.ndarray,
    profile_m: np.ndarray,
    base_colour: np.ndarray,
) -> None:
    if not mask.any():
        return
    active_values = profile_m[mask]
    maximum = max(float(np.percentile(active_values, 95)), 1.0)
    strength = np.clip(active_values / maximum, 0.0, 1.0)
    strength = 0.35 + 0.65 * strength
    image[mask] = np.clip(base_colour[None, :] * strength[:, None], 0, 255).astype(np.uint8)


def _transport_image(
    road_mask: torch.Tensor,
    rail_mask: torch.Tensor,
    road_profile_m: torch.Tensor,
    rail_profile_m: torch.Tensor,
) -> Image.Image:
    road = road_mask.detach().cpu().numpy().astype(bool)
    rail = rail_mask.detach().cpu().numpy().astype(bool)
    road_profile = road_profile_m.detach().cpu().numpy().astype(np.float32)
    rail_profile = rail_profile_m.detach().cpu().numpy().astype(np.float32)
    height, width = road.shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = BACKGROUND
    _apply_profile_colour(image, road, road_profile, ROAD_COLOUR)
    _apply_profile_colour(image, rail, rail_profile, RAIL_COLOUR)
    return Image.fromarray(image)


def render_triptych(values: torch.Tensor, valid_mask: torch.Tensor | None = None) -> Image.Image:
    decoded = model_space_to_layers(values)
    surface_image = _surface_image(decoded["surface"], valid_mask)
    underground = _transport_image(
        decoded["road_underground"],
        decoded["rail_underground"],
        decoded["road_underground_depth_m"],
        decoded["rail_underground_depth_m"],
    )
    elevated = _transport_image(
        decoded["road_elevated"],
        decoded["rail_elevated"],
        decoded["road_elevated_height_m"],
        decoded["rail_elevated_height_m"],
    )

    width, height = surface_image.size
    header = 18
    canvas = Image.new("RGB", (width * 3, height + header), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(
        (("surface", surface_image), ("underground depth", underground), ("elevated height", elevated))
    ):
        canvas.paste(image, (index * width, header))
        draw.text((index * width + 4, 3), name, fill=(0, 0, 0))
    return canvas


def save_sheet(
    values: torch.Tensor,
    path: str | Path,
    *,
    valid_masks: torch.Tensor | None = None,
) -> Path:
    path = Path(path)
    images = [
        render_triptych(
            values[index],
            None if valid_masks is None else valid_masks[index, 0],
        )
        for index in range(values.shape[0])
    ]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    top = 0
    for image in images:
        canvas.paste(image, (0, top))
        top += image.height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
    return path
