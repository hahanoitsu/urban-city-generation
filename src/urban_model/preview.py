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
ROAD_COLOURS = {
    "major": np.asarray((215, 58, 48), dtype=np.uint8),
    "secondary": np.asarray((239, 132, 57), dtype=np.uint8),
    "local": np.asarray((246, 224, 125), dtype=np.uint8),
}
RAIL_COLOUR = np.asarray((85, 176, 194), dtype=np.uint8)


def _surface_image(surface: torch.Tensor, valid_mask: torch.Tensor | None = None) -> Image.Image:
    array = surface.detach().cpu().numpy().astype(np.int64)
    image = SURFACE_PALETTE[array]
    if valid_mask is not None:
        valid = valid_mask.detach().cpu().numpy() > 0.5
        image[~valid] = (17, 23, 26)
    return Image.fromarray(image)


def _transport_image(
    road_mask: torch.Tensor,
    rail_mask: torch.Tensor,
    *,
    surface: torch.Tensor,
) -> Image.Image:
    road = road_mask.detach().cpu().numpy() > 0
    rail = rail_mask.detach().cpu().numpy() > 0
    surface_array = surface.detach().cpu().numpy()
    height, width = road.shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (17, 23, 26)

    major = road & (surface_array == 3)
    secondary = road & (surface_array == 4)
    local = road & ~(major | secondary)
    image[local] = ROAD_COLOURS["local"]
    image[secondary] = ROAD_COLOURS["secondary"]
    image[major] = ROAD_COLOURS["major"]
    image[rail] = RAIL_COLOUR
    return Image.fromarray(image)


def render_triptych(values: torch.Tensor, valid_mask: torch.Tensor | None = None) -> Image.Image:
    decoded = model_space_to_layers(values)
    surface = decoded["surface"]
    surface_image = _surface_image(surface, valid_mask)
    underground = _transport_image(
        decoded["road_underground"],
        decoded["rail_underground"],
        surface=surface,
    )
    elevated = _transport_image(
        decoded["road_elevated"],
        decoded["rail_elevated"],
        surface=surface,
    )

    width, height = surface_image.size
    header = 18
    canvas = Image.new("RGB", (width * 3, height + header), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(
        (("surface", surface_image), ("underground", underground), ("elevated", elevated))
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
