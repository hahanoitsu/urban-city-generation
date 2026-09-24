from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon
from shapely import affinity

from urban_dataset.obj_export import _ObjMesh, _write_materials

from .object3d import (
    AREA,
    BUILDING_LENGTH_SCALE_M,
    BUILDING_WIDTH_SCALE_M,
    CLASS_OFFSET,
    CX,
    CY,
    CZ,
    DX,
    DY,
    DZ,
    GEOMETRY_OFFSET,
    HEIGHT,
    HEIGHT_SCALE_M,
    LENGTH,
    LENGTH_SCALE_M,
    TOKEN_CLASSES,
    TOKEN_TYPES,
    TYPE_OFFSET,
    VERTICAL_MODES,
    VERTICAL_OFFSET,
    WIDTH,
    WIDTH_SCALE_M,
    Z_SCALE_M,
)

COLOURS = {
    "road": (219, 91, 66),
    "rail": (85, 176, 194),
    "building": (120, 124, 132),
}


def _xy(value: float, minimum: float, maximum: float) -> float:
    return minimum + (float(value) + 1.0) * 0.5 * (maximum - minimum)


def decode_tokens(
    tokens: torch.Tensor | np.ndarray,
    *,
    bounds_m: tuple[float, float, float, float] = (0.0, 0.0, 1024.0, 1024.0),
) -> list[dict[str, Any]]:
    if torch.is_tensor(tokens):
        values = tokens.detach().cpu().numpy()
    else:
        values = np.asarray(tokens)

    minx, miny, maxx, maxy = bounds_m
    objects = []
    for index, token in enumerate(values):
        token_type = int(np.argmax(token[TYPE_OFFSET:CLASS_OFFSET]))
        kind = TOKEN_TYPES[token_type]
        if kind == "pad":
            continue

        cx = _xy(token[CX], minx, maxx)
        cy = _xy(token[CY], miny, maxy)
        cz = float(token[CZ]) * Z_SCALE_M
        direction = np.asarray([token[DX], token[DY], token[DZ]], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-4:
            continue
        direction /= norm

        if kind == "building":
            length = max(0.0, float(token[LENGTH])) * BUILDING_LENGTH_SCALE_M
            width = max(0.0, float(token[WIDTH])) * BUILDING_WIDTH_SCALE_M
        else:
            length = max(0.0, float(token[LENGTH])) * LENGTH_SCALE_M
            width = max(0.0, float(token[WIDTH])) * WIDTH_SCALE_M
        height = max(0.0, float(token[HEIGHT])) * HEIGHT_SCALE_M

        if kind in {"road", "rail"}:
            if length < 3.0 or width < 0.5:
                continue
            edge_class = TOKEN_CLASSES[
                int(np.argmax(token[CLASS_OFFSET:VERTICAL_OFFSET]))
            ]
            vertical = VERTICAL_MODES[
                int(np.argmax(token[VERTICAL_OFFSET:GEOMETRY_OFFSET]))
            ]
            if kind == "road" and edge_class not in {"major", "secondary", "local"}:
                edge_class = "local"
            if kind == "rail" and edge_class in {"major", "secondary", "local"}:
                edge_class = "rail"

            half = direction * (length / 2.0)
            start = [cx - half[0], cy - half[1], cz - half[2]]
            end = [cx + half[0], cy + half[1], cz + half[2]]
            objects.append(
                {
                    "id": f"{kind}-{index:04d}",
                    "type": kind,
                    "class": edge_class,
                    "vertical_mode": vertical,
                    "width_m": width,
                    "center_m": [cx, cy, cz],
                    "start_m": start,
                    "end_m": end,
                    "length_m": length,
                }
            )
            continue

        if kind == "building":
            length = max(length, 2.0)
            width = max(width, 2.0)
            height = max(height, 2.0)
            objects.append(
                {
                    "id": f"building-{index:04d}",
                    "type": "building",
                    "center_m": [cx, cy, cz],
                    "direction_xy": [float(direction[0]), float(direction[1])],
                    "length_m": length,
                    "width_m": width,
                    "height_m": height,
                    "footprint_area_hint": float(max(0.0, token[AREA])),
                }
            )
    return objects


def write_scene(
    tokens: torch.Tensor | np.ndarray,
    path: str | Path,
    *,
    seed: int | None = None,
    bounds_m: tuple[float, float, float, float] = (0.0, 0.0, 1024.0, 1024.0),
) -> dict[str, Any]:
    path = Path(path)
    objects = decode_tokens(tokens, bounds_m=bounds_m)
    payload = {
        "format": "urban-learned-object-scene",
        "version": "0.1.0",
        "bounds_m": list(bounds_m),
        "axis_convention": "x-east, y-north, z-up",
        "source": {"kind": "generated", "seed": seed},
        "objects": objects,
        "summary": {
            "roads": sum(obj["type"] == "road" for obj in objects),
            "rail": sum(obj["type"] == "rail" for obj in objects),
            "buildings": sum(obj["type"] == "building" for obj in objects),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _building_polygon(obj: dict[str, Any]) -> Polygon:
    cx, cy, _cz = obj["center_m"]
    length = float(obj["length_m"])
    width = float(obj["width_m"])
    dx, dy = obj["direction_xy"]
    angle = math.degrees(math.atan2(dy, dx))
    polygon = Polygon(
        [
            (-length / 2, -width / 2),
            (length / 2, -width / 2),
            (length / 2, width / 2),
            (-length / 2, width / 2),
        ]
    )
    polygon = affinity.rotate(polygon, angle, origin=(0, 0))
    return affinity.translate(polygon, cx, cy)


def export_obj(scene: dict[str, Any], output: str | Path) -> dict[str, Any]:
    output = Path(output)
    if output.suffix.lower() != ".obj":
        output = output.with_suffix(".obj")

    mesh = _ObjMesh()
    for obj in scene["objects"]:
        if obj["type"] == "building":
            polygon = _building_polygon(obj)
            _cx, _cy, cz = obj["center_m"]
            height = float(obj["height_m"])
            mesh.prism(
                polygon,
                bottom_z=cz - height / 2.0,
                top_z=cz + height / 2.0,
                group=obj["id"],
                material="building",
            )
            continue

        start = obj["start_m"]
        end = obj["end_m"]
        line = LineString([(start[0], start[1]), (end[0], end[1])])
        if line.length <= 1e-5:
            continue
        ribbon = line.buffer(float(obj["width_m"]) / 2.0, cap_style="flat")
        z = (float(start[2]) + float(end[2])) / 2.0
        material = f"{obj['type']}_{obj['vertical_mode']}"
        mesh.prism(
            ribbon,
            bottom_z=z - 0.10,
            top_z=z + 0.10,
            group=obj["id"],
            material=material,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    material = output.with_suffix(".mtl")
    _write_materials(material)
    mesh.write(output, material.name)
    return {
        "obj": str(output),
        "material": str(material),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
    }


def save_topdown(scene: dict[str, Any], output: str | Path, size: int = 1024) -> Path:
    output = Path(output)
    minx, miny, maxx, maxy = [float(value) for value in scene["bounds_m"]]

    image = Image.new("RGB", (size, size), (236, 233, 222))
    draw = ImageDraw.Draw(image)

    def project(x: float, y: float) -> tuple[float, float]:
        px = (x - minx) / max(maxx - minx, 1e-6) * (size - 1)
        py = (1.0 - (y - miny) / max(maxy - miny, 1e-6)) * (size - 1)
        return px, py

    for obj in scene["objects"]:
        if obj["type"] == "building":
            polygon = _building_polygon(obj)
            points = [project(x, y) for x, y in polygon.exterior.coords]
            draw.polygon(points, fill=COLOURS["building"])
            continue

        start = project(obj["start_m"][0], obj["start_m"][1])
        end = project(obj["end_m"][0], obj["end_m"][1])
        pixels_per_metre = size / max(maxx - minx, 1.0)
        line_width = max(1, int(round(float(obj["width_m"]) * pixels_per_metre)))
        draw.line([start, end], fill=COLOURS[obj["type"]], width=line_width)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)
    return output
