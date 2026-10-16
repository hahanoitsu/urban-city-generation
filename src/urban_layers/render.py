from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from urban_layers.data import CHANNELS, LANDUSE, LINE_CLASSES

LANDUSE_COLOURS = {
    "none": (242, 240, 233),
    "residential": (240, 226, 200),
    "commercial": (232, 208, 205),
    "industrial": (218, 212, 228),
    "civic": (222, 230, 205),
    "green": (196, 224, 180),
}
ROAD_WIDTH = {"major": 4, "secondary": 3, "local": 2}


def plan_image(plan, scale=6):
    channel = {name: plan[index] for index, name in enumerate(CHANNELS)}
    roads = np.clip(channel["major"] * 1.5 + channel["secondary"] + channel["local"] * 0.6, 0, 1)
    red = np.clip(0.95 - 0.5 * channel["green"] - 0.6 * channel["water"] + 0.4 * roads, 0, 1)
    green = np.clip(0.93 - 0.4 * channel["water"] - 0.5 * channel["coverage"] - 0.2 * roads, 0, 1)
    blue = np.clip(0.88 - 0.5 * channel["green"] + 0.1 * channel["water"] - 0.6 * channel["coverage"] - 0.3 * roads, 0, 1)
    image = np.stack([red, green, blue], axis=-1)[::-1]
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image).resize((plan.shape[2] * scale, plan.shape[1] * scale), Image.NEAREST)


def city_image(city, size=1024):
    width_m, height_m = city["size_m"]
    scale = size / max(width_m, height_m)
    image = Image.new("RGB", (int(width_m * scale) + 1, int(height_m * scale) + 1), LANDUSE_COLOURS["none"])
    draw = ImageDraw.Draw(image)

    def points(values):
        return [(float(x) * scale, (height_m - float(y)) * scale) for x, y in values]

    for block in city.get("blocks", []):
        draw.polygon(points(block["polygon"]), fill=LANDUSE_COLOURS[block.get("landuse", "none")])
    for area in city.get("green", []):
        draw.polygon(points(area["polygon"]), fill=LANDUSE_COLOURS["green"])
    for area in city.get("water", []):
        draw.polygon(points(area["polygon"]), fill=(160, 200, 230))
    for building in city.get("buildings", []):
        shade = int(np.clip(170 - building["height_m"] * 0.8, 70, 170))
        draw.polygon(points(building["footprint"]), fill=(shade, shade, shade + 8), outline=(80, 80, 90))
    for road in city.get("roads", []):
        colour = (210, 90, 60) if road["vertical"] == "surface" else (150, 70, 160)
        draw.line(points(road["points"]), fill=colour, width=ROAD_WIDTH.get(road["class"], 2))
    for rail in city.get("rail", []):
        colour = (40, 120, 190) if rail["vertical"] != "underground" else (120, 160, 200)
        draw.line(points(rail["points"]), fill=colour, width=3)
    return image


def tile_image(lines, context=(), ports=(), size=512, extent=512.0, margin=128.0):
    scale = size / (extent + 2 * margin)
    image = Image.new("RGB", (size, size), (246, 244, 238))
    draw = ImageDraw.Draw(image)

    def points(values):
        return [((float(x) + margin) * scale, (extent + margin - float(y)) * scale) for x, y in values]

    draw.rectangle(points([(0, extent), (extent, 0)]), outline=(150, 150, 150))
    for class_id, values in context:
        draw.line(points(values), fill=(170, 170, 170), width=2)
    for class_id, values in lines:
        mode, name, _level = LINE_CLASSES[class_id]
        colour = (40, 120, 190) if mode == "rail" else (210, 90, 60)
        draw.line(points(values), fill=colour, width=ROAD_WIDTH.get(name, 3))
    for x, y, *_rest in ports:
        px, py = points([(x, y)])[0]
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=(20, 20, 20))
    return image


def landuse_name(index):
    return LANDUSE[index] if 0 <= index < len(LANDUSE) else "none"
