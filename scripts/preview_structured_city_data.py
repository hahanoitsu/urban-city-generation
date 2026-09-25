from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape

from urban_model.structured_city_data import SceneTensorConfig, encode_scene


def polygons(geometry):
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon | GeometryCollection):
        for part in geometry.geoms:
            yield from polygons(part)


def point(x, y, size):
    return (
        int(round(float(x) / 512.0 * (size - 1))),
        int(round((1.0 - float(y) / 512.0) * (size - 1))),
    )


def draw_polygon(draw, geometry, size, fill, outline=None):
    for polygon in polygons(geometry):
        values = [point(x, y, size) for x, y in polygon.exterior.coords]
        if len(values) >= 3:
            draw.polygon(values, fill=fill, outline=outline)


def render_raw(payload, size=640):
    image = Image.new("RGB", (size, size), (247, 245, 238))
    draw = ImageDraw.Draw(image)

    landuse_colours = {
        "green": (180, 213, 169),
        "residential": (238, 225, 197),
        "commercial_mixed": (225, 207, 190),
        "industrial": (211, 207, 197),
        "civic": (219, 213, 190),
    }
    for record in payload["target"].get("landuse", []):
        colour = landuse_colours.get(str(record.get("class")), (234, 227, 201))
        draw_polygon(draw, shape(record["geometry_local_m"]), size, colour)
    for record in payload["target"].get("green", []):
        draw_polygon(draw, shape(record["geometry_local_m"]), size, (180, 213, 169))
    for record in payload["target"].get("water", []):
        draw_polygon(draw, shape(record["geometry_local_m"]), size, (159, 202, 226))
    for record in payload["target"].get("buildings", []):
        draw_polygon(
            draw,
            shape(record["footprint_local_m"]),
            size,
            (158, 158, 158),
            (105, 105, 105),
        )

    road_width = {"major": 5, "secondary": 3, "local": 2}
    for record in payload["target"].get("roads", []):
        values = [point(x, y, size) for x, y in record["geometry_local_m"]]
        if len(values) >= 2:
            draw.line(
                values,
                fill=(202, 78, 58),
                width=road_width.get(str(record["class"]), 2),
                joint="curve",
            )
    for record in payload["target"].get("rail", []):
        values = [point(x, y, size) for x, y in record["geometry_local_m"]]
        if len(values) >= 2:
            draw.line(values, fill=(62, 132, 183), width=2, joint="curve")

    for port in payload["input"].get("boundary_ports", []):
        x, y = port["position_local_m"]
        px, py = point(x, y, size)
        colour = (62, 132, 183) if port["mode"] == "rail" else (25, 25, 25)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], outline=colour, width=2)

    return image


def tensor_point(value, size):
    x = float((value[0] + 1.0) * 0.5 * (size - 1))
    y = float((1.0 - (value[1] + 1.0) * 0.5) * (size - 1))
    return int(round(x)), int(round(y))


def render_tensor(scene, config, size=640):
    image = Image.new("RGB", (size, size), (247, 245, 238))
    draw = ImageDraw.Draw(image)

    colours = {
        0: (180, 213, 169),
        1: (159, 202, 226),
        2: (238, 225, 197),
        3: (225, 207, 190),
        4: (211, 207, 197),
        5: (219, 213, 190),
    }
    for index in range(config.area_slots):
        if int(scene["area_presence"][index]) != 1:
            continue
        kind = int(scene["area_kind"][index])
        values = [tensor_point(value, size) for value in scene["area_shape"][index]]
        if len(values) >= 3 and kind in colours:
            draw.polygon(values, fill=colours[kind])

    for index in range(config.building_slots):
        if int(scene["building_presence"][index]) != 1:
            continue
        values = [tensor_point(value, size) for value in scene["building_shape"][index]]
        if len(values) >= 3:
            draw.polygon(values, fill=(158, 158, 158), outline=(105, 105, 105))

    nodes = scene["node_position"]
    for index in range(config.edge_slots):
        if int(scene["edge_presence"][index]) != 1:
            continue
        left = int(scene["edge_from"][index])
        right = int(scene["edge_to"][index])
        if left == right:
            continue
        start = nodes[left, :2]
        end = nodes[right, :2]
        values = []
        count = config.edge_shape_points
        for shape_index in range(count):
            fraction = shape_index / max(count - 1, 1)
            base = start + (end - start) * fraction
            values.append(tensor_point(base + scene["edge_shape"][index, shape_index, :2], size))
        colour = (202, 78, 58) if int(scene["edge_mode"][index]) == 0 else (62, 132, 183)
        draw.line(values, fill=colour, width=2, joint="curve")

    return image


def choose_rows(rows, count):
    selected = []
    rankings = [
        sorted(rows, key=lambda row: (row.get("rail", 0), row.get("roads", 0)), reverse=True),
        sorted(rows, key=lambda row: row.get("buildings", 0), reverse=True),
        sorted(
            rows,
            key=lambda row: row.get("green", 0) + row.get("water", 0) + row.get("landuse", 0),
            reverse=True,
        ),
    ]
    seen = set()
    for ranking in rankings:
        for row in ranking:
            if row["id"] in seen:
                continue
            selected.append(row)
            seen.add(row["id"])
            if len(selected) >= count:
                return selected
            if len(selected) % 4 == 0:
                break
    stride = max(1, len(rows) // max(count, 1))
    for row in rows[::stride]:
        if row["id"] in seen:
            continue
        selected.append(row)
        seen.add(row["id"])
        if len(selected) >= count:
            break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (args.data / "targets.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = choose_rows(rows, args.samples)
    config = SceneTensorConfig(
        node_slots=448,
        edge_slots=512,
        building_slots=832,
        area_slots=256,
        maximum_ports=96,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    panels = []
    records = []
    for index, row in enumerate(selected):
        with gzip.open(args.data / row["sample_path"], "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw = render_raw(payload)
        scene = encode_scene(payload, config)
        encoded = render_tensor(scene, config)
        panel = Image.new("RGB", (1280, 680), "white")
        panel.paste(raw, (0, 40))
        panel.paste(encoded, (640, 40))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 10), f"{row['id']} raw vectors", fill="black")
        draw.text((648, 10), "model tensor representation", fill="black")
        path = args.output / f"{index:02d}-{row['id']}.png"
        panel.save(path)
        panels.append(panel)
        records.append(
            {
                "sample_id": row["id"],
                "roads": row.get("roads", 0),
                "rail": row.get("rail", 0),
                "buildings": row.get("buildings", 0),
                "green": row.get("green", 0),
                "water": row.get("water", 0),
                "landuse": row.get("landuse", 0),
            }
        )

    sheet = Image.new("RGB", (1280, 680 * len(panels)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 680))
    sheet.save(args.output / "structured-targets.png")
    (args.output / "summary.json").write_text(json.dumps({"samples": records}, indent=2) + "\n")
    print(json.dumps({"samples": records}, indent=2))


if __name__ == "__main__":
    main()
