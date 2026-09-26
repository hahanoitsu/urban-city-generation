from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape


def polygons(geometry):
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon | GeometryCollection):
        for part in geometry.geoms:
            yield from polygons(part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (args.data / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(
        key=lambda row: (
            row["visible_rail"] + row["raw_edges"],
            row["buildings"],
        ),
        reverse=True,
    )
    stride = max(1, len(rows) // max(args.samples, 1))
    selected = rows[::stride][: args.samples]

    args.output.mkdir(parents=True, exist_ok=True)
    panels = []
    for sample_index, row in enumerate(selected):
        with gzip.open(args.data / row["sample_path"], "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        context_bounds = payload["bounds"]["context_projected_m"]
        target_bounds = payload["bounds"]["target_projected_m"]
        target_origin = payload["coordinate_system"]["target_origin_projected_m"]
        width_m = context_bounds[2] - context_bounds[0]
        size = 900
        scale = size / width_m

        def point_projected(x, y):
            return (
                int(round((x - context_bounds[0]) * scale)),
                int(round((context_bounds[3] - y) * scale)),
            )

        def point_local(x, y):
            return point_projected(
                target_origin[0] + x,
                target_origin[1] + y,
            )

        image = Image.new("RGB", (size, size), (247, 246, 242))
        draw = ImageDraw.Draw(image)

        for record in payload["input"]["visible_transport"]["roads"]:
            values = [point_local(x, y) for x, y in record["geometry_local_m"]]
            draw.line(values, fill=(150, 150, 150), width=1)
        for record in payload["input"]["visible_transport"]["rail"]:
            values = [point_local(x, y) for x, y in record["geometry_local_m"]]
            draw.line(values, fill=(120, 165, 195), width=1)

        for record in payload["target"]["buildings"]:
            geometry = shape(record["footprint_local_m"])
            for polygon in polygons(geometry):
                values = [point_local(x, y) for x, y in polygon.exterior.coords]
                draw.polygon(values, fill=(205, 205, 205))

        graph = payload["target"]["transport_graph"]
        for edge in graph["edges"]:
            values = [
                point_local(float(value[0]), float(value[1]))
                for value in edge["geometry_local_m"]
            ]
            colour = (
                (205, 75, 55)
                if edge["transport_mode"] == "road"
                else (55, 125, 185)
            )
            draw.line(values, fill=colour, width=3)

        left, bottom = point_projected(target_bounds[0], target_bounds[1])
        right, top = point_projected(target_bounds[2], target_bounds[3])
        draw.rectangle([left, top, right, bottom], outline=(20, 20, 20), width=3)

        title = (
            f"{row['id']}  nodes={row['nodes']} edges={row['edges']} "
            f"buildings={row['buildings']} visible={row['visible_roads'] + row['visible_rail']}"
        )
        draw.rectangle([0, 0, size, 28], fill=(255, 255, 255))
        draw.text((8, 8), title, fill=(0, 0, 0))
        path = args.output / f"{sample_index:02d}-{row['id']}.png"
        image.save(path)
        panels.append(image)

    sheet = Image.new("RGB", (900, 900 * len(panels)), "white")
    for index, image in enumerate(panels):
        sheet.paste(image, (0, index * 900))
    sheet.save(args.output / "spatial-world-previews.png")


if __name__ == "__main__":
    main()
