from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import shape

from urban_dataset.context_graph import ContextGraphConfig, build_context_graph


def _project(
    x: float,
    y: float,
    bounds: list[float],
    size: int,
    padding: int,
) -> tuple[int, int]:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    scale = (size - padding * 2) / max(width, height)
    px = int(round((x - minx) * scale + padding))
    py = int(round((maxy - y) * scale + padding))
    return px, py


def render_context_graph(output: Path) -> Path:
    graph = json.loads((output / "context-graph.json").read_text(encoding="utf-8"))
    bounds = [float(value) for value in graph["city_bounds_projected_m"]]
    size = 1600
    padding = 30
    image = Image.new("RGB", (size, size), (247, 246, 241))
    draw = ImageDraw.Draw(image)

    nodes = {node["id"]: node for node in graph["nodes"]}
    for edge in graph["edges"]:
        left = nodes[edge["from"]]["center_projected_m"]
        right = nodes[edge["to"]]["center_projected_m"]
        start = _project(left[0], left[1], bounds, size, padding)
        end = _project(right[0], right[1], bounds, size, padding)

        if edge["rail_port_count"] and edge["road_port_count"]:
            colour = (133, 91, 163)
            width = 4
        elif edge["rail_port_count"]:
            colour = (85, 176, 194)
            width = 4
        elif edge["road_port_count"]:
            colour = (215, 82, 58)
            width = 3
        else:
            colour = (205, 205, 198)
            width = 1
        draw.line([start, end], fill=colour, width=width)

    for node in graph["nodes"]:
        x, y = _project(
            node["center_projected_m"][0],
            node["center_projected_m"][1],
            bounds,
            size,
            padding,
        )
        building = float(node["features"]["building_coverage"])
        radius = 3 + min(7, int(round(building * 20)))
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(58, 63, 70),
            outline=(255, 255, 255),
        )

    path = output / "context-graph-preview.png"
    image.save(path, optimize=True)
    return path


def _sample_image(payload: dict, size: int = 300) -> Image.Image:
    target_bounds = payload["target_bounds_projected_m"]
    width_m = float(target_bounds[2] - target_bounds[0])
    height_m = float(target_bounds[3] - target_bounds[1])
    scale = size / max(width_m, height_m, 1.0)

    image = Image.new("RGB", (size, size), (239, 237, 226))
    draw = ImageDraw.Draw(image)

    for building in payload["target"]["buildings"]:
        geometry = shape(building["footprint_local_m"])
        if geometry.is_empty:
            continue
        parts = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
        for polygon in parts:
            points = [
                (int(round(x * scale)), int(round(size - y * scale)))
                for x, y in polygon.exterior.coords
            ]
            draw.polygon(points, fill=(132, 137, 143))

    colours = {
        "major": (215, 58, 48),
        "secondary": (239, 116, 66),
        "local": (246, 180, 90),
    }
    for road in payload["target"]["roads"]:
        points = [
            (int(round(x * scale)), int(round(size - y * scale)))
            for x, y in road["geometry_local_m"]
        ]
        draw.line(
            points,
            fill=colours.get(road["class"], colours["local"]),
            width=max(1, int(round(float(road.get("width_m", 5.0)) * scale))),
            joint="curve",
        )

    for rail in payload["target"]["rail"]:
        points = [
            (int(round(x * scale)), int(round(size - y * scale)))
            for x, y in rail["geometry_local_m"]
        ]
        draw.line(points, fill=(85, 176, 194), width=max(2, int(round(4 * scale))))

    for port in payload["input"]["boundary_ports"]:
        x, y = port["position_local_m"]
        px = int(round(x * scale))
        py = int(round(size - y * scale))
        colour = (85, 176, 194) if port["mode"] == "rail" else (32, 32, 32)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=colour, outline=(255, 255, 255))

    return image


def render_targets(output: Path, limit: int = 12) -> Path | None:
    rows = [
        json.loads(line)
        for line in (output / "targets.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return None

    rows.sort(
        key=lambda row: (
            row["rail"] > 0,
            row["boundary_ports"],
            row["transport_length_m"],
        ),
        reverse=True,
    )
    rows = rows[:limit]

    tile = 300
    header = 36
    columns = 4
    grid_rows = math.ceil(len(rows) / columns)
    canvas = Image.new(
        "RGB",
        (columns * tile, grid_rows * (tile + header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for index, row in enumerate(rows):
        with gzip.open(output / row["sample_path"], "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        preview = _sample_image(payload, tile)
        grid_row, column = divmod(index, columns)
        x = column * tile
        y = grid_row * (tile + header)
        draw.text(
            (x + 5, y + 4),
            f"{row['id']}  ports={row['boundary_ports']} rail={row['rail']}",
            fill="black",
        )
        canvas.paste(preview, (x, y + header))

    path = output / "target-preview.png"
    canvas.save(path, optimize=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--region-size-m", type=float, default=2048.0)
    parser.add_argument("--target-size-m", type=float, default=512.0)
    parser.add_argument("--target-stride-m", type=float, default=512.0)
    parser.add_argument("--minimum-transport-length-m", type=float, default=40.0)
    parser.add_argument("--no-diagonal-region-edges", action="store_true")
    args = parser.parse_args()

    config = ContextGraphConfig(
        region_size_m=args.region_size_m,
        target_size_m=args.target_size_m,
        target_stride_m=args.target_stride_m,
        minimum_transport_length_m=args.minimum_transport_length_m,
        include_diagonal_region_edges=not args.no_diagonal_region_edges,
    )

    summary = build_context_graph(
        args.city,
        args.output,
        config=config,
    )
    graph_preview = render_context_graph(args.output)
    target_preview = render_targets(args.output)

    print(json.dumps(summary, indent=2))
    print(f"context graph: {args.output / 'context-graph.json'}")
    print(f"graph preview: {graph_preview}")
    if target_preview is not None:
        print(f"target preview: {target_preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
