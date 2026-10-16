from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from urban_dataset.extract import CityLayers
from urban_dataset.prepared import load_city_gpkg
from urban_layers.audit import audit_city
from urban_layers.city import WIDTHS, lines_to_layers
from urban_layers.data import CELL, LANDUSE, LANDUSE_SOURCE, LINE_CLASSES, city_bounds, city_lines, make_blocks, plan_raster
from urban_layers.plan import urban_cells
from urban_layers.render import city_image

# cut real windows out of a training city and audit them exactly like a generated city,
# so the generated numbers have something real to be compared with


def parts(geometry):
    if geometry.is_empty:
        return []
    return list(getattr(geometry, "geoms", [geometry]))


def outline(polygon, origin):
    return (np.asarray(polygon.exterior.coords)[:, :2] - origin).tolist()


def real_window(layers, lines, bounds):
    minx, miny, maxx, maxy = bounds
    window = box(*bounds)
    origin = np.array([minx, miny])
    local = []
    for class_id, geometry, _width in lines:
        for part in parts(geometry.intersection(window)):
            if part.geom_type == "LineString" and part.length > 2.0:
                local.append((class_id, np.asarray(part.coords)[:, :2] - origin))
    size = (maxx - minx, maxy - miny)
    _roads, _rail, graph = lines_to_layers(local, *size)

    def clipped(frame):
        result = []
        if frame is None or frame.empty:
            return result
        for geometry in frame.geometry:
            if geometry is None or geometry.is_empty or not geometry.intersects(window):
                continue
            for part in parts(geometry.intersection(window)):
                if part.geom_type == "Polygon" and part.area > 50.0:
                    result.append(part)
        return result

    water = clipped(layers.water)
    green = clipped(layers.green)
    local_lines = [(class_id, LineString(points + origin), 0.0) for class_id, points in local]
    blocks = make_blocks(local_lines, unary_union(water) if water else None, edge=window.exterior)
    landuse = []
    if not layers.landuse.empty and "landuse_class" in layers.landuse.columns:
        for geometry, name in zip(layers.landuse.geometry, layers.landuse["landuse_class"].fillna("").astype(str)):
            if geometry is not None and name in LANDUSE_SOURCE and geometry.intersects(window):
                landuse.append((geometry, LANDUSE_SOURCE[name]))
    tree = STRtree([item[0] for item in landuse]) if landuse else None
    block_rows = []
    for polygon in blocks:
        use = 0
        if tree is not None:
            share = np.zeros(len(LANDUSE))
            for index in tree.query(polygon, predicate="intersects"):
                share[landuse[index][1]] += landuse[index][0].intersection(polygon).area
            if share.max() > 0.3 * polygon.area:
                use = int(np.argmax(share))
        block_rows.append({"landuse": LANDUSE[use], "polygon": outline(polygon, origin)})

    buildings = []
    if not layers.buildings.empty:
        column = layers.buildings["estimated_height_m"] if "estimated_height_m" in layers.buildings.columns else pd.Series(np.nan, index=layers.buildings.index)
        heights = pd.to_numeric(column, errors="coerce").fillna(9.0).to_numpy()
        for position, geometry in enumerate(layers.buildings.geometry):
            if geometry is None or geometry.is_empty or geometry.geom_type != "Polygon":
                continue
            if not window.contains(geometry.representative_point()):
                continue
            buildings.append({"kind": "real", "height_m": float(heights[position]), "footprint": outline(geometry, origin)})

    roads = []
    rail = []
    for index, (class_id, points) in enumerate(local):
        mode, name, level = LINE_CLASSES[class_id]
        record = {"id": index, "class": name, "vertical": level, "points": points.tolist()}
        if mode == "road":
            record["width_m"] = WIDTHS.get(name, 6.0)
            roads.append(record)
        else:
            rail.append(record)
    return {
        "size_m": [float(size[0]), float(size[1])],
        "roads": roads,
        "rail": rail,
        "nodes": [{"id": node["id"], "xy": node["position_local_m"][:2]} for node in graph["nodes"]],
        "edges": [{"from": edge["from_node"], "to": edge["to_node"]} for edge in graph["edges"]],
        "blocks": block_rows,
        "buildings": buildings,
        "water": [{"polygon": outline(polygon, origin)} for polygon in water],
        "green": [{"polygon": outline(polygon, origin)} for polygon in green],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=Path, required=True, help="prepared city gpkg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="plan.npz from build_layer_data, saves rasterising the city again")
    parser.add_argument("--size", type=float, default=4096.0, help="window size in metres, same as the generated cities")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    layers, _metadata = load_city_gpkg(args.city)
    lines = city_lines(layers)
    if args.plan is not None:
        saved = np.load(args.plan)
        plan, bounds = saved["plan"], tuple(saved["bounds"].tolist())
    else:
        bounds = city_bounds(layers)
        plan = plan_raster(layers, lines, bounds)
    urban = urban_cells(plan)
    cells = int(round(args.size / CELL))
    rows, columns = urban.shape
    spots = []
    for row in range(0, rows - cells + 1, 4):
        for column in range(0, columns - cells + 1, 4):
            spots.append((urban[row : row + cells, column : column + cells].mean(), row, column))
    spots.sort(reverse=True)
    good = [spot for spot in spots if spot[0] >= 0.8] or spots[: args.count * 4]
    rng = np.random.default_rng(args.seed)
    picked = [good[index] for index in rng.choice(len(good), size=min(args.count, len(good)), replace=False)]

    args.output.mkdir(parents=True, exist_ok=True)
    line_tree = STRtree([geometry for _class_id, geometry, _width in lines])
    reports = []
    for number, (share, row, column) in enumerate(picked):
        minx = bounds[0] + column * CELL
        miny = bounds[1] + row * CELL
        window = (minx, miny, minx + args.size, miny + args.size)
        # only hand over what is near the window, a whole city is slow to loop over
        nearby = CityLayers(
            **{name: frame.cx[minx : minx + args.size, miny : miny + args.size] if not frame.empty else frame for name, frame in layers.items()}
        )
        city = real_window(nearby, [lines[index] for index in line_tree.query(box(*window))], window)
        report = audit_city(city)
        report["window"] = window
        report["urban_share"] = float(share)
        folder = args.output / f"window-{number}"
        folder.mkdir(exist_ok=True)
        (folder / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        city_image(city, 1400).save(folder / "city.png")
        reports.append(report)
        print(json.dumps({"window": number, "network": report["network"]}), flush=True)

    keys = ["components", "lone_lines", "largest_component_share", "dead_ends_per_km", "junction_share", "broken_tile_seams", "road_km_per_km2"]
    summary = {"windows": len(reports), "size_m": args.size}
    summary["network_median"] = {key: float(np.median([report["network"][key] for report in reports])) for key in keys}
    building_keys = ["buildings", "coverage", "median_height_m", "blocks"]
    summary["buildings_median"] = {key: float(np.median([report["buildings"][key] for report in reports])) for key in building_keys}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
