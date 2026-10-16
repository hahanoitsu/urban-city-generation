from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import torch
from rasterio import features
from rasterio.transform import from_origin
from shapely import affinity
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiLineString, Polygon, box
from shapely.ops import linemerge, polygonize, unary_union
from shapely.strtree import STRtree

from urban_dataset.prepared import load_city_gpkg
from urban_model.structured_city_data import BUILDING_KINDS, _building_kind

CELL = 128.0
TILE = 512.0
MARGIN = 128.0
FINE = 8.0
BLOCK_SPLIT = 384.0

CHANNELS = (
    "major",
    "secondary",
    "local",
    "rail",
    "coverage",
    "height",
    "green",
    "water",
    "residential",
    "commercial",
    "industrial",
    "civic",
)
ROAD_CLASSES = ("major", "secondary", "local")
RAIL_CLASSES = ("rail", "subway", "light_rail", "tram", "monorail")
VERTICAL = ("surface", "elevated", "underground", "unknown")
LINE_CLASSES = [("road", name, level) for name in ROAD_CLASSES for level in VERTICAL] + [
    ("rail", name, level) for name in RAIL_CLASSES for level in VERTICAL
]
LANDUSE = ("none", "residential", "commercial", "industrial", "civic", "green")
LANDUSE_SOURCE = {"residential": 1, "commercial_mixed": 2, "industrial": 3, "civic": 4, "green": 5}


def line_class(mode, name, level):
    names = ROAD_CLASSES if mode == "road" else RAIL_CLASSES
    name = name if name in names else names[-1] if mode == "road" else "rail"
    level = level if level in VERTICAL else "unknown"
    return LINE_CLASSES.index((mode, name, level))


def is_road(class_id):
    return LINE_CLASSES[class_id][0] == "road"


def blocks_ground(class_id):
    mode, _name, level = LINE_CLASSES[class_id]
    return level in ("surface", "unknown")


def _text(frame, column, fallback):
    if column not in frame.columns:
        return pd.Series([fallback] * len(frame), index=frame.index)
    return frame[column].fillna(fallback).astype(str).str.strip().str.lower()


def _numbers(frame, column):
    if column not in frame.columns:
        return pd.Series([np.nan] * len(frame), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def city_lines(layers):
    lines = []
    for mode, frame in (("road", layers.roads), ("rail", layers.rail)):
        if frame.empty:
            continue
        frame = frame.explode(index_parts=False)
        names = _text(frame, "road_class" if mode == "road" else "railway", "local" if mode == "road" else "rail")
        levels = _text(frame, "vertical_mode", "unknown")
        widths = _numbers(frame, "estimated_width_m")
        for position, geometry in enumerate(frame.geometry):
            if geometry is None or geometry.is_empty or geometry.geom_type != "LineString" or geometry.length < 1.0:
                continue
            class_id = line_class(mode, names.iloc[position], levels.iloc[position])
            width = float(widths.iloc[position]) if pd.notna(widths.iloc[position]) else 0.0
            lines.append((class_id, LineString(np.asarray(geometry.coords)[:, :2]), width))
    return lines


def city_bounds(layers):
    values = [frame.total_bounds for _name, frame in layers.items() if not frame.empty]
    values = np.asarray(values)
    minx, miny = values[:, 0].min(), values[:, 1].min()
    maxx, maxy = values[:, 2].max(), values[:, 3].max()
    minx = math.floor(minx / CELL) * CELL
    miny = math.floor(miny / CELL) * CELL
    maxx = math.ceil(maxx / CELL) * CELL
    maxy = math.ceil(maxy / CELL) * CELL
    return float(minx), float(miny), float(maxx), float(maxy)


def _polygons(frame):
    if frame is None or frame.empty:
        return []
    return [geometry for geometry in frame.geometry if geometry is not None and not geometry.is_empty and geometry.area > 1.0]


def _rasterize(shapes, bounds, dtype="uint8"):
    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / FINE))
    height = int(round((maxy - miny) / FINE))
    if not shapes:
        return np.zeros((height, width), dtype=dtype)
    image = features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=from_origin(minx, maxy, FINE, FINE),
        fill=0,
        dtype=dtype,
    )
    # rasterio rows go north to south, the plan rows go south to north
    return image[::-1]


def _pool(image, bounds):
    factor = int(round(CELL / FINE))
    height = image.shape[0] // factor
    width = image.shape[1] // factor
    return image[: height * factor, : width * factor].reshape(height, factor, width, factor).mean(axis=(1, 3))


def plan_raster(layers, lines, bounds):
    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / CELL))
    height = int(round((maxy - miny) / CELL))
    plan = np.zeros((len(CHANNELS), height, width), dtype=np.float32)

    for class_id in range(len(LINE_CLASSES)):
        geometries = [geometry for item_class, geometry, _width in lines if item_class == class_id]
        if not geometries:
            continue
        mode, name, _level = LINE_CLASSES[class_id]
        channel = CHANNELS.index("rail" if mode == "rail" else name)
        dense = shapely.segmentize(np.asarray(geometries, dtype=object), 8.0)
        coords, index = shapely.get_coordinates(dense, return_index=True)
        same = index[1:] == index[:-1]
        start = coords[:-1][same]
        end = coords[1:][same]
        middle = (start + end) / 2.0
        length = np.linalg.norm(end - start, axis=1)
        column = np.floor((middle[:, 0] - minx) / CELL).astype(int)
        row = np.floor((middle[:, 1] - miny) / CELL).astype(int)
        keep = (column >= 0) & (column < width) & (row >= 0) & (row < height)
        np.add.at(plan[channel], (row[keep], column[keep]), length[keep])
    plan[:4] = np.clip(plan[:4] / (2.0 * CELL), 0.0, 3.0)

    buildings = layers.buildings
    footprints = []
    heights = []
    if not buildings.empty:
        values = _numbers(buildings, "estimated_height_m")
        for position, geometry in enumerate(buildings.geometry):
            if geometry is None or geometry.is_empty:
                continue
            height_m = float(values.iloc[position]) if pd.notna(values.iloc[position]) else 9.0
            footprints.append(geometry)
            heights.append(max(height_m, 1.0))
    mask = _rasterize([(geometry, 1) for geometry in footprints], bounds)
    tall = _rasterize([(geometry, value) for geometry, value in zip(footprints, heights)], bounds, "float32")
    coverage = _pool(mask.astype(np.float32), bounds)
    summed = _pool(tall, bounds)
    plan[CHANNELS.index("coverage")] = coverage[:height, :width]
    plan[CHANNELS.index("height")] = np.where(coverage > 0, summed / np.maximum(coverage, 1e-6), 0.0)[:height, :width] / 100.0

    landuse = layers.landuse
    classes = _text(landuse, "landuse_class", "") if not landuse.empty else None
    green = _polygons(layers.green)
    if classes is not None:
        green += [geometry for geometry, name in zip(landuse.geometry, classes) if name == "green" and geometry is not None]
    plan[CHANNELS.index("green")] = _pool(_rasterize([(geometry, 1) for geometry in green], bounds).astype(np.float32), bounds)[:height, :width]
    plan[CHANNELS.index("water")] = _pool(
        _rasterize([(geometry, 1) for geometry in _polygons(layers.water)], bounds).astype(np.float32), bounds
    )[:height, :width]
    for channel, source in (("residential", "residential"), ("commercial", "commercial_mixed"), ("industrial", "industrial"), ("civic", "civic")):
        shapes = []
        if classes is not None:
            shapes = [(geometry, 1) for geometry, name in zip(landuse.geometry, classes) if name == source and geometry is not None]
        plan[CHANNELS.index(channel)] = _pool(_rasterize(shapes, bounds).astype(np.float32), bounds)[:height, :width]
    return plan


def _parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if hasattr(geometry, "geoms"):
        result = []
        for part in geometry.geoms:
            result.extend(_parts(part))
        return result
    return []


def boundary_side(point, size=TILE, eps=0.05):
    x, y = point
    if abs(x) < eps:
        return 0
    if abs(y) < eps:
        return 1
    if abs(x - size) < eps:
        return 2
    if abs(y - size) < eps:
        return 3
    return -1


def ring_box(dx, dy):
    x0 = -MARGIN if dx < 0 else (0.0 if dx == 0 else TILE)
    x1 = 0.0 if dx < 0 else (TILE if dx == 0 else TILE + MARGIN)
    y0 = -MARGIN if dy < 0 else (0.0 if dy == 0 else TILE)
    y1 = 0.0 if dy < 0 else (TILE if dy == 0 else TILE + MARGIN)
    return box(x0, y0, x1, y1)


def ring_boxes():
    return [ring_box(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]


RING = ring_boxes()


def touch_lists(lines, reach=2.5):
    # for both ends of every line, the other lines of the same mode that end touches
    geometries = [LineString(np.asarray(item[1], dtype=np.float64)) for item in lines]
    tree = STRtree(geometries)
    modes = [LINE_CLASSES[item[0]][0] for item in lines]
    result = []
    for index, item in enumerate(lines):
        ends = []
        for point in (item[1][0], item[1][-1]):
            found = tree.query(shapely.Point(float(point[0]), float(point[1])), predicate="dwithin", distance=reach)
            ends.append(sorted(int(other) for other in found if other != index and modes[other] == modes[index]))
        result.append(ends)
    return result


def street_tiles(lines, bounds, stride=TILE, minimum_length=40.0):
    minx, miny, maxx, maxy = bounds
    geometries = [geometry for _class_id, geometry, _width in lines]
    tree = STRtree(geometries)
    tile_box = box(0.0, 0.0, TILE, TILE)
    tiles = []
    for y0 in np.arange(miny, maxy - TILE / 2, stride):
        for x0 in np.arange(minx, maxx - TILE / 2, stride):
            window = box(x0 - MARGIN, y0 - MARGIN, x0 + TILE + MARGIN, y0 + TILE + MARGIN)
            found = tree.query(window, predicate="intersects")
            if len(found) == 0:
                continue
            grouped = {}
            for index in found:
                class_id = lines[index][0]
                clipped = affinity.translate(lines[index][1].intersection(window), -x0, -y0)
                grouped.setdefault(class_id, []).extend(_parts(clipped))
            targets = []
            context = []
            inside_length = 0.0
            for class_id, parts in grouped.items():
                merged = _parts(linemerge(MultiLineString(parts))) if len(parts) > 1 else parts
                for line in merged:
                    for piece in _parts(line.intersection(tile_box)):
                        piece = piece.simplify(1.0)
                        if piece.length < 2.0:
                            continue
                        inside_length += piece.length
                        targets.append((class_id, np.asarray(piece.coords, dtype=np.float32)[:, :2]))
                    outside = line.difference(tile_box)
                    for ring_index, ring in enumerate(RING):
                        for piece in _parts(outside.intersection(ring)):
                            piece = piece.simplify(2.0)
                            if piece.length < 2.0:
                                continue
                            context.append((class_id, np.asarray(piece.coords, dtype=np.float32)[:, :2], ring_index))
            if inside_length < minimum_length:
                continue
            tiles.append(
                {
                    "origin": (float(x0), float(y0)),
                    "cell": (int(round((y0 - miny) / CELL)), int(round((x0 - minx) / CELL))),
                    "targets": targets,
                    "context": context,
                    "touches": touch_lists(targets),
                }
            )
    return tiles


def block_frame(polygon):
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        rectangle = box(*polygon.bounds).buffer(0.5)
    coords = np.asarray(rectangle.exterior.coords)[:4]
    edges = np.diff(np.vstack([coords, coords[:1]]), axis=0)
    longest = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    angle = math.atan2(longest[1], longest[0])
    if angle <= -math.pi / 2:
        angle += math.pi
    if angle > math.pi / 2:
        angle -= math.pi
    centre = np.asarray(rectangle.centroid.coords[0])
    return centre, angle


def to_frame(points, centre, angle):
    cos, sin = math.cos(-angle), math.sin(-angle)
    shifted = np.asarray(points, dtype=np.float64) - centre
    return np.stack([shifted[:, 0] * cos - shifted[:, 1] * sin, shifted[:, 0] * sin + shifted[:, 1] * cos], axis=1)


def from_frame(points, centre, angle):
    cos, sin = math.cos(angle), math.sin(angle)
    points = np.asarray(points, dtype=np.float64)
    return np.stack([points[:, 0] * cos - points[:, 1] * sin, points[:, 0] * sin + points[:, 1] * cos], axis=1) + centre


def box_of(polygon, centre, angle):
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.geom_type != "Polygon":
        return None
    coords = to_frame(np.asarray(rectangle.exterior.coords)[:4], centre, angle)
    first = coords[1] - coords[0]
    second = coords[2] - coords[1]
    width, depth = np.linalg.norm(first), np.linalg.norm(second)
    along = first if width >= depth else second
    width, depth = max(width, depth), min(width, depth)
    turn = math.atan2(along[1], along[0])
    if turn < -math.pi / 2:
        turn += math.pi
    if turn >= math.pi / 2:
        turn -= math.pi
    middle = coords.mean(axis=0)
    return float(middle[0]), float(middle[1]), float(width), float(depth), float(turn)


def box_polygon(x, y, width, depth, turn):
    along = np.array([math.cos(turn), math.sin(turn)]) * width / 2.0
    across = np.array([-math.sin(turn), math.cos(turn)]) * depth / 2.0
    centre = np.array([x, y])
    return np.stack([centre - along - across, centre + along - across, centre + along + across, centre - along + across])


def split_block(polygon, size=BLOCK_SPLIT):
    centre, angle = block_frame(polygon)
    local = Polygon(to_frame(np.asarray(polygon.exterior.coords), centre, angle), [to_frame(np.asarray(ring.coords), centre, angle) for ring in polygon.interiors])
    minx, miny, maxx, maxy = local.bounds
    if maxx - minx <= size and maxy - miny <= size:
        return [polygon]
    columns = max(1, math.ceil((maxx - minx) / size))
    rows = max(1, math.ceil((maxy - miny) / size))
    step_x = (maxx - minx) / columns
    step_y = (maxy - miny) / rows
    pieces = []
    for row in range(rows):
        for column in range(columns):
            cell = box(minx + column * step_x, miny + row * step_y, minx + (column + 1) * step_x, miny + (row + 1) * step_y)
            part = local.intersection(cell)
            for item in getattr(part, "geoms", [part]):
                if item.geom_type == "Polygon" and item.area > 50.0:
                    pieces.append(Polygon(from_frame(np.asarray(item.exterior.coords), centre, angle)))
    return pieces


def make_blocks(lines, water=None, minimum_area=150.0, edge=None, split=True):
    barriers = [geometry for class_id, geometry, _width in lines if blocks_ground(class_id)]
    if edge is not None:
        barriers.append(edge)
    if not barriers:
        return []
    merged = _parts(unary_union(barriers))
    # only cut each block with the water parts that touch it, a whole city of water is slow
    water = [] if water is None or water.is_empty else list(getattr(water, "geoms", [water]))
    water_tree = STRtree(water) if water else None
    blocks = []
    for polygon in polygonize(merged):
        if polygon.area < minimum_area:
            continue
        shrunk = polygon.buffer(-3.0)
        if water_tree is not None:
            touching = water_tree.query(shrunk, predicate="intersects")
            if len(touching):
                shrunk = shrunk.difference(unary_union([water[index] for index in touching]))
        for part in getattr(shrunk, "geoms", [shrunk]):
            if part.geom_type != "Polygon" or part.area < minimum_area:
                continue
            blocks.extend(split_block(part) if split else [part])
    return blocks


def ring_points(polygon, centre, angle, count=48):
    ring = LineString(to_frame(np.asarray(polygon.exterior.coords), centre, angle))
    distances = np.linspace(0.0, ring.length, count, endpoint=False)
    return np.asarray([ring.interpolate(distance).coords[0] for distance in distances], dtype=np.float32)


def plan_at(plan, bounds, point):
    minx, miny, _maxx, _maxy = bounds
    column = int(np.clip((point[0] - minx) // CELL, 0, plan.shape[2] - 1))
    row = int(np.clip((point[1] - miny) // CELL, 0, plan.shape[1] - 1))
    return plan[:, row, column].astype(np.float32)


def frontage(polygon, road_tree, road_classes, count=32, reach=20.0):
    shares = np.zeros(3, dtype=np.float32)
    if road_tree is None:
        return shares
    ring = polygon.exterior
    points = shapely.line_interpolate_point(ring, np.linspace(0.0, ring.length, count, endpoint=False))
    found, nearest = road_tree.query_nearest(points, max_distance=reach, all_matches=False)
    _unique, first = np.unique(found, return_index=True)
    for index in nearest[first]:
        shares[road_classes[index]] += 1.0
    return shares / count


def block_record(polygon, plan, bounds, road_tree, road_classes):
    centre, angle = block_frame(polygon)
    return {
        "centre": centre.astype(np.float32),
        "angle": float(angle),
        "ring": ring_points(polygon, centre, angle),
        "area": float(polygon.area),
        "plan": plan_at(plan, bounds, centre),
        "frontage": frontage(polygon, road_tree, road_classes),
    }


def road_index(lines):
    roads = [(geometry, ROAD_CLASSES.index(LINE_CLASSES[class_id][1])) for class_id, geometry, _width in lines if is_road(class_id)]
    if not roads:
        return None, []
    return STRtree([item[0] for item in roads]), [item[1] for item in roads]


def block_records(layers, lines, plan, bounds):
    water = unary_union(_polygons(layers.water)) if not layers.water.empty else None
    blocks = make_blocks(lines, water)
    road_tree, road_classes = road_index(lines)
    buildings = []
    if road_tree is None:
        return [], 0, len(layers.buildings)
    if not layers.buildings.empty:
        values = _numbers(layers.buildings, "estimated_height_m")
        kinds = _text(layers.buildings, "building", "yes")
        for position, geometry in enumerate(layers.buildings.geometry):
            if geometry is None or geometry.is_empty or geometry.area < 4.0:
                continue
            height_m = float(values.iloc[position]) if pd.notna(values.iloc[position]) else 0.0
            buildings.append((geometry, height_m, _building_kind(kinds.iloc[position])))
    building_tree = STRtree([item[0].representative_point() for item in buildings]) if buildings else None

    landuse = []
    if not layers.landuse.empty:
        names = _text(layers.landuse, "landuse_class", "")
        for geometry, name in zip(layers.landuse.geometry, names):
            if geometry is not None and name in LANDUSE_SOURCE:
                landuse.append((geometry, LANDUSE_SOURCE[name]))
    for geometry in _polygons(layers.green):
        landuse.append((geometry, LANDUSE.index("green")))
    landuse_tree = STRtree([item[0] for item in landuse]) if landuse else None

    records = []
    assigned = 0
    for polygon in blocks:
        record = block_record(polygon, plan, bounds, road_tree, road_classes)
        centre, angle = record["centre"], record["angle"]
        items = []
        if building_tree is not None:
            for index in building_tree.query(polygon, predicate="contains"):
                geometry, height_m, kind = buildings[index]
                shape = box_of(geometry, centre, angle)
                if shape is None:
                    continue
                items.append((kind, *shape, height_m))
        assigned += len(items)
        use = 0
        if landuse_tree is not None:
            share = np.zeros(len(LANDUSE))
            for index in landuse_tree.query(polygon, predicate="intersects"):
                geometry, value = landuse[index]
                try:
                    share[value] += geometry.intersection(polygon).area
                except GEOSException:
                    continue
            if share.max() > 0.3 * polygon.area:
                use = int(np.argmax(share))
        record["landuse"] = use
        record["buildings"] = np.asarray(items, dtype=np.float32).reshape(-1, 7)
        records.append(record)
    return records, assigned, len(buildings)


def build_city(gpkg, name, output, stride=TILE):
    layers, _metadata = load_city_gpkg(gpkg)
    lines = city_lines(layers)
    bounds = city_bounds(layers)
    plan = plan_raster(layers, lines, bounds)
    tiles = street_tiles(lines, bounds, stride)
    blocks, assigned, total = block_records(layers, lines, plan, bounds)
    folder = Path(output) / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "plan.npz", plan=plan, bounds=np.asarray(bounds))
    torch.save(tiles, folder / "streets.pt")
    torch.save(blocks, folder / "blocks.pt")
    summary = {
        "name": name,
        "source": str(gpkg),
        "bounds": bounds,
        "plan_shape": list(plan.shape),
        "channels": list(CHANNELS),
        "lines": len(lines),
        "tiles": len(tiles),
        "blocks": len(blocks),
        "buildings_in_blocks": assigned,
        "buildings": total,
        "channel_means": {channel: float(plan[index].mean()) for index, channel in enumerate(CHANNELS)},
    }
    summary.update(token_lengths(tiles, blocks))
    (folder / "meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def token_lengths(tiles, blocks):
    # pick --max-len from these, anything longer gets cut when training
    from urban_layers.tokens import BuildingTokens, StreetTokens

    streets = StreetTokens()
    buildings = BuildingTokens()
    street_lengths = [len(streets.encode(tile["targets"])) for tile in tiles] or [0]
    building_lengths = [len(buildings.encode(block["landuse"], block["buildings"])) for block in blocks] or [0]
    result = {}
    for name, values in (("street_tokens", street_lengths), ("building_tokens", building_lengths)):
        result[name] = {key: int(np.percentile(values, value)) for key, value in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}
    return result


def load_cities(root):
    root = Path(root)
    cities = []
    for folder in sorted(path for path in root.iterdir() if (path / "meta.json").exists()):
        plan = np.load(folder / "plan.npz")
        cities.append(
            {
                "name": folder.name,
                "plan": plan["plan"],
                "bounds": tuple(plan["bounds"].tolist()),
                "folder": folder,
            }
        )
    return cities
