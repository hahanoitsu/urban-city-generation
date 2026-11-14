from __future__ import annotations

import json
import math
from dataclasses import replace
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
from shapely.ops import linemerge, nearest_points, polygonize, substring, unary_union
from shapely.strtree import STRtree

from urban_dataset.prepared import load_city_gpkg
from urban_model.structured_city_data import BUILDING_KINDS, _building_kind

CELL = 128.0
TILE = 512.0
MARGIN = 128.0
FINE = 8.0
BLOCK_SPLIT = 384.0
TRUNK = 4.0

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
PATTERN = ("grid_x", "grid_y", "grid_known")
LANDUSE = ("none", "residential", "commercial", "industrial", "civic", "green")
LANDUSE_SOURCE = {"residential": 1, "commercial_mixed": 2, "industrial": 3, "civic": 4, "green": 5}


def line_class(mode, name, level):
    names = ROAD_CLASSES if mode == "road" else RAIL_CLASSES
    name = name if name in names else names[-1] if mode == "road" else "rail"
    level = level if level in VERTICAL else "unknown"
    return LINE_CLASSES.index((mode, name, level))


def is_trunk(class_id, roads=("major",)):
    mode, name, _level = LINE_CLASSES[class_id]
    return name in roads if mode == "road" else name != "tram"


def corridor(points, reach):
    line = LineString(points)
    if not line.is_simple:
        line = unary_union(line)
    return line.buffer(reach)


def polygon_of(item):
    return Polygon(item["polygon"], item.get("holes", []))


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


def city_stations(layers, same=150.0):
    frame = layers.stations
    if frame is None or frame.empty:
        return np.zeros((0, 3), dtype=np.float32)
    kinds = _text(frame, "station_kind", "rail")
    rows = []
    for geometry, kind in zip(frame.geometry, kinds):
        if geometry is None or geometry.is_empty:
            continue
        point = geometry.representative_point()
        rows.append((point.x, point.y, RAIL_CLASSES.index(kind) if kind in RAIL_CLASSES else 0))
    rows = np.asarray(sorted(rows), dtype=np.float64).reshape(-1, 3)
    keep = []
    for row in rows:
        twin = False
        for other in reversed(keep):
            if row[0] - other[0] > same:
                break
            if (other[2] == 3) == (row[2] == 3) and np.hypot(other[0] - row[0], other[1] - row[1]) <= same:
                twin = True
                break
        if not twin:
            keep.append(row)
    return np.asarray(keep, dtype=np.float64).reshape(-1, 3)


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
    return image[::-1]


def _pool(image, bounds):
    factor = int(round(CELL / FINE))
    height = image.shape[0] // factor
    width = image.shape[1] // factor
    return image[: height * factor, : width * factor].reshape(height, factor, width, factor).mean(axis=(1, 3))


def line_cells(lines, minx, miny, width, height):
    cells = np.zeros((4, height, width), dtype=np.float32)
    for class_id in range(len(LINE_CLASSES)):
        geometries = [item[1] for item in lines if item[0] == class_id]
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
        np.add.at(cells[channel], (row[keep], column[keep]), length[keep])
    return np.clip(cells / (2.0 * CELL), 0.0, 3.0)


def plan_raster(layers, lines, bounds):
    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / CELL))
    height = int(round((maxy - miny) / CELL))
    plan = np.zeros((len(CHANNELS), height, width), dtype=np.float32)

    plan[:4] = line_cells(lines, minx, miny, width, height)

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


def pool_plan(plan, factor):
    factor = int(factor)
    rows = -(-plan.shape[1] // factor) * factor
    columns = -(-plan.shape[2] // factor) * factor
    padded = np.zeros((plan.shape[0], rows, columns), dtype=plan.dtype)
    padded[:, : plan.shape[1], : plan.shape[2]] = plan
    return padded.reshape(plan.shape[0], rows // factor, factor, columns // factor, factor).mean(axis=(2, 4))


def tangents(line, along, length):
    ahead = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(along + 1.0, length)))
    behind = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(along - 1.0, 0.0)))
    step = ahead - behind
    return step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9)


def is_rail_trunk(class_id):
    return LINE_CLASSES[class_id][0] == "rail" and is_trunk(class_id)


def rail_corridors(lines, gap=12.0):
    return merge_corridors(lines, is_rail_trunk, gap)


def guide_lines(layers, lines, roads=("major",)):
    return [line for name in roads for line in main_corridors(layers, name=name)] + [line for line in rail_corridors(lines) if is_rail_trunk(line[0])]


def main_corridors(layers, reach=40.0, share=0.7, step=20.0, name="major"):
    if layers.roads.empty:
        return []
    frame = layers.roads.explode(index_parts=False)
    names = _text(frame, "road_class", "local")
    tags = _text(frame, "highway", "")
    ways = _text(frame, "oneway", "")
    levels = _text(frame, "vertical_mode", "unknown")
    axis = np.array([math.cos(0.4), math.sin(0.4)])
    keep = []
    spare = []
    for position, geometry in enumerate(frame.geometry):
        if geometry is None or geometry.is_empty or geometry.geom_type != "LineString" or geometry.length < 1.0:
            continue
        if names.iloc[position] != name or tags.iloc[position].endswith("_link"):
            continue
        line = LineString(np.asarray(geometry.coords)[:, :2])
        way = ways.iloc[position]
        single = way in ("yes", "true", "1", "-1") or (tags.iloc[position] == "motorway" and way not in ("no", "false", "0"))
        heading = np.subtract(line.coords[-1], line.coords[0]) * (-1.0 if way == "-1" else 1.0)
        item = (line_class("road", name, levels.iloc[position]), line, 0.0)
        (spare if single and float(heading @ axis) < 0.0 else keep).append(item)
    if keep and spare:
        tree = STRtree([line for _class_id, line, _width in keep])
        classes = np.asarray([class_id for class_id, _line, _width in keep])
        for class_id, line, width in spare:
            count = max(int(line.length // step), 2)
            points = shapely.line_interpolate_point(line, np.linspace(0.0, line.length, count))
            found, nearest = tree.query(points, predicate="dwithin", distance=reach)
            twin = np.zeros(count, dtype=bool)
            twin[found[classes[nearest] == class_id]] = True
            if twin.mean() < share:
                keep.append((class_id, line, width))
    else:
        keep += spare
    return reattach(keep, reach)


def reattach(lines, reach, touch=1.0):
    shapes = [line for _class_id, line, _width in lines]
    tree = STRtree(shapes) if shapes else None
    result = []
    for index, (class_id, line, width) in enumerate(lines):
        points = np.asarray(line.coords)[:, :2]
        for slot in (0, -1):
            end = shapely.Point(points[slot])
            near = [int(other) for other in tree.query(end, predicate="dwithin", distance=reach) if other != index and lines[other][0] == class_id]
            gaps = [shapes[other].distance(end) for other in near]
            if near and min(gaps) > touch:
                target = np.asarray(nearest_points(shapes[near[int(np.argmin(gaps))]], end)[0].coords[0])
                points = np.vstack([target[None], points]) if slot == 0 else np.vstack([points, target[None]])
        result.append((class_id, LineString(points), width))
    return result


def merge_corridors(lines, chosen, gap=12.0, step=10.0, shortest=30.0, shared=100.0):
    rail = [index for index, (class_id, _geometry, _width) in enumerate(lines) if chosen(class_id)]
    if len(rail) < 2:
        return list(lines)
    shapes = np.asarray([lines[index][1] for index in rail], dtype=object)
    classes = np.asarray([lines[index][0] for index in rail])
    length = shapely.length(shapes)
    first, second = STRtree(shapes).query(shapes, predicate="dwithin", distance=gap)
    covers = (first != second) & (classes[first] == classes[second]) & ((length[second] > length[first]) | ((length[second] == length[first]) & (second < first)))
    partners = {}
    for mine, other in zip(first[covers].tolist(), second[covers].tolist()):
        partners.setdefault(mine, []).append(other)
    picked = set(rail)
    kept = [item for index, item in enumerate(lines) if index not in picked]
    for local, shape in enumerate(shapes):
        others = partners.get(local)
        if not others:
            kept.append((int(classes[local]), shape, 0.0))
            continue
        count = max(int(np.ceil(length[local] / step)), 1)
        edges = np.linspace(0.0, length[local], count + 1)
        along = (edges[:-1] + edges[1:]) / 2.0
        here = shapely.line_interpolate_point(shape, along)
        heading = tangents(shape, along, length[local])
        covered = np.zeros(count, dtype=bool)
        for other in others:
            position = shapely.line_locate_point(shapes[other], here)
            near = shapely.distance(here, shapely.line_interpolate_point(shapes[other], position)) <= gap
            if near.any():
                covered |= near & (np.abs((heading * tangents(shapes[other], position, length[other])).sum(axis=1)) > 0.85)
        begin = None
        for index in range(count + 1):
            inside = index < count and covered[index]
            if inside and begin is None:
                begin = index
            if not inside and begin is not None:
                if edges[index] - edges[begin] < shared and (begin > 0 or index < count):
                    covered[begin:index] = False
                begin = None
        if not covered.any():
            kept.append((int(classes[local]), shape, 0.0))
            continue
        rest = unary_union([shapes[other] for other in others])
        start = None
        for index in range(count + 1):
            free = index < count and not covered[index]
            if free and start is None:
                start = index
            if not free and start is not None:
                if edges[index] - edges[start] >= shortest:
                    points = np.asarray(substring(shape, edges[start], edges[index]).coords)[:, :2]
                    for cut, end in ((start > 0, 0), (index < count, -1)):
                        if cut:
                            target = np.asarray(nearest_points(rest, shapely.Point(points[end]))[0].coords[0])
                            if np.linalg.norm(target - points[end]) <= 3.0 * gap:
                                points = np.vstack([target[None], points]) if end == 0 else np.vstack([points, target[None]])
                    kept.append((int(classes[local]), LineString(points), 0.0))
                start = None
    return close_gaps(kept, gap, chosen)


def close_gaps(lines, gap, chosen):
    rail = [index for index, (class_id, geometry, _width) in enumerate(lines) if chosen(class_id) and geometry.length > 2.0 * gap]
    if len(rail) < 2:
        return lines
    coords = [np.asarray(lines[index][1].coords)[:, :2].copy() for index in rail]
    ends = np.asarray([points[slot] for points in coords for slot in (0, -1)])
    first, second = STRtree(shapely.points(ends)).query(shapely.points(ends), predicate="dwithin", distance=gap)
    parent = list(range(len(ends)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in zip(first.tolist(), second.tolist()):
        if a // 2 != b // 2:
            parent[find(a)] = find(b)
    groups = {}
    for index in range(len(ends)):
        groups.setdefault(find(index), []).append(index)
    for members in groups.values():
        if len(members) > 1:
            middle = ends[members].mean(axis=0)
            for member in members:
                coords[member // 2][0 if member % 2 == 0 else -1] = middle
    shapes = [LineString(points) for points in coords]
    tree = STRtree(shapes)
    for number, points in enumerate(coords):
        for slot in (0, -1):
            end = shapely.Point(points[slot])
            near = [int(other) for other in tree.query(end, predicate="dwithin", distance=gap) if other != number]
            gaps = [shapes[other].distance(end) for other in near]
            if near and min(gaps) > 0.5:
                points[slot] = nearest_points(shapes[near[int(np.argmin(gaps))]], end)[0].coords[0]
    lines = list(lines)
    for index, points in zip(rail, coords):
        lines[index] = (lines[index][0], LineString(points), lines[index][2])
    return [line for line in lines if line[1].length > 1.0]


def add_stops(tiles, stations, unit, reach=80.0, same=1.0):
    spots = np.asarray(stations, dtype=np.float64).reshape(-1, 3)[:, :2] / unit
    for tile in tiles:
        x0, y0 = tile["origin"]
        stops = [[] for _target in tile["targets"]]
        inside = spots[(spots[:, 0] >= x0) & (spots[:, 0] < x0 + TILE) & (spots[:, 1] >= y0) & (spots[:, 1] < y0 + TILE)] - (x0, y0) if len(spots) else []
        for spot in inside:
            point = shapely.Point(spot)
            for number, (class_id, points) in enumerate(tile["targets"]):
                if LINE_CLASSES[class_id][0] != "rail":
                    continue
                line = LineString(points)
                if line.distance(point) > reach / unit:
                    continue
                on_line = np.asarray(line.interpolate(line.project(point)).coords[0], dtype=np.float32)
                gaps = np.linalg.norm(points - on_line, axis=1)
                index = int(np.argmin(gaps))
                if gaps[index] > same:
                    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
                    index = int(np.clip(np.searchsorted(cumulative, line.project(point)), 1, len(points) - 1))
                    points = np.insert(points, index, on_line, axis=0)
                    tile["targets"][number] = (class_id, points)
                    stops[number] = [stop + (stop >= index) for stop in stops[number]]
                if index not in stops[number]:
                    stops[number] = sorted(stops[number] + [index])
        tile["stops"] = stops
    return tiles


def trunk_tiles(lines, bounds, stations=(), unit=TRUNK, stride=TILE / 2, roads=("major",)):
    small = [(class_id, affinity.scale(geometry, 1.0 / unit, 1.0 / unit, origin=(0, 0)), width) for class_id, geometry, width in lines if is_trunk(class_id, roads)]
    if not small:
        return []
    tiles = street_tiles(small, tuple(value / unit for value in bounds), stride)
    return add_stops(tiles, stations, unit)


def street_tiles(lines, bounds, stride=TILE, minimum_length=40.0, guide=None):
    minx, miny, maxx, maxy = bounds
    geometries = [geometry for _class_id, geometry, _width in lines]
    tree = STRtree(geometries)
    guide_tree = STRtree([geometry for _class_id, geometry, _width in guide]) if guide else None
    tile_box = box(0.0, 0.0, TILE, TILE)
    tiles = []
    for y0 in np.arange(miny, maxy - TILE / 2, stride):
        for x0 in np.arange(minx, maxx - TILE / 2, stride):
            window = box(x0 - MARGIN, y0 - MARGIN, x0 + TILE + MARGIN, y0 + TILE + MARGIN)
            found = tree.query(window, predicate="intersects")
            spare = (int(round((x0 - minx) / stride)) * 7 + int(round((y0 - miny) / stride)) * 13) % 3 == 0
            if len(found) == 0 and not spare:
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
            if inside_length < minimum_length and not spare:
                continue
            tile = {
                "origin": (float(x0), float(y0)),
                "cell": (int(round((y0 - miny) / CELL)), int(round((x0 - minx) / CELL))),
                "targets": targets,
                "context": context,
                "touches": touch_lists(targets),
            }
            if guide_tree is not None:
                tile["corridors"] = guide_pieces(guide, guide_tree, window, (x0, y0))
            tiles.append(tile)
    return tiles


def street_pattern(tiles, shape, window=5, step=32.0, least=200.0):
    total = np.zeros((3, *shape), dtype=np.float64)
    for tile in tiles:
        row, column = tile["cell"]
        for _class_id, points in tile["targets"]:
            points = np.asarray(points, dtype=np.float64)
            move = np.diff(points, axis=0)
            length = np.hypot(move[:, 0], move[:, 1])
            keep = length > 1e-6
            if not keep.any():
                continue
            start, move, length = points[:-1][keep], move[keep], length[keep]
            count = np.maximum(np.ceil(length / step), 1).astype(int)
            index = np.repeat(np.arange(len(count)), count)
            part = (np.concatenate([np.arange(value) for value in count]) + 0.5) / count[index]
            middle = start[index] + move[index] * part[:, None]
            turn = 4.0 * np.arctan2(move[index, 1], move[index, 0])
            weight = length[index] / count[index]
            rows = row + np.floor(middle[:, 1] / CELL).astype(int)
            columns = column + np.floor(middle[:, 0] / CELL).astype(int)
            inside = (rows >= 0) & (rows < shape[0]) & (columns >= 0) & (columns < shape[1])
            for layer, value in enumerate((weight * np.cos(turn), weight * np.sin(turn), weight)):
                np.add.at(total[layer], (rows[inside], columns[inside]), value[inside])
    reach = window // 2
    padded = np.pad(total, ((0, 0), (reach, reach), (reach, reach)))
    summed = sum(padded[:, dy : dy + shape[0], dx : dx + shape[1]] for dy in range(window) for dx in range(window))
    known = summed[2] >= least
    pattern = np.zeros_like(summed)
    pattern[:2, known] = summed[:2, known] / summed[2, known]
    pattern[2] = known
    return pattern.astype(np.float32)


def guide_pieces(guide, tree, window, origin):
    pieces = []
    for index in tree.query(window, predicate="intersects"):
        for part in _parts(guide[index][1].intersection(window)):
            part = part.simplify(2.0)
            if part.length >= 4.0:
                pieces.append((guide[index][0], np.asarray(part.coords, dtype=np.float32)[:, :2] - np.asarray(origin, dtype=np.float32)))
    return pieces


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


def half_turn(angle):
    return (angle + math.pi / 2.0) % math.pi - math.pi / 2.0


def road_segments(lines):
    pieces = []
    for class_id, shape, *_rest in lines:
        mode, _name, level = LINE_CLASSES[class_id]
        points = np.asarray(shape.coords if hasattr(shape, "coords") else shape, dtype=np.float64)[:, :2]
        if mode == "road" and level != "underground" and len(points) >= 2:
            pieces.append(np.stack([points[:-1], points[1:]], axis=1))
    return np.concatenate(pieces) if pieces else np.zeros((0, 2, 2))


def road_turn(point, segments, reach=60.0):
    if not len(segments):
        return None
    start, step = segments[:, 0], segments[:, 1] - segments[:, 0]
    along = np.clip(((point - start) * step).sum(axis=1) / np.maximum((step * step).sum(axis=1), 1e-9), 0.0, 1.0)
    distance = np.linalg.norm(start + step * along[:, None] - point, axis=1)
    nearest = int(np.argmin(distance))
    if distance[nearest] > reach:
        return None
    return math.atan2(step[nearest, 1], step[nearest, 0])


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
                    holes = [from_frame(np.asarray(ring.coords), centre, angle) for ring in item.interiors]
                    pieces.append(Polygon(from_frame(np.asarray(item.exterior.coords), centre, angle), holes))
    return pieces


def make_blocks(lines, water=None, minimum_area=150.0, edge=None, split=True):
    barriers = [geometry for class_id, geometry, _width in lines if blocks_ground(class_id)]
    if edge is not None:
        barriers.append(edge)
    if not barriers:
        return []
    merged = _parts(unary_union(barriers))
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


def line_index(lines):
    items = [(geometry, class_id) for class_id, geometry, _width in lines if LINE_CLASSES[class_id][2] != "underground"]
    if not items:
        return None, []
    return STRtree([item[0] for item in items]), [item[1] for item in items]


def edge_lines(polygon, tree, classes, count=48, reach=30.0):
    features = np.zeros((count, 5), dtype=np.float32)
    if tree is None:
        return features
    ring = polygon.exterior
    points = shapely.line_interpolate_point(ring, np.linspace(0.0, ring.length, count, endpoint=False))
    (found, nearest), gaps = tree.query_nearest(points, max_distance=reach, all_matches=False, return_distance=True)
    for point, line, gap in zip(found.tolist(), nearest.tolist(), gaps.tolist()):
        mode, name, _level = LINE_CLASSES[classes[line]]
        features[point, 3 if mode == "rail" else ROAD_CLASSES.index(name)] = 1.0
        features[point, 4] = gap / reach
    return features


def inner_lines(polygon, centre, angle, tree, classes, count=48, spacing=12.0, inset=2.0):
    xy = np.zeros((count, 2), dtype=np.float32)
    heading = np.zeros((count, 2), dtype=np.float32)
    kind = np.zeros(count, dtype=np.int64)
    mask = np.zeros(count, dtype=bool)
    core = polygon.buffer(-inset)
    if tree is None or core.is_empty:
        return xy, heading, kind, mask
    used = 0
    for index in tree.query(core, predicate="intersects"):
        for part in _parts(tree.geometries[index].intersection(core)):
            if part.length < 4.0 or used >= count:
                continue
            steps = min(max(int(part.length // spacing), 1), count - used)
            along = (np.arange(steps) + 0.5) * part.length / steps
            here = shapely.get_coordinates(shapely.line_interpolate_point(part, along))
            ahead = shapely.get_coordinates(shapely.line_interpolate_point(part, np.minimum(along + 1.0, part.length)))
            local = to_frame(here, centre, angle)
            step = to_frame(ahead, centre, angle) - local
            xy[used : used + steps] = local
            heading[used : used + steps] = step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-6)
            kind[used : used + steps] = classes[index]
            mask[used : used + steps] = True
            used += steps
    return xy, heading, kind, mask


def block_record(polygon, plan, bounds, road_tree, road_classes, line_tree=None, line_classes=()):
    centre, angle = block_frame(polygon)
    xy, heading, kind, mask = inner_lines(polygon, centre, angle, line_tree, line_classes)
    return {
        "centre": centre.astype(np.float32),
        "angle": float(angle),
        "ring": ring_points(polygon, centre, angle),
        "area": float(polygon.area),
        "plan": plan_at(plan, bounds, centre),
        "frontage": frontage(polygon, road_tree, road_classes),
        "edge": edge_lines(polygon, line_tree, line_classes),
        "inner_xy": xy,
        "inner_heading": heading,
        "inner_class": kind,
        "inner_mask": mask,
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
    line_tree, line_classes = line_index(lines)
    segments = road_segments(lines)
    segment_tree = STRtree(shapely.linestrings(segments)) if len(segments) else None
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
        record = block_record(polygon, plan, bounds, road_tree, road_classes, line_tree, line_classes)
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
        if items and segment_tree is not None:
            spots = from_frame(np.asarray([(item[1], item[2]) for item in items]), centre, angle)
            near = segment_tree.query(box(*spots.min(axis=0) - 60.0, *spots.max(axis=0) + 60.0))
            for number, spot in enumerate(spots):
                road = road_turn(spot, segments[near])
                if road is not None:
                    items[number] = (*items[number][:5], half_turn(items[number][5] - (road - angle)), items[number][6])
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
    lines = [line for line in city_lines(layers) if LINE_CLASSES[line[0]][1] != "tram"]
    bounds = city_bounds(layers)
    plan = plan_raster(layers, lines, bounds)
    stations = city_stations(layers)
    guide = guide_lines(layers, lines)
    tiles = street_tiles([line for line in lines if is_road(line[0])], bounds, stride, guide=guide)
    trunk = trunk_tiles(guide, bounds, stations[stations[:, 2] != RAIL_CLASSES.index("tram")])
    blocks, assigned, total = block_records(layers, lines, plan, bounds)
    folder = Path(output) / name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "plan.npz", plan=plan, bounds=np.asarray(bounds))
    np.save(folder / "stations.npy", stations)
    np.save(folder / "pattern.npy", street_pattern(tiles, plan.shape[1:]))
    torch.save(tiles, folder / "streets.pt")
    torch.save(trunk, folder / "trunk.pt")
    torch.save(blocks, folder / "blocks.pt")
    summary = {
        "name": name,
        "source": str(gpkg),
        "bounds": bounds,
        "plan_shape": list(plan.shape),
        "channels": list(CHANNELS),
        "lines": len(lines),
        "tiles": len(tiles),
        "trunk_tiles": len(trunk),
        "trunk_unit_m": TRUNK,
        "blocks": len(blocks),
        "buildings_in_blocks": assigned,
        "buildings": total,
        "stations": {name: int((stations[:, 2] == index).sum()) for index, name in enumerate(RAIL_CLASSES)},
        "channel_means": {channel: float(plan[index].mean()) for index, channel in enumerate(CHANNELS)},
    }
    summary.update(token_lengths(tiles, blocks, trunk))
    (folder / "meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_trunk(gpkg, name, source, output, roads=("major", "secondary")):
    layers, _metadata = load_city_gpkg(gpkg)
    lines = [line for line in city_lines(layers) if LINE_CLASSES[line[0]][1] != "tram"]
    before = Path(source) / name
    bounds = tuple(np.load(before / "plan.npz")["bounds"].tolist())
    stations = np.load(before / "stations.npy")
    trunk = trunk_tiles(guide_lines(layers, lines, roads), bounds, stations[stations[:, 2] != RAIL_CLASSES.index("tram")], roads=roads)
    folder = Path(output) / name
    folder.mkdir(parents=True, exist_ok=True)
    for item in ("plan.npz", "stations.npy", "pattern.npy"):
        (folder / item).write_bytes((before / item).read_bytes())
    torch.save(trunk, folder / "trunk.pt")
    summary = json.loads((before / "meta.json").read_text())
    summary.update({"trunk_tiles": len(trunk), "trunk_roads": list(roads), "trunk_tokens": token_lengths([], [], trunk)["trunk_tokens"]})
    (folder / "meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def token_lengths(tiles, blocks, trunk=()):
    from urban_layers.tokens import BuildingTokens, StreetTokens

    streets = StreetTokens()
    marked = StreetTokens(stations=True)
    buildings = BuildingTokens()
    street_lengths = [len(streets.encode(tile["targets"])) for tile in tiles] or [0]
    trunk_lengths = [len(marked.encode([(class_id, points, (False, False), stops) for (class_id, points), stops in zip(tile["targets"], tile["stops"])])) for tile in trunk] or [0]
    building_lengths = [len(buildings.encode(block["landuse"], block["buildings"])) for block in blocks] or [0]
    result = {}
    for name, values in (("street_tokens", street_lengths), ("trunk_tokens", trunk_lengths), ("building_tokens", building_lengths)):
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
                "stations": np.load(folder / "stations.npy") if (folder / "stations.npy").exists() else np.zeros((0, 3)),
                "pattern": np.load(folder / "pattern.npy") if (folder / "pattern.npy").exists() else None,
                "folder": folder,
            }
        )
    return cities
