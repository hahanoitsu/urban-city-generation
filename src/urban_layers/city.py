from __future__ import annotations

import json
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
import torch
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree

from urban_dataset.city_state import build_transport_graph
from urban_dataset.extract import CityLayers
from urban_dataset.tile import TileSpec
from urban_layers.buildings import BuildingModel, block_arrays, place_buildings
from urban_layers.data import (
    CELL,
    CHANNELS,
    LANDUSE,
    LINE_CLASSES,
    TILE,
    MARGIN,
    block_record,
    boundary_side,
    corridor,
    guide_pieces,
    is_trunk,
    line_index,
    make_blocks,
    plan_raster,
    polygon_of,
    pool_plan,
    ring_box,
    road_index,
    split_block,
)
from urban_layers.network import join_network
from urban_layers.plan import PlanNet, sample_plan
from urban_layers.rail import RAIL_WIDTH, TRACKS, heading_at, rail_corridors, rail_runs, rail_stations
from urban_layers.streets import PLAN_CELLS, PORTS, SIDES, StreetModel, collate, condition_arrays, plan_crop, port_key, snap_to_ports
from urban_model.structured_city_data import BUILDING_KINDS

SLIDERS = {
    "roads": ("major", "secondary", "local"),
    "major": ("major",),
    "local": ("local",),
    "rail": ("rail",),
    "buildings": ("coverage",),
    "height": ("height",),
    "green": ("green",),
    "water": ("water",),
    "residential": ("residential",),
    "commercial": ("commercial",),
    "industrial": ("industrial",),
    "civic": ("civic",),
}
WIDTHS = {"major": 14.0, "secondary": 9.0, "local": 6.0}
SETBACKS = {"major": 6.0, "secondary": 4.0, "local": 3.0}


def load_layer(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    layer = checkpoint["layer"]
    if layer == "plan":
        model = PlanNet(**checkpoint["config"])
    elif layer in ("streets", "trunk"):
        model = StreetModel(**checkpoint["config"])
    else:
        model = BuildingModel(**checkpoint["config"])
    model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    return model.to(device).eval(), checkpoint


def style_weights(cities, style):
    weights = np.zeros(len(cities), dtype=np.float32)
    for name, value in style.items():
        if name not in cities:
            raise ValueError(f"model was not trained on {name}, it knows {cities}")
        weights[cities.index(name)] = float(value)
    if weights.sum() <= 0:
        weights[:] = 1.0
    return weights / weights.sum()


def requested_controls(checkpoint, weights, sliders):
    controls = np.zeros(len(CHANNELS), dtype=np.float32)
    for weight, values in zip(weights, checkpoint["city_controls"]):
        controls += weight * np.asarray(values, dtype=np.float32)
    for name, factor in sliders.items():
        for channel in SLIDERS[name]:
            controls[CHANNELS.index(channel)] *= float(factor)
    return controls


REGION_CLASSES = ("residential", "commercial", "industrial", "civic", "green")


def plan_field(plan, channel, upsample=8):
    return np.clip(ndimage.zoom(plan[CHANNELS.index(channel)], upsample, order=3), 0.0, None)


def smooth(polygon, rounds=2):
    def ring(coords):
        points = np.asarray(coords, dtype=np.float64)[:-1]
        if len(points) < 3:
            return np.asarray(coords, dtype=np.float64)
        for _round in range(rounds):
            following = np.roll(points, -1, axis=0)
            points = np.stack([0.75 * points + 0.25 * following, 0.25 * points + 0.75 * following], axis=1).reshape(-1, 2)
        return np.vstack([points, points[:1]])

    return Polygon(ring(polygon.exterior.coords), [ring(inner.coords) for inner in polygon.interiors]).buffer(0)


def trace(mask, size, minimum_area):
    polygons = []
    for geometry, _value in features.shapes(mask.astype(np.uint8), mask=mask, transform=Affine(size, 0, 0, 0, size, 0)):
        polygon = shape(geometry).buffer(size).buffer(-size).simplify(size / 2)
        for part in getattr(polygon, "geoms", [polygon]):
            if part.geom_type != "Polygon":
                continue
            part = smooth(part)
            for piece in getattr(part, "geoms", [part]):
                if piece.geom_type == "Polygon" and piece.area >= minimum_area:
                    polygons.append(piece)
    return polygons


def plan_polygons(plan, channel, upsample=8, minimum_area=1500.0):
    field = plan_field(plan, channel, upsample)
    share = float(plan[CHANNELS.index(channel)].mean())
    if share <= 0.0:
        return []
    threshold = max(float(np.quantile(field, 1.0 - min(share, 1.0))), 0.15)
    return trace(field > threshold, CELL / upsample, minimum_area)


def landuse_labels(plan, upsample=8, rounds=60):
    shares = np.stack([plan_field(plan, name, upsample) for name in REGION_CLASSES])
    scores = np.concatenate([np.clip(1.0 - shares.sum(axis=0), 0.0, None)[None], shares])
    wanted = np.array([plan[CHANNELS.index(name)].mean() for name in REGION_CLASSES])
    wanted = np.concatenate([[max(1.0 - wanted.sum(), 0.0)], wanted])
    weights = np.ones(len(wanted))
    best = (np.inf, weights)
    for _round in range(rounds):
        got = np.bincount((weights[:, None, None] * scores).argmax(axis=0).ravel(), minlength=len(wanted)) / scores[0].size
        error = float(np.abs(got - wanted).sum())
        if error < best[0]:
            best = (error, weights.copy())
        weights = weights * np.sqrt((wanted + 1e-4) / (got + 1e-4))
        weights = np.maximum(weights / weights.max(), 1e-6)
    return (best[1][:, None, None] * scores).argmax(axis=0)


def landuse_regions(plan, upsample=8, minimum_area=1500.0, rounds=60, labels=None):
    labels = landuse_labels(plan, upsample, rounds) if labels is None else labels
    regions = []
    for name in REGION_CLASSES:
        index = LANDUSE.index(name)
        regions += [(polygon, index) for polygon in trace(labels == index, CELL / upsample, minimum_area)]
    return regions


def natural_cover(plan, labels, upsample=8, minimum_area=10000.0, cover=0.03, roads=0.3):
    thin = plan_field(plan, "coverage", upsample) < cover
    thin &= sum(plan_field(plan, name, upsample) for name in ("major", "secondary", "local")) < roads
    thin &= plan_field(plan, "water", upsample) < 0.5
    return trace((labels == 0) & thin, CELL / upsample, minimum_area)


def landuse_blocks(lines, water, regions, edge):
    parcels = make_blocks(lines, water, edge=edge, split=False)
    tree = STRtree([polygon for polygon, _use in regions]) if regions else None
    pieces = []
    for parcel in parcels:
        hits = sorted(tree.query(parcel, predicate="intersects")) if tree is not None else []
        rest = parcel
        cut = []
        for index in hits:
            cut.append((rest.intersection(regions[index][0]), regions[index][1]))
            rest = rest.difference(regions[index][0])
        for geometry, use in cut + [(rest, 0)]:
            for part in getattr(geometry, "geoms", [geometry]):
                if part.geom_type == "Polygon" and part.area >= 150.0:
                    pieces.append((part, use))
    return pieces


def neighbour_lines(done, row, column):
    context = []
    ports = []
    visible = set()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            key = (row + dy, column + dx)
            if (dx or dy) and key in done:
                visible.add((dx, dy))
                ring = ring_box(dx, dy)
                for class_id, points, joins in done[key]:
                    moved = np.asarray(points, dtype=np.float64) + (dx * TILE, dy * TILE)
                    clipped = LineString(moved).intersection(ring)
                    for part in getattr(clipped, "geoms", [clipped]):
                        if part.geom_type == "LineString" and part.length > 2.0:
                            context.append((class_id, np.asarray(part.coords, dtype=np.float32)))
                    if dx == 0 or dy == 0:
                        for slot, end in ((0, 0), (1, -1)):
                            point = moved[end]
                            side = boundary_side(point)
                            if side >= 0 and SIDES[side] == (dx, dy) and not joins[slot]:
                                inner = moved[1] if end == 0 else moved[-2]
                                step = point - inner
                                step = step / max(np.linalg.norm(step), 1e-6)
                                ports.append((float(point[0]), float(point[1]), float(step[0]), float(step[1]), class_id))
    ports.sort(key=port_key)
    return context, ports[:PORTS], visible


def to_edge(points, reach=4.0, visible=(), joins=(False, False)):
    points = np.asarray(points, dtype=np.float32).copy()
    for slot, end in ((0, 0), (1, -1)):
        if joins[slot]:
            continue
        for axis in (0, 1):
            low, high = ((-1, 0), (1, 0)) if axis == 0 else ((0, -1), (0, 1))
            if points[end, axis] < reach and low not in visible:
                points[end, axis] = 0.0
            elif points[end, axis] > TILE - reach and high not in visible:
                points[end, axis] = TILE
    return points


def generate_streets(model, checkpoint, plan, weights, generator=None, temperature=1.0, top_p=0.95, device="cpu", log=print, rail=False, classes=None, plan_cells=PLAN_CELLS, given=None, name="streets"):
    spread = [CHANNELS.index(channel) for channel in checkpoint.get("spread", ())]
    rows = math.ceil(plan.shape[1] * CELL / TILE)
    columns = math.ceil(plan.shape[2] * CELL / TILE)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    per_tile = int(round(TILE / CELL))
    if given is not None:
        given = [(class_id, LineString(points), 0.0) for class_id, points in given]
        guide_tree = STRtree([geometry for _class_id, geometry, _width in given]) if given else None
        classes = [mode == "road" for mode, _name, _level in LINE_CLASSES]
    elif classes is None and not rail:
        plan = plan.copy()
        plan[CHANNELS.index("rail")] = 0.0
        classes = [mode == "road" for mode, _name, _level in LINE_CLASSES]
    done = {}
    marks = {}
    for diagonal in range(rows + columns - 1):
        batch = [(row, diagonal - row) for row in range(rows) if 0 <= diagonal - row < columns]
        items = []
        meta = []
        for row, column in batch:
            context, ports, visible = neighbour_lines(done, row, column)
            crop = plan_crop(plan, row * per_tile, column * per_tile, size=plan_cells) * scale[:, None, None]
            if spread:
                crop[spread] = crop[spread].mean(axis=(1, 2), keepdims=True)
            corridors = []
            if given:
                window = box(column * TILE - MARGIN, row * TILE - MARGIN, (column + 1) * TILE + MARGIN, (row + 1) * TILE + MARGIN)
                corridors = guide_pieces(given, guide_tree, window, (column * TILE, row * TILE))
            items.append(condition_arrays(crop, context, ports, weights, corridors))
            meta.append((ports, visible))
        arrays = {name: value.to(device) for name, value in collate(items).items()}
        arrays["sides"] = torch.tensor([[SIDES[side] in visible for side in range(4)] for _ports, visible in meta], dtype=torch.bool, device=device)
        results = model.generate(arrays, temperature=temperature, top_p=top_p, generator=generator, classes=classes)
        for (row, column), lines, (ports, visible) in zip(batch, results, meta):
            snapped = snap_to_ports(lines, ports)
            done[(row, column)] = [(class_id, to_edge(points, visible=visible, joins=joins), joins) for class_id, points, joins, *_rest in snapped]
            marks[(row, column)] = [rest[0] if rest else () for _class_id, _points, _joins, *rest in snapped]
        log(f"{name}: diagonal {diagonal + 1}/{rows + columns - 1}, tiles {len(done)}/{rows * columns}")
    lines = []
    stops = []
    for (row, column), items in done.items():
        for (class_id, points, joins), stopped in zip(items, marks[(row, column)]):
            moved = np.asarray(points, dtype=np.float64) + (column * TILE, row * TILE)
            lines.append((class_id, moved, joins))
            stops += [(float(moved[index][0]), float(moved[index][1])) for index in stopped if index < len(moved)]
    return (lines, stops) if model.tokens.station is not None else lines


def clip_lines(lines, size):
    area = box(0.0, 0.0, *size)
    clipped = []
    for class_id, points, joins in lines:
        line = LineString(points)
        if area.covers(line):
            clipped.append((class_id, np.asarray(points, dtype=np.float64), joins))
            continue
        piece = line.intersection(area)
        for part in getattr(piece, "geoms", [piece]):
            if part.geom_type == "LineString" and part.length > 2.0:
                clipped.append((class_id, np.asarray(part.coords), (False, False)))
    return clipped


def rail_lines(lines, touch=1.0):
    rail = [(class_id, points) for class_id, points in lines if LINE_CLASSES[class_id][0] == "rail"]
    parent = list(range(len(rail)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    shapes = [LineString(points) for _class_id, points in rail]
    tree = STRtree(shapes) if shapes else None
    for index, (_class_id, points) in enumerate(rail):
        for end in (points[0], points[-1]):
            for other in tree.query(Point(end), predicate="dwithin", distance=touch):
                parent[find(index)] = find(int(other))
    names = {}
    return [(class_id, points, names.setdefault(find(index), len(names))) for index, (class_id, points) in enumerate(rail)]


def place_stations(stops, runs, reach=100.0, same=60.0, touch=1.0):
    shapes = [LineString(points) for _class_id, points, _number in runs]
    if not shapes or not stops:
        return []
    tree = STRtree(shapes)
    spots = []
    for x, y in stops:
        nearest = tree.query_nearest(Point(x, y), max_distance=reach, all_matches=False)
        if len(nearest):
            point = shapes[int(nearest[0])].interpolate(shapes[int(nearest[0])].project(Point(x, y)))
            if all(np.hypot(point.x - other.x, point.y - other.y) > same for other in spots):
                spots.append(point)
    stations = []
    for point in spots:
        near = sorted(tree.query(point, predicate="dwithin", distance=touch).tolist())
        numbers = sorted({runs[index][2] for index in near})
        stations.append({"xy": [point.x, point.y], "lines": numbers, "heading_deg": heading_at(shapes[near[0]], shapes[near[0]].project(point))})
    return stations


def generate_trunk(model, checkpoint, plan, weights, size, generator=None, temperature=1.0, top_p=0.95, device="cpu", log=print):
    unit = float(checkpoint["unit"])
    classes = [is_trunk(class_id) for class_id in range(len(LINE_CLASSES))]
    lines, stops = generate_streets(model, checkpoint, pool_plan(plan, unit), weights, generator, temperature, top_p, device, log, classes=classes, plan_cells=int(checkpoint.get("plan_cells", PLAN_CELLS)), name="trunk")
    snapped = snap_lines(clip_lines(lines, (size[0] / unit, size[1] / unit)))
    main = [(class_id, np.asarray(points, dtype=np.float64) * unit) for class_id, points in snapped]
    return main, [(x * unit, y * unit) for x, y in stops]


def insert_vertex(points, point):
    line = LineString(points)
    distance = line.project(Point(point))
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    index = int(np.clip(np.searchsorted(cumulative, distance), 1, len(points) - 1))
    if np.linalg.norm(points[index - 1] - point) < 1e-6 or np.linalg.norm(points[index] - point) < 1e-6:
        return points
    return np.insert(points, index, point, axis=0)


def snap_lines(lines, reach=3.0, join_reach=40.0, corner=2.0):
    items = [(item[0], np.asarray(item[1], dtype=np.float64).copy(), item[2] if len(item) > 2 else (False, False)) for item in lines]
    geometries = [LineString(points) for _class_id, points, _joins in items]
    tree = STRtree(geometries)
    groups = [(LINE_CLASSES[class_id][0], LINE_CLASSES[class_id][2]) for class_id, _points, _joins in items]
    original = {(index, slot): points[0 if slot == 0 else -1].copy() for index, (_c, points, _j) in enumerate(items) for slot in (0, 1)}
    moves = {}
    links = {}
    inserts = {}
    for index, (_class_id, points, joins) in enumerate(items):
        for slot in (0, 1):
            point = Point(original[(index, slot)])
            limit = join_reach if joins[slot] else reach
            best = None
            for other in tree.query(point, predicate="dwithin", distance=limit):
                if other == index or groups[other][0] != groups[index][0]:
                    continue
                if groups[other] != groups[index] and not joins[slot]:
                    continue
                order = (groups[other] != groups[index], geometries[other].distance(point))
                if best is None or order < best[0]:
                    best = (order, int(other))
            if best is None:
                continue
            best = (best[0][1], best[1])
            target = geometries[best[1]]
            spot = nearest_points(target, point)[0]
            ends = [Point(target.coords[0]), Point(target.coords[-1])]
            near = min((0, 1), key=lambda k: ends[k].distance(spot))
            if ends[near].distance(spot) <= corner:
                links[(index, slot)] = (best[1], near)
            elif best[0] > 1e-6:
                moves[(index, slot)] = np.array([spot.x, spot.y])
                inserts.setdefault(best[1], []).append((spot.x, spot.y))
            else:
                inserts.setdefault(best[1], []).append(tuple(original[(index, slot)]))

    parent = {}

    def find(key):
        while parent.get(key, key) != key:
            key = parent[key]
        return key

    for key, other in links.items():
        a, b = find(key), find(other)
        if a != b:
            parent[a] = b
    groups_of = {}
    for key in original:
        groups_of.setdefault(find(key), []).append(key)
    final = {}
    for members in groups_of.values():
        anchored = [key for key in members if key in moves]
        still = [key for key in members if key not in moves and key not in links]
        if anchored:
            position = moves[anchored[0]]
        elif still:
            position = original[still[0]]
        else:
            position = original[min(members)]
        for key in members:
            final[key] = position

    result = []
    for index, (class_id, points, _joins) in enumerate(items):
        points[0] = final[(index, 0)]
        points[-1] = final[(index, 1)]
        for point in inserts.get(index, []):
            points = insert_vertex(points, np.asarray(point))
        if len(points) >= 2 and LineString(points).length > 1.0:
            result.append((class_id, points))
    return result


def lines_to_layers(lines, width_m, height_m):
    roads = []
    rail = []
    for index, (class_id, points) in enumerate(lines):
        mode, name, level = LINE_CLASSES[class_id]
        geometry = LineString(points)
        if mode == "road":
            roads.append({"id": f"road-{index}", "road_class": name, "vertical_mode": level, "estimated_width_m": WIDTHS.get(name, 6.0), "geometry": geometry})
        else:
            rail.append({"id": f"rail-{index}", "railway": name, "vertical_mode": level, "geometry": geometry})
    roads_frame = gpd.GeoDataFrame(roads, geometry="geometry", crs="EPSG:3857") if roads else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
    rail_frame = gpd.GeoDataFrame(rail, geometry="geometry", crs="EPSG:3857") if rail else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
    tile = TileSpec(city_id="generated", column=0, row=0, minx=0.0, miny=0.0, maxx=float(width_m), maxy=float(height_m))
    return roads_frame, rail_frame, join_levels(build_transport_graph(roads_frame, rail_frame, tile))


def join_levels(graph):
    spots = {}
    for node in graph["nodes"]:
        x, y = node["position_local_m"][:2]
        spots.setdefault((node["transport_mode"], round(x, 2), round(y, 2)), []).append(node)
    merged = {}
    for group in spots.values():
        if len({node["vertical_mode"] for node in group}) < 2 or not any(node["degree"] == 1 for node in group):
            continue
        keep = min(group, key=lambda node: (node["vertical_mode"] != "surface", node["id"]))
        for node in group:
            if node is not keep:
                merged[node["id"]] = keep["id"]
    if not merged:
        return graph
    degree = {}
    for edge in graph["edges"]:
        edge["from_node"] = merged.get(edge["from_node"], edge["from_node"])
        edge["to_node"] = merged.get(edge["to_node"], edge["to_node"])
        for name in (edge["from_node"], edge["to_node"]):
            degree[name] = degree.get(name, 0) + 1
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in merged]
    for node in graph["nodes"]:
        node["degree"] = degree.get(node["id"], 0)
        if node["boundary_port_key"] is None:
            node["node_type"] = "endpoint" if node["degree"] <= 1 else "intersection" if node["degree"] >= 3 else "continuation"
    graph["statistics"]["nodes"] = len(graph["nodes"])
    graph["statistics"]["intersections"] = sum(node["node_type"] == "intersection" for node in graph["nodes"])
    return graph


def planned_cover(plan, polygon, step=32.0):
    minx, miny, maxx, maxy = polygon.bounds
    xs, ys = np.meshgrid(np.arange(minx + step / 2, maxx, step), np.arange(miny + step / 2, maxy, step))
    xs, ys = xs.ravel(), ys.ravel()
    inside = shapely.contains_xy(polygon, xs, ys)
    if not inside.any():
        point = polygon.representative_point()
        xs, ys, inside = np.array([point.x]), np.array([point.y]), np.array([True])
    columns = np.clip((xs[inside] // CELL).astype(int), 0, plan.shape[2] - 1)
    rows = np.clip((ys[inside] // CELL).astype(int), 0, plan.shape[1] - 1)
    return float(plan[CHANNELS.index("coverage"), rows, columns].mean())


def pick_samples(areas, targets, goal, rounds=40):
    def chosen(scale):
        return [int(np.argmin(np.abs(values - scale * target))) for values, target in zip(areas, targets)]

    def total(picks):
        return float(sum(values[pick] for values, pick in zip(areas, picks)))

    low, high = 0.0, 20.0
    for _round in range(rounds):
        middle = (low + high) / 2.0
        if total(chosen(middle)) < goal:
            low = middle
        else:
            high = middle
    return chosen((low + high) / 2.0)


def reachable(placed, road_tree, reach=120.0):
    if road_tree is None or not placed:
        return placed
    centres = shapely.points(np.asarray([np.mean(item["footprint"], axis=0) for item in placed]))
    near = set(road_tree.query_nearest(centres, max_distance=reach, all_matches=False)[0].tolist())
    return [item for index, item in enumerate(placed) if index in near]


def drop_overlaps(chosen, tolerance=1.0):
    shapes = [Polygon(item["footprint"]) for _block_id, item in chosen]
    tree = STRtree(shapes) if shapes else None
    keep = np.ones(len(shapes), dtype=bool)
    for index, building in enumerate(shapes):
        for other in tree.query(building, predicate="intersects"):
            if other < index and keep[other] and building.intersection(shapes[other]).area > tolerance:
                keep[index] = False
                break
    return keep, sum(shape.area for shape, kept in zip(shapes, keep) if kept)


def keep_out(lines, size, water=None, setback=True):
    shapes = []
    for class_id, points in lines:
        mode, name, level = LINE_CLASSES[class_id]
        if level == "underground":
            continue
        reach = WIDTHS.get(name, 6.0) / 2.0 if mode == "road" else RAIL_WIDTH / 2.0
        if setback:
            reach += SETBACKS.get(name, 3.0) if mode == "road" else 4.0
        shapes.append(corridor(points, reach))
    if water is not None and not water.is_empty:
        shapes += list(getattr(water, "geoms", [water]))
    shapes.append(box(-500.0, -500.0, size[0] + 500.0, size[1] + 500.0).difference(box(0.0, 0.0, size[0], size[1])))
    return shapes


def rings(polygon):
    record = {"polygon": np.asarray(polygon.exterior.coords).tolist()}
    if polygon.interiors:
        record["holes"] = [np.asarray(ring.coords).tolist() for ring in polygon.interiors]
    return record


def cut_off_roads(pieces, lines):
    paved = []
    for class_id, points in lines:
        mode, name, level = LINE_CLASSES[class_id]
        if level in ("surface", "unknown"):
            paved.append(corridor(points, (WIDTHS.get(name, 6.0) if mode == "road" else RAIL_WIDTH) / 2.0))
    tree = STRtree(paved) if paved else None
    result = []
    for polygon, use in pieces:
        near = tree.query(polygon, predicate="intersects") if tree is not None else []
        rest = polygon.difference(unary_union([paved[index] for index in near])) if len(near) else polygon
        result += [(part, use) for part in getattr(rest, "geoms", [rest]) if part.geom_type == "Polygon" and part.area > 10.0]
    return result


def bridge_spans(points, water):
    if water is None or water.is_empty or not LineString(points).intersects(water):
        return []
    spans = []
    start = 0.0
    for first, second in zip(points[:-1], points[1:]):
        segment = LineString([first, second])
        wet = segment.intersection(water)
        for piece in getattr(wet, "geoms", [wet]):
            if piece.geom_type == "LineString" and piece.length > 0.0:
                ends = sorted((segment.project(Point(piece.coords[0])), segment.project(Point(piece.coords[-1]))))
                spans.append([start + ends[0], start + ends[1]])
        start += segment.length
    joined = []
    for low, high in sorted(spans):
        if joined and low <= joined[-1][1] + 1e-6:
            joined[-1][1] = max(joined[-1][1], high)
        else:
            joined.append([low, high])
    return [[round(low, 2), round(high, 2)] for low, high in joined if high - low > 1.0]


def road_segments(lines):
    pieces = []
    for class_id, points in lines:
        mode, _name, level = LINE_CLASSES[class_id]
        points = np.asarray(points, dtype=np.float64)
        if mode == "road" and level != "underground" and len(points) >= 2:
            pieces.append(np.stack([points[:-1], points[1:]], axis=1))
    return np.concatenate(pieces) if pieces else np.zeros((0, 2, 2))


def generate_buildings(model, checkpoint, pieces, plan, size, lines, weights, generator=None, temperature=1.0, top_p=0.95, samples=4, batch_size=64, device="cpu", log=print, water=None):
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    shapes = [(class_id, LineString(points), 0.0) for class_id, points in lines]
    road_tree, road_classes = road_index(shapes)
    line_tree, line_classes = line_index(shapes)
    bounds = (0.0, 0.0, size[0], size[1])
    parts = [(block_id, part, use) for block_id, (polygon, use) in enumerate(pieces) for part in split_block(polygon)]
    records = [block_record(part, plan, bounds, road_tree, road_classes, line_tree, line_classes) for _block_id, part, _use in parts]
    covers = [planned_cover(plan, part) for _block_id, part, _use in parts]
    grounds = [keep_out(lines, size, water, setback) for setback in (True, False)]
    trees = [STRtree(shapes) for shapes in grounds]
    segments = road_segments(lines)
    segment_tree = STRtree(shapely.linestrings(segments)) if len(segments) else None

    def surroundings(part):
        near_bounds = part.buffer(40.0).bounds
        blocked = []
        for shapes, tree in zip(grounds, trees):
            near = tree.query(box(*near_bounds))
            ground = unary_union([shapely.clip_by_rect(shapes[index], *near_bounds) for index in near]) if len(near) else None
            if ground is not None and not ground.is_empty:
                ground = ground if ground.is_valid else shapely.make_valid(ground)
                shapely.prepare(ground)
                blocked.append(ground)
        roads = segments[segment_tree.query(part.buffer(30.0))] if segment_tree is not None else None
        return blocked, roads

    around = [surroundings(part) for _block_id, part, _use in parts]
    options = [[] for _part in parts]
    tally = {}
    for start in range(0, len(parts), batch_size):
        chunk = list(range(start, min(start + batch_size, len(parts))))
        items = [block_arrays(records[index], scale, weights) for index in chunk]
        arrays = {name: torch.from_numpy(np.stack([item[name] for item in items])).to(device) for name in items[0]}
        uses = torch.tensor([parts[index][2] for index in chunk], dtype=torch.long, device=device)
        for _sample in range(samples):
            results = model.generate(arrays, temperature=temperature, top_p=top_p, generator=generator, landuse=uses)
            for index, (_landuse, items_out) in zip(chunk, results):
                blocked, roads = around[index]
                placed = place_buildings(parts[index][1], records[index]["centre"], records[index]["angle"], items_out, blocked=blocked, roads=roads, face=not model.config.get("roads", False), tally=tally)
                options[index].append(reachable(placed, road_tree, 120.0 if covers[index] >= 0.06 else 60.0))
        log(f"buildings: parts {chunk[-1] + 1}/{len(parts)}")

    areas = [np.array([sum(Polygon(item["footprint"]).area for item in placed) for placed in option]) for option in options]
    targets = np.array([cover * part.area for cover, (_block_id, part, _use) in zip(covers, parts)])
    goal = float(plan[CHANNELS.index("coverage")].mean()) * size[0] * size[1]
    picks = pick_samples(areas, targets, goal)
    ground = cut_off_roads(pieces, lines)
    blocks = [{"id": block_id, "landuse": LANDUSE[use], **rings(polygon)} for block_id, (polygon, use) in enumerate(ground)]
    ground_tree = STRtree([polygon for polygon, _use in ground]) if ground else None
    chosen = [(block_id, item) for (block_id, _part, _use), option, pick in zip(parts, options, picks) for item in option[pick]]
    keep, built = drop_overlaps(chosen)
    if 0.0 < built < 0.98 * goal:
        picks = pick_samples(areas, targets, goal * goal / built)
        chosen = [(block_id, item) for (block_id, _part, _use), option, pick in zip(parts, options, picks) for item in option[pick]]
        keep, built = drop_overlaps(chosen)
    buildings = []
    for (_block_id, item), kept in zip(chosen, keep):
        if not kept:
            continue
        footprint = Polygon(item["footprint"])
        near = ground_tree.query(footprint, predicate="intersects") if ground_tree is not None else []
        item["block"] = int(max(near, key=lambda index: ground[index][0].intersection(footprint).area)) if len(near) else -1
        item["id"] = len(buildings)
        item["kind"] = BUILDING_KINDS[item["kind"]] if item["kind"] < len(BUILDING_KINDS) else "generic"
        buildings.append(item)
    log(f"buildings: {len(buildings)} in {len(blocks)} blocks, built area {built / max(goal, 1.0):.2f} of the plan")
    drawn = max(tally.get("drawn", 0), 1)
    placement = {"kept_as_drawn": tally.get("kept", 0) / drawn, "moved_or_cut": tally.get("changed", 0) / drawn, "dropped": 1.0 - (tally.get("kept", 0) + tally.get("changed", 0)) / drawn}
    log("boxes from the model: " + ", ".join(f"{name.replace('_', ' ')} {share:.0%}" for name, share in placement.items()))
    return blocks, buildings, placement


def generate_city(
    models,
    size_m=(4096.0, 4096.0),
    style=None,
    sliders=None,
    seed=0,
    plan_steps=50,
    guidance=1.5,
    temperature=1.0,
    top_p=0.95,
    device="cpu",
    known_plan=None,
    known_mask=None,
    match=True,
    building_samples=4,
    building_batch=64,
    log=print,
    transport_rules=True,
):
    started = time.time()
    plan_model, plan_checkpoint = models["plan"]
    street_model, street_checkpoint = models["streets"]
    building_model, building_checkpoint = models["buildings"]
    cities = plan_checkpoint["cities"]
    weights = style_weights(cities, style or {name: 1.0 for name in cities})
    sliders = sliders or {}
    controls = requested_controls(plan_checkpoint, weights, sliders)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    height = int(math.ceil(size_m[1] / CELL / 4) * 4)
    width = int(math.ceil(size_m[0] / CELL / 4) * 4)
    plan = sample_plan(
        plan_model,
        height,
        width,
        weights,
        controls * np.asarray(plan_checkpoint["control_scale"]),
        plan_checkpoint["mean"],
        plan_checkpoint["std"],
        steps=plan_steps,
        guidance=guidance,
        generator=generator,
        known=known_plan,
        known_mask=known_mask,
        wanted=controls if match else None,
    )
    size = (width * CELL, height * CELL)
    log(f"plan: {height} x {width} cells, {time.time() - started:.0f}s")
    water = plan_polygons(plan, "water")
    water_union = unary_union(water) if water else None
    labels = landuse_labels(plan)
    regions = landuse_regions(plan, labels=labels)
    natural = natural_cover(plan, labels)
    street_weights = style_weights(street_checkpoint["cities"], {name: weights[cities.index(name)] for name in cities if name in street_checkpoint["cities"]})
    area = box(0.0, 0.0, *size)
    rules = ["plan totals"] if match else []
    rules += ["natural cover", "building clearance" if building_model.config.get("roads", False) else "building placement"]
    if models.get("trunk") is None and not transport_rules:
        lines = generate_streets(street_model, street_checkpoint, plan, street_weights, generator, temperature, top_p, device, log, rail=True)
        snapped = snap_lines(clip_lines(lines, size))
        streets = [(class_id, points) for class_id, points in snapped if LINE_CLASSES[class_id][0] == "road"]
        runs = rail_lines(snapped)
        stations = []
    elif models.get("trunk") is None:
        rules += ["rail", "stations", "road joins"]
        corridors = rail_corridors(plan, size)
        runs = rail_runs(corridors, plan)
        stations = rail_stations(corridors, plan)
        log(f"rail: {len(corridors)} lines, {sum(LineString(points).length for points in corridors) / 1000.0:.1f} km, {len(stations)} stations, {time.time() - started:.0f}s")
        lines = generate_streets(street_model, street_checkpoint, plan, street_weights, generator, temperature, top_p, device, log)
        streets, joined = join_network(snap_lines(clip_lines(lines, size)), plan, size, water_union)
        if joined:
            log(f"roads joined: {joined['ramps']} ramps landed, {joined['stitched']} ends stitched, {joined['dropped']} stray pieces dropped, {joined['major_links']} main road links ({joined['major_link_km']:.1f} km), {joined['links']} other links ({joined['link_km']:.1f} km)")
    else:
        trunk_model, trunk_checkpoint = models["trunk"]
        trunk_weights = style_weights(trunk_checkpoint["cities"], {name: weights[cities.index(name)] for name in cities if name in trunk_checkpoint["cities"]})
        main, stops = generate_trunk(trunk_model, trunk_checkpoint, plan, trunk_weights, size, generator, temperature, top_p, device, log)
        km = lambda mode: sum(LineString(points).length for class_id, points in main if LINE_CLASSES[class_id][0] == mode) / 1000.0
        log(f"trunk: {km('rail'):.1f} km of rail, {km('road'):.1f} km of main road corridor, {time.time() - started:.0f}s")
        lines = generate_streets(street_model, street_checkpoint, plan, street_weights, generator, temperature, top_p, device, log, given=main)
        streets = snap_lines(clip_lines(lines, size))
        runs = rail_lines(main)
        stations = place_stations(stops, runs)
        log(f"rail: {len(runs)} pieces, {len(stations)} stations")
    lines = streets + [(class_id, points) for class_id, points, _number in runs]
    roads_frame, rail_frame, graph = lines_to_layers(lines, *size)
    log(f"network: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, {time.time() - started:.0f}s")

    pieces = landuse_blocks([(class_id, LineString(points), 0.0) for class_id, points in lines], water_union, regions, area.exterior)
    green = [polygon for polygon, use in pieces if LANDUSE[use] == "green"]
    taken = unary_union(green + ([water_union] if water_union is not None else []))
    kept = []
    for polygon in natural:
        rest = polygon.difference(taken) if not taken.is_empty else polygon
        kept += [part for part in getattr(rest, "geoms", [rest]) if part.geom_type == "Polygon" and part.area >= 1500.0]
    natural = kept
    building_weights = style_weights(building_checkpoint["cities"], {name: weights[cities.index(name)] for name in cities if name in building_checkpoint["cities"]})
    blocks, buildings, placement = generate_buildings(
        building_model, building_checkpoint, pieces, plan, size, lines, building_weights, generator, temperature, top_p, samples=building_samples, batch_size=building_batch, device=device, log=log, water=water_union
    )
    log(f"buildings done, {time.time() - started:.0f}s")

    city = {
        "format": "urban-layers-city",
        "version": "0.1.0",
        "units": "metres",
        "axes": "x east, y north, z up",
        "size_m": [float(size[0]), float(size[1])],
        "seed": int(seed),
        "style": {name: float(weight) for name, weight in zip(cities, weights)},
        "sliders": {name: float(value) for name, value in sliders.items()},
        "controls": {channel: float(value) for channel, value in zip(CHANNELS, controls)},
        "rules": rules,
        "placement": placement,
        "plan": {"cell_m": CELL, "channels": list(CHANNELS), "shape": list(plan.shape)},
        "roads": [],
        "rail": [],
        "nodes": [
            {"id": node["id"], "xy": node["position_local_m"][:2], "mode": node["transport_mode"], "vertical": node["vertical_mode"], "degree": node["degree"], "type": node["node_type"]}
            for node in graph["nodes"]
        ],
        "edges": [
            {"id": edge["id"], "from": edge["from_node"], "to": edge["to_node"], "mode": edge["transport_mode"], "class": edge["class"], "vertical": edge["vertical_mode"], "width_m": edge["width_m"], "points": [point[:2] for point in edge["geometry_local_m"]]}
            for edge in graph["edges"]
        ],
        "blocks": blocks,
        "buildings": buildings,
        "water": [rings(polygon) for polygon in water],
        "green": [{**rings(polygon), "kind": "park"} for polygon in green] + [{**rings(polygon), "kind": "natural"} for polygon in natural],
        "stations": stations,
    }
    for index, (class_id, points) in enumerate(streets):
        mode, name, level = LINE_CLASSES[class_id]
        record = {"id": index, "class": name, "vertical": level, "points": np.asarray(points).tolist()}
        if mode == "road":
            record["width_m"] = WIDTHS.get(name, 6.0)
            spans = bridge_spans(points, water_union) if level in ("surface", "unknown") else []
            if spans:
                record["bridges"] = spans
            city["roads"].append(record)
        else:
            city["rail"].append(record)
    for class_id, points, number in runs:
        _mode, name, level = LINE_CLASSES[class_id]
        record = {"id": len(city["roads"]) + len(city["rail"]), "class": name, "vertical": level, "points": np.asarray(points).tolist()}
        city["rail"].append({**record, "line": number, "tracks": TRACKS, "width_m": RAIL_WIDTH})
    return city, plan


def city_layers(city):
    crs = "EPSG:3857"

    def frame(rows):
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)

    roads = frame([{"road_class": road["class"], "vertical_mode": road["vertical"], "estimated_width_m": road.get("width_m", 6.0), "geometry": LineString(road["points"])} for road in city["roads"]])
    rail = frame([{"railway": item["class"], "vertical_mode": item["vertical"], "geometry": LineString(item["points"])} for item in city["rail"] for _track in range(item.get("tracks", 1))])
    buildings = frame([{"estimated_height_m": item["height_m"], "building": item["kind"], "geometry": shape({"type": "Polygon", "coordinates": [item["footprint"]]})} for item in city["buildings"]])
    landuse_rows = []
    for block in city["blocks"]:
        if block["landuse"] in ("residential", "commercial", "industrial", "civic", "green"):
            name = "commercial_mixed" if block["landuse"] == "commercial" else block["landuse"]
            landuse_rows.append({"landuse_class": name, "geometry": polygon_of(block)})
    landuse = frame(landuse_rows)
    water = frame([{"geometry": polygon_of(item)} for item in city["water"]])
    green = frame([{"geometry": polygon_of(item)} for item in city["green"] if item.get("kind") != "natural"])
    return CityLayers(roads=roads, buildings=buildings, landuse=landuse, landuse_known=landuse, water=water, green=green, rail=rail)


def export_obj(city, path):
    vertices = []
    faces = []
    lines_out = ["mtllib city.mtl"]

    def add(points, group, material):
        lines_out.append(f"o {group}")
        lines_out.append(f"usemtl {material}")
        start = len(vertices) + 1
        for point in points:
            vertices.append(point)
        return start

    for building in city["buildings"]:
        corners = building["footprint"][:4]
        height = max(float(building["height_m"]), 3.0)
        start = add([(x, 0.0, -y) for x, y in corners] + [(x, height, -y) for x, y in corners], f"building_{building['id']}", "building")
        quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
        for quad in quads:
            lines_out.append("f " + " ".join(str(start + index) for index in quad))
    for kind, items in (("road", city["roads"]), ("rail", city["rail"])):
        for item in items:
            points = np.asarray(item["points"], dtype=np.float64)
            width = float(item.get("width_m", 4.0))
            level = 6.0 if item["vertical"] == "elevated" else -12.0 if item["vertical"] == "underground" else 0.05
            for index in range(len(points) - 1):
                a, b = points[index], points[index + 1]
                step = b - a
                length = np.linalg.norm(step)
                if length < 1e-6:
                    continue
                side = np.array([-step[1], step[0]]) / length * width / 2.0
                quad = [a - side, b - side, b + side, a + side]
                start = add([(x, level, -y) for x, y in quad], f"{kind}_{item['id']}_{index}", kind)
                lines_out.append(f"f {start} {start + 1} {start + 2} {start + 3}")
    for kind in ("water", "green"):
        for index, item in enumerate(city[kind]):
            ring = item["polygon"][:-1]
            start = add([(x, 0.02, -y) for x, y in ring], f"{kind}_{index}", kind)
            lines_out.append("f " + " ".join(str(start + offset) for offset in range(len(ring))))
    text = "\n".join(["# generated city preview", *(f"v {x:.2f} {y:.2f} {z:.2f}" for x, y, z in vertices), *lines_out]) + "\n"
    Path(path).write_text(text)
    materials = {"building": (0.7, 0.7, 0.72), "road": (0.25, 0.25, 0.27), "rail": (0.3, 0.35, 0.5), "water": (0.4, 0.6, 0.85), "green": (0.45, 0.7, 0.4)}
    Path(path).with_name("city.mtl").write_text("".join(f"newmtl {name}\nKd {r} {g} {b}\n\n" for name, (r, g, b) in materials.items()))


def save_city(city, plan, folder):
    from urban_layers.render import city_image, plan_image

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "city.json").write_text(json.dumps(city) + "\n")
    np.save(folder / "plan.npy", plan)
    city_image(city, int(min(8192, max(1400, max(city["size_m"]) / 3.0)))).save(folder / "city.png")
    plan_image(plan, 8).save(folder / "plan.png")
    export_obj(city, folder / "city.obj")
