from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import LineString, Point, box, shape
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
    block_record,
    boundary_side,
    make_blocks,
    plan_raster,
    ring_box,
    road_index,
)
from urban_layers.plan import PlanNet, sample_plan
from urban_layers.streets import PORTS, SIDES, StreetModel, collate, condition_arrays, plan_crop, port_key, snap_to_ports
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


def load_layer(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    layer = checkpoint["layer"]
    if layer == "plan":
        model = PlanNet(**checkpoint["config"])
    elif layer == "streets":
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


def plan_polygons(plan, channel, threshold=0.5, upsample=8, minimum_area=1500.0):
    field = ndimage.zoom(plan[CHANNELS.index(channel)], upsample, order=1)
    mask = (field > threshold).astype(np.uint8)
    size = CELL / upsample
    polygons = []
    for geometry, value in features.shapes(mask, mask=mask.astype(bool), transform=Affine(size, 0, 0, 0, size, 0)):
        polygon = shape(geometry).buffer(size).buffer(-size).simplify(size)
        for part in getattr(polygon, "geoms", [polygon]):
            if part.geom_type == "Polygon" and part.area >= minimum_area:
                polygons.append(part)
    return polygons


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
                            # a joined end gets snapped onto a line later, it does not cross into this tile
                            if side >= 0 and SIDES[side] == (dx, dy) and not joins[slot]:
                                inner = moved[1] if end == 0 else moved[-2]
                                step = point - inner
                                step = step / max(np.linalg.norm(step), 1e-6)
                                ports.append((float(point[0]), float(point[1]), float(step[0]), float(step[1]), class_id))
    ports.sort(key=port_key)
    return context, ports[:PORTS], visible


def to_edge(points, reach=4.0, visible=(), joins=(False, False)):
    # a line that stops just short of the tile edge is meant to leave the tile, so the next tile can pick it up.
    # not towards a tile that is already done, there it would just be a dead end on the seam,
    # and not for joined ends, those get snapped onto a line instead
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


def generate_streets(model, checkpoint, plan, weights, generator=None, temperature=1.0, top_p=0.95, device="cpu", log=print):
    rows = math.ceil(plan.shape[1] * CELL / TILE)
    columns = math.ceil(plan.shape[2] * CELL / TILE)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    per_tile = int(round(TILE / CELL))
    done = {}
    for diagonal in range(rows + columns - 1):
        batch = [(row, diagonal - row) for row in range(rows) if 0 <= diagonal - row < columns]
        items = []
        meta = []
        for row, column in batch:
            context, ports, visible = neighbour_lines(done, row, column)
            crop = plan_crop(plan, row * per_tile, column * per_tile) * scale[:, None, None]
            items.append(condition_arrays(crop, context, ports, weights))
            meta.append((ports, visible))
        arrays = {name: value.to(device) for name, value in collate(items).items()}
        arrays["sides"] = torch.tensor([[SIDES[side] in visible for side in range(4)] for _ports, visible in meta], dtype=torch.bool, device=device)
        results = model.generate(arrays, temperature=temperature, top_p=top_p, generator=generator)
        for (row, column), lines, (ports, visible) in zip(batch, results, meta):
            done[(row, column)] = [(class_id, to_edge(points, visible=visible, joins=joins), joins) for class_id, points, joins in snap_to_ports(lines, ports)]
        log(f"streets: diagonal {diagonal + 1}/{rows + columns - 1}, tiles {len(done)}/{rows * columns}")
    lines = []
    for (row, column), items in done.items():
        for class_id, points, joins in items:
            lines.append((class_id, np.asarray(points, dtype=np.float64) + (column * TILE, row * TILE), joins))
    return lines


def insert_vertex(points, point):
    line = LineString(points)
    distance = line.project(Point(point))
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    index = int(np.clip(np.searchsorted(cumulative, distance), 1, len(points) - 1))
    if np.linalg.norm(points[index - 1] - point) < 1e-6 or np.linalg.norm(points[index] - point) < 1e-6:
        return points
    return np.insert(points, index, point, axis=0)


def snap_lines(lines, reach=3.0, join_reach=40.0, corner=2.0):
    # lines are (class, points) or (class, points, joins). an end the model marked as a join moves onto
    # the nearest line of the same mode and level within join_reach, any other end only if it nearly
    # touches one. the meeting point is added to the other line too so the graph really joins them
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
                if other == index or groups[other] != groups[index]:
                    continue
                distance = geometries[other].distance(point)
                if best is None or distance < best[0]:
                    best = (distance, int(other))
            if best is None:
                continue
            target = geometries[best[1]]
            spot = nearest_points(target, point)[0]
            ends = [Point(target.coords[0]), Point(target.coords[-1])]
            near = min((0, 1), key=lambda k: ends[k].distance(spot))
            if ends[near].distance(spot) <= corner:
                # meets the other line's end, these two ends become one junction
                links[(index, slot)] = (best[1], near)
            elif best[0] > 1e-6:
                moves[(index, slot)] = np.array([spot.x, spot.y])
                inserts.setdefault(best[1], []).append((spot.x, spot.y))
            else:
                inserts.setdefault(best[1], []).append(tuple(original[(index, slot)]))

    # ends that meet at a corner are grouped, so a chain of snaps cannot leave a gap behind
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
    return roads_frame, rail_frame, build_transport_graph(roads_frame, rail_frame, tile)


def generate_buildings(model, checkpoint, polygons, plan, size, lines, weights, generator=None, temperature=1.0, top_p=0.95, batch_size=64, device="cpu", log=print):
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    road_tree, road_classes = road_index([(class_id, LineString(points), 0.0) for class_id, points in lines])
    bounds = (0.0, 0.0, size[0], size[1])
    records = [block_record(polygon, plan, bounds, road_tree, road_classes) for polygon in polygons]
    blocks = []
    buildings = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        items = [block_arrays(record, scale, weights) for record in chunk]
        arrays = {name: torch.from_numpy(np.stack([item[name] for item in items])).to(device) for name in items[0]}
        results = model.generate(arrays, temperature=temperature, top_p=top_p, generator=generator)
        for offset, (record, (landuse, items_out)) in enumerate(zip(chunk, results)):
            block_id = start + offset
            polygon = polygons[block_id]
            blocks.append({"id": block_id, "landuse": LANDUSE[landuse] if landuse < len(LANDUSE) else "none", "polygon": np.asarray(polygon.exterior.coords).tolist()})
            for item in place_buildings(polygon, record["centre"], record["angle"], items_out):
                item["block"] = block_id
                item["id"] = len(buildings)
                item["kind"] = BUILDING_KINDS[item["kind"]] if item["kind"] < len(BUILDING_KINDS) else "generic"
                buildings.append(item)
        log(f"buildings: blocks {min(start + batch_size, len(records))}/{len(records)}, buildings {len(buildings)}")
    return blocks, buildings


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
    log=print,
):
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
    log(f"plan: {height} x {width} cells")
    water = plan_polygons(plan, "water")
    green = plan_polygons(plan, "green")
    water_union = unary_union(water) if water else None

    street_weights = style_weights(street_checkpoint["cities"], {name: weights[cities.index(name)] for name in cities if name in street_checkpoint["cities"]})
    lines = generate_streets(street_model, street_checkpoint, plan, street_weights, generator, temperature, top_p, device, log)
    area = box(0.0, 0.0, *size)
    clipped = []
    for class_id, points, joins in lines:
        line = LineString(points)
        if area.covers(line):
            # intersection would also cut a line where it crosses itself, so only clip what sticks out
            clipped.append((class_id, np.asarray(points, dtype=np.float64), joins))
            continue
        piece = line.intersection(area)
        for part in getattr(piece, "geoms", [piece]):
            if part.geom_type == "LineString" and part.length > 2.0:
                clipped.append((class_id, np.asarray(part.coords), (False, False)))
    lines = snap_lines(clipped)
    roads_frame, rail_frame, graph = lines_to_layers(lines, *size)
    log(f"network: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

    polygons = make_blocks([(class_id, LineString(points), 0.0) for class_id, points in lines], water_union, edge=area.exterior)
    building_weights = style_weights(building_checkpoint["cities"], {name: weights[cities.index(name)] for name in cities if name in building_checkpoint["cities"]})
    blocks, buildings = generate_buildings(building_model, building_checkpoint, polygons, plan, size, lines, building_weights, generator, temperature, top_p, device=device, log=log)

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
        "water": [{"polygon": np.asarray(polygon.exterior.coords).tolist()} for polygon in water],
        "green": [{"polygon": np.asarray(polygon.exterior.coords).tolist()} for polygon in green],
    }
    for index, (class_id, points) in enumerate(lines):
        mode, name, level = LINE_CLASSES[class_id]
        record = {"id": index, "class": name, "vertical": level, "points": np.asarray(points).tolist()}
        if mode == "road":
            record["width_m"] = WIDTHS.get(name, 6.0)
            city["roads"].append(record)
        else:
            city["rail"].append(record)
    return city, plan


def city_layers(city):
    crs = "EPSG:3857"

    def frame(rows):
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)

    roads = frame([{"road_class": road["class"], "vertical_mode": road["vertical"], "estimated_width_m": road.get("width_m", 6.0), "geometry": LineString(road["points"])} for road in city["roads"]])
    rail = frame([{"railway": item["class"], "vertical_mode": item["vertical"], "geometry": LineString(item["points"])} for item in city["rail"]])
    buildings = frame([{"estimated_height_m": item["height_m"], "building": item["kind"], "geometry": shape({"type": "Polygon", "coordinates": [item["footprint"]]})} for item in city["buildings"]])
    landuse_rows = []
    for block in city["blocks"]:
        if block["landuse"] in ("residential", "commercial", "industrial", "civic", "green"):
            name = "commercial_mixed" if block["landuse"] == "commercial" else block["landuse"]
            landuse_rows.append({"landuse_class": name, "geometry": shape({"type": "Polygon", "coordinates": [block["polygon"]]})})
    landuse = frame(landuse_rows)
    water = frame([{"geometry": shape({"type": "Polygon", "coordinates": [item["polygon"]]})} for item in city["water"]])
    green = frame([{"geometry": shape({"type": "Polygon", "coordinates": [item["polygon"]]})} for item in city["green"]])
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
            level = 6.0 if item["vertical"] == "elevated" else 0.05
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
    city_image(city, 1400).save(folder / "city.png")
    plan_image(plan, 8).save(folder / "plan.png")
    export_obj(city, folder / "city.obj")
