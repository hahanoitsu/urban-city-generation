from __future__ import annotations

import math

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree

from urban_layers.data import CELL, CHANNELS, TILE, city_lines, corridor, plan_raster, polygon_of


def network_stats(city, seam=4.0):
    graph = nx.Graph()
    positions = {}
    for node in city["nodes"]:
        graph.add_node(node["id"])
        positions[node["id"]] = node["xy"]
    for edge in city["edges"]:
        graph.add_edge(edge["from"], edge["to"])
    degrees = np.asarray([degree for _node, degree in graph.degree()]) if graph.number_of_nodes() else np.zeros(0)
    components = list(nx.connected_components(graph))
    largest = max((len(item) for item in components), default=0)
    degree_of = dict(graph.degree())
    lone = sum(1 for item in components if all(degree_of[node] <= 2 for node in item) and sum(degree_of[node] == 1 for node in item) == 2)
    width, height = city["size_m"]
    seams = 0
    for node, degree in graph.degree():
        if degree != 1:
            continue
        x, y = positions[node]
        on_x = abs(x / TILE - round(x / TILE)) * TILE <= seam and seam < x < width - seam
        on_y = abs(y / TILE - round(y / TILE)) * TILE <= seam and seam < y < height - seam
        seams += int(on_x or on_y)
    length = sum(np.linalg.norm(np.diff(np.asarray(road["points"]), axis=0), axis=1).sum() for road in city["roads"])
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(components),
        "lone_lines": int(lone),
        "largest_component_share": largest / max(graph.number_of_nodes(), 1),
        "dead_ends_per_km": float((degrees == 1).sum() / max(length / 1000.0, 1e-6)),
        "junction_share": float((degrees >= 3).mean()) if len(degrees) else 0.0,
        "broken_tile_seams": int(seams),
        "road_km_per_km2": float(length / 1000.0 / (width * height / 1e6)),
    }


def street_relations(city, shapes, reach=30.0, within=5.0):
    roads = [road for road in city["roads"] if road["vertical"] != "underground" and len(road["points"]) >= 2]
    if not shapes or not roads:
        return 0.0, 0.0
    corridors = [corridor(road["points"], road.get("width_m", 6.0) / 2.0) for road in roads]
    tree = STRtree(corridors)
    on_road = sum(any(shape.intersection(corridors[index]).area > 1.0 for index in tree.query(shape, predicate="intersects")) for shape in shapes)
    segments = np.concatenate([np.stack([np.asarray(road["points"])[:-1], np.asarray(road["points"])[1:]], axis=1) for road in roads])
    segments = segments[np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) > 0.5]
    segment_tree = STRtree(shapely.linestrings(segments))
    near = along = 0
    for shape in shapes:
        rectangle = shape.minimum_rotated_rectangle
        hit = segment_tree.query_nearest(shape.centroid, max_distance=reach)
        if rectangle.geom_type != "Polygon" or not len(hit):
            continue
        corners = np.asarray(rectangle.exterior.coords)[:4]
        axis = max((corners[1] - corners[0], corners[2] - corners[1]), key=np.linalg.norm)
        start, end = segments[int(hit[0])]
        angle = math.degrees(math.atan2(axis[1], axis[0]) - math.atan2(end[1] - start[1], end[0] - start[0])) % 90.0
        near += 1
        along += int(min(angle, 90.0 - angle) < within)
    return on_road / len(shapes), along / max(near, 1)


def building_stats(city):
    shapes = [Polygon(item["footprint"]) for item in city["buildings"]]
    heights = np.asarray([item["height_m"] for item in city["buildings"]]) if shapes else np.zeros(0)
    width, height = city["size_m"]
    overlaps = 0
    touching = 0
    if shapes:
        tree = STRtree(shapes)
        for index, shape in enumerate(shapes):
            for other in tree.query(shape, predicate="intersects"):
                if other <= index:
                    continue
                area = shape.intersection(shapes[other]).area
                overlaps += int(area > 0.2 * min(shape.area, shapes[other].area))
                touching += int(area > 1.0)
    on_road, facing = street_relations(city, shapes)
    return {
        "buildings": len(shapes),
        "valid": float(np.mean([shape.is_valid for shape in shapes])) if shapes else 0.0,
        "coverage": float(sum(shape.area for shape in shapes) / (width * height)),
        "overlap_pairs": overlaps,
        "any_overlap_pairs": touching,
        "on_road_share": float(on_road),
        "facing_road_share": float(facing),
        "median_height_m": float(np.median(heights)) if len(heights) else 0.0,
        "blocks": len(city["blocks"]),
        "landuse": {name: sum(block["landuse"] == name for block in city["blocks"]) for name in sorted({block["landuse"] for block in city["blocks"]})},
    }


def follows_plan(city, plan):
    from urban_layers.city import city_layers

    layers = city_layers(city)
    width, height = city["size_m"]
    realised = plan_raster(layers, city_lines(layers), (0.0, 0.0, width, height))
    rows = min(realised.shape[1], plan.shape[1])
    columns = min(realised.shape[2], plan.shape[2])
    result = {}
    for index, channel in enumerate(CHANNELS):
        wanted = plan[index, :rows, :columns].ravel()
        got = realised[index, :rows, :columns].ravel()
        correlation = float(np.corrcoef(wanted, got)[0, 1]) if wanted.std() > 1e-6 and got.std() > 1e-6 else 0.0
        result[channel] = {"plan_mean": float(wanted.mean()), "city_mean": float(got.mean()), "correlation": correlation}
    return result, realised


def rail_stats(city):
    lines = [(LineString(item["points"]), item.get("tracks", 1)) for item in city["rail"] if len(item["points"]) >= 2]
    stations = len(city.get("stations", []))
    if not lines:
        return {"track_km": 0.0, "pieces": 0, "loose_ends": 0, "stations": stations}
    width, height = city["size_m"]
    joined = unary_union([line for line, _tracks in lines])
    merged = linemerge(list(getattr(joined, "geoms", [joined])))
    pieces = list(getattr(merged, "geoms", [merged]))
    ends = {}
    for piece in pieces:
        for x, y in (piece.coords[0], piece.coords[-1]):
            key = (round(x, 1), round(y, 1))
            ends[key] = ends.get(key, 0) + 1
    loose = sum(1 for (x, y), count in ends.items() if count == 1 and 20.0 < x < width - 20.0 and 20.0 < y < height - 20.0)
    return {"track_km": sum(line.length * tracks for line, tracks in lines) / 1000.0, "pieces": len(pieces), "loose_ends": loose, "stations": stations}


def cover_stats(city):
    width, height = city["size_m"]
    areas = {"park": 0.0, "natural": 0.0}
    for item in city.get("green", []):
        areas["natural" if item.get("kind") == "natural" else "park"] += polygon_of(item).area
    return {f"{name}_share": area / (width * height) for name, area in areas.items()}


def audit_city(city, plan=None):
    report = {"network": network_stats(city), "buildings": building_stats(city), "rail": rail_stats(city), "cover": cover_stats(city)}
    if plan is not None:
        report["follows_plan"], _realised = follows_plan(city, plan)
    return report
