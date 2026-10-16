from __future__ import annotations

import networkx as nx
import numpy as np
from shapely.geometry import Polygon

from urban_layers.data import CELL, CHANNELS, TILE, city_lines, plan_raster


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
    width, height = city["size_m"]
    seams = 0
    for node, degree in graph.degree():
        if degree != 1:
            continue
        x, y = positions[node]
        # a dead end within a few metres of an inside tile edge
        on_x = abs(x / TILE - round(x / TILE)) * TILE <= seam and seam < x < width - seam
        on_y = abs(y / TILE - round(y / TILE)) * TILE <= seam and seam < y < height - seam
        seams += int(on_x or on_y)
    length = sum(np.linalg.norm(np.diff(np.asarray(road["points"]), axis=0), axis=1).sum() for road in city["roads"])
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(components),
        "largest_component_share": largest / max(graph.number_of_nodes(), 1),
        "dead_ends_per_km": float((degrees == 1).sum() / max(length / 1000.0, 1e-6)),
        "junction_share": float((degrees >= 3).mean()) if len(degrees) else 0.0,
        "broken_tile_seams": int(seams),
        "road_km_per_km2": float(length / 1000.0 / (width * height / 1e6)),
    }


def building_stats(city):
    shapes = [Polygon(item["footprint"]) for item in city["buildings"]]
    heights = np.asarray([item["height_m"] for item in city["buildings"]]) if shapes else np.zeros(0)
    width, height = city["size_m"]
    overlaps = 0
    if shapes:
        from shapely.strtree import STRtree

        tree = STRtree(shapes)
        for index, shape in enumerate(shapes):
            for other in tree.query(shape, predicate="intersects"):
                if other > index and shape.intersection(shapes[other]).area > 0.2 * min(shape.area, shapes[other].area):
                    overlaps += 1
    return {
        "buildings": len(shapes),
        "valid": float(np.mean([shape.is_valid for shape in shapes])) if shapes else 0.0,
        "coverage": float(sum(shape.area for shape in shapes) / (width * height)),
        "overlap_pairs": overlaps,
        "median_height_m": float(np.median(heights)) if len(heights) else 0.0,
        "blocks": len(city["blocks"]),
        "landuse": {name: sum(block["landuse"] == name for block in city["blocks"]) for name in sorted({block["landuse"] for block in city["blocks"]})},
    }


def follows_plan(city, plan):
    # measure the finished city with the same code that made the training plans
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


def audit_city(city, plan=None):
    report = {"network": network_stats(city), "buildings": building_stats(city)}
    if plan is not None:
        report["follows_plan"], _realised = follows_plan(city, plan)
    return report
