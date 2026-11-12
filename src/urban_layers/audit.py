from __future__ import annotations

import math

import networkx as nx
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
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


def road_pieces(roads, touch=1.0):
    lines = np.asarray([LineString(road["points"]) for road in roads], dtype=object)
    levels = np.asarray(["surface" if road["vertical"] == "unknown" else road["vertical"] for road in roads])
    first, second = STRtree(lines).query(lines, predicate="dwithin", distance=touch)
    keep = first < second
    first, second = first[keep], second[keep]
    starts = shapely.points([line.coords[0] for line in lines])
    ends = shapely.points([line.coords[-1] for line in lines])
    ramp = np.zeros(len(first), dtype=bool)
    for a, b in ((first, second), (second, first)):
        ramp |= (shapely.distance(starts[a], lines[b]) <= touch) | (shapely.distance(ends[a], lines[b]) <= touch)
    joined = (levels[first] == levels[second]) | ramp
    graph = coo_matrix((np.ones(int(joined.sum())), (first[joined], second[joined])), shape=(len(lines), len(lines)))
    _count, label = connected_components(graph, directed=False)
    return lines, label


def detour(city, low=1000.0, high=3000.0, sources=40, targets=50, seed=0):
    edges = [edge for edge in city["edges"] if edge.get("mode") == "road" and "points" in edge]
    if not edges:
        return None
    where = {node["id"]: node["xy"] for node in city["nodes"]}
    spot = {}
    for edge in edges:
        for name in (edge["from"], edge["to"]):
            x, y = where[name]
            spot.setdefault((round(x), round(y)), len(spot))
    index = lambda name: spot[(round(where[name][0]), round(where[name][1]))]
    rows = [index(edge["from"]) for edge in edges]
    columns = [index(edge["to"]) for edge in edges]
    lengths = [max(LineString(edge["points"]).length, 0.01) for edge in edges]
    graph = coo_matrix((lengths, (rows, columns)), shape=(len(spot), len(spot))).tocsr()
    graph = graph.maximum(graph.T)
    _count, label = connected_components(graph, directed=False)
    positions = np.zeros((len(spot), 2))
    for (x, y), number in spot.items():
        positions[number] = (x, y)
    biggest = np.nonzero(label == np.bincount(label).argmax())[0]
    rng = np.random.default_rng(seed)
    starts = rng.choice(biggest, min(sources, len(biggest)), replace=False)
    ratios = []
    for start, distances in zip(starts, dijkstra(graph, directed=False, indices=starts)):
        straight = np.linalg.norm(positions[biggest] - positions[start], axis=1)
        options = np.nonzero((straight >= low) & (straight <= high))[0]
        for option in rng.choice(options, min(targets, len(options)), replace=False) if len(options) else []:
            ratios.append(distances[biggest[option]] / straight[option])
    return float(np.median(ratios)) if ratios else None


def function_stats(city):
    width, height = city["size_m"]
    area = width * height / 1e6
    roads = [road for road in city["roads"] if len(road["points"]) >= 2]
    shapes = [Polygon(item["footprint"]) for item in city["buildings"]]
    if not roads or not shapes:
        return {}
    lines, label = road_pieces(roads)
    lengths = shapely.length(lines)
    piece_length = np.bincount(label, weights=lengths)
    order = np.argsort(piece_length)[::-1]
    major = [road for road in roads if road["class"] == "major"]
    major_share = 0.0
    if major:
        major_lines, major_label = road_pieces(major)
        major_length = np.bincount(major_label, weights=shapely.length(major_lines))
        major_share = float(major_length.max() / major_length.sum())

    middles = shapely.points([shape.centroid.coords[0] for shape in shapes])
    xy = shapely.get_coordinates(middles)
    floor = np.asarray([shape.area * max(item["height_m"] / 3.0, 1.0) for shape, item in zip(shapes, city["buildings"])])
    nearest = STRtree(lines).nearest(middles)
    result = {
        "largest_piece_road_share": float(piece_length[order[0]] / piece_length.sum()),
        "second_piece_road_share": float(piece_length[order[1]] / piece_length.sum()) if len(order) > 1 else 0.0,
        "major_largest_piece_share": major_share,
        "pieces_under_200m_per_km2": float((piece_length < 200.0).sum() / area),
        "floor_on_largest_piece": float(floor[label[nearest] == order[0]].sum() / floor.sum()),
        "detour_1_to_3km": detour(city),
    }

    stations = np.asarray([item["xy"] for item in city.get("stations", []) if item.get("kind") != "tram"])
    if len(stations):
        gap = np.linalg.norm(xy[:, None] - stations[None], axis=2)
        around = np.asarray([floor[gap[:, number] <= 800.0].sum() for number in range(len(stations))])
        result["floor_within_800m_of_a_station"] = float(floor[gap.min(axis=1) <= 800.0].sum() / floor.sum())
        result["stations_with_almost_nobody_near"] = int((around < 0.1 * np.median(around)).sum())

    blocks = [polygon_of(item) for item in city["blocks"]]
    use = np.asarray(["none"] * len(shapes), dtype=object)
    if blocks:
        inside, block = STRtree(blocks).query(middles, predicate="within")
        use[inside] = [city["blocks"][number]["landuse"] for number in block]
    homes = use == "residential"
    green = [polygon_of(item) for item in city.get("green", [])]
    shops = [shape for shape, item in zip(blocks, city["blocks"]) if item["landuse"] == "commercial"]
    for name, places, reach in (("homes_within_400m_of_green", green, 400.0), ("homes_within_800m_of_shops", shops, 800.0)):
        if homes.any() and places:
            near = STRtree(places).query(middles[homes], predicate="dwithin", distance=reach)[0]
            result[name] = float(floor[homes][np.unique(near)].sum() / floor[homes].sum())

    cells = max(int(min(width, height) // 1024), 1)
    grid = np.zeros((cells, cells))
    shops_grid = np.zeros((cells, cells))
    column = np.clip((xy[:, 0] // 1024).astype(int), 0, cells - 1)
    row = np.clip((xy[:, 1] // 1024).astype(int), 0, cells - 1)
    np.add.at(grid, (row, column), floor / 1024.0**2)
    np.add.at(shops_grid, (row[use == "commercial"], column[use == "commercial"]), floor[use == "commercial"] / 1024.0**2)
    result["busiest_km2_floor_ratio"] = float(grid.max())
    result["busiest_km2_shop_floor_ratio"] = float(shops_grid.max())
    for scale in (1, 4):
        if cells >= 2 * scale:
            coarse = grid[: cells // scale * scale, : cells // scale * scale].reshape(cells // scale, scale, cells // scale, scale).mean(axis=(1, 3))
            result[f"floor_spread_{scale}km"] = float(coarse.std() / max(coarse.mean(), 1e-9))
    return result


def audit_city(city, plan=None):
    report = {"network": network_stats(city), "buildings": building_stats(city), "rail": rail_stats(city), "cover": cover_stats(city), "function": function_stats(city)}
    if plan is not None:
        report["follows_plan"], _realised = follows_plan(city, plan)
    return report
