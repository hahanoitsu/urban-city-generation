from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

from urban_layers.audit import follows_plan
from urban_layers.data import CHANNELS, TILE, corridor, polygon_of
from urban_layers.rail import RAIL_WIDTH


def points_outside(city, slack=0.5):
    width, height = city["size_m"]
    groups = {
        "roads": [road["points"] for road in city["roads"]],
        "rail": [item["points"] for item in city["rail"]],
        "buildings": [item["footprint"] for item in city["buildings"]],
        "blocks": [item["polygon"] + sum(item.get("holes", []), []) for item in city["blocks"]],
        "water": [item["polygon"] + sum(item.get("holes", []), []) for item in city["water"]],
        "green": [item["polygon"] + sum(item.get("holes", []), []) for item in city["green"]],
        "stations": [[item["xy"]] for item in city.get("stations", [])],
        "nodes": [[item["xy"]] for item in city["nodes"]],
    }
    result = {}
    for name, shapes in groups.items():
        points = np.asarray([point for shape in shapes for point in shape], dtype=np.float64).reshape(-1, 2)
        bad = (points[:, 0] < -slack) | (points[:, 0] > width + slack) | (points[:, 1] < -slack) | (points[:, 1] > height + slack)
        bad |= ~np.isfinite(points).all(axis=1)
        result[name] = int(bad.sum())
    return result


def bad_shapes(city):
    result = {}
    for name, key in (("buildings", "footprint"), ("blocks", "polygon"), ("water", "polygon"), ("green", "polygon")):
        shapes = [Polygon(item[key]) if key == "footprint" else polygon_of(item) for item in city[name]]
        result[name] = int(sum(not shape.is_valid or shape.area <= 0.0 for shape in shapes))
    result["lines"] = int(sum(LineString(item["points"]).length <= 0.0 for item in city["roads"] + city["rail"]))
    result["building_heights"] = int(sum(not np.isfinite(item["height_m"]) or item["height_m"] <= 0.0 for item in city["buildings"]))
    return result


def plan_orientation(city, plan):
    _report, realised = follows_plan(city, plan)
    roads = [CHANNELS.index(name) for name in ("major", "secondary", "local")]
    got = realised[roads].sum(axis=0)
    wanted = plan[roads].sum(axis=0)
    rows, columns = min(got.shape[0], wanted.shape[0]), min(got.shape[1], wanted.shape[1])
    got, wanted = got[:rows, :columns], wanted[:rows, :columns]
    views = {"as is": wanted, "flipped north south": wanted[::-1], "flipped east west": wanted[:, ::-1], "turned round": wanted[::-1, ::-1]}
    if rows == columns:
        views["x and y swapped"] = wanted.T
    scores = {}
    for name, view in views.items():
        flat = (view.ravel(), got.ravel())
        scores[name] = float(np.corrcoef(*flat)[0, 1]) if min(flat[0].std(), flat[1].std()) > 1e-9 else 0.0
    return scores


def building_problems(city):
    shapes = [Polygon(item["footprint"]) for item in city["buildings"]]
    blocks = [polygon_of(item) for item in city["blocks"]]
    ids = [item["id"] for item in city["blocks"]]
    block_tree = STRtree(blocks)
    off_land = 0
    wrong_block = 0
    for shape, item in zip(shapes, city["buildings"]):
        near = block_tree.query(shape, predicate="intersects")
        shares = {ids[index]: blocks[index].intersection(shape).area for index in near}
        off_land += int(sum(shares.values()) < 0.7 * shape.area)
        wrong_block += int(bool(shares) and item["block"] != max(shares, key=shares.get))
    tree = STRtree(shapes)
    overlaps = 0
    for index, shape in enumerate(shapes):
        overlaps += sum(1 for other in tree.query(shape, predicate="intersects") if other > index and shape.intersection(shapes[other]).area > 1.0)
    grounds = [corridor(road["points"], road["width_m"] / 2.0) for road in city["roads"] if road["vertical"] != "underground"]
    grounds += [corridor(item["points"], item.get("width_m", RAIL_WIDTH) / 2.0) for item in city["rail"] if item["vertical"] != "underground"]
    on_lines = 0
    if grounds:
        ground_tree = STRtree(grounds)
        on_lines = sum(any(shape.intersection(grounds[index]).area > 1.0 for index in ground_tree.query(shape, predicate="intersects")) for shape in shapes)
    water = [polygon_of(item) for item in city["water"]]
    in_water = 0
    if water:
        water_tree = STRtree(water)
        in_water = sum(any(shape.intersection(water[index]).area > 1.0 for index in water_tree.query(shape, predicate="intersects")) for shape in shapes)
    return {"off_block_ground": int(off_land), "listed_under_the_wrong_block": int(wrong_block), "overlapping_pairs": int(overlaps), "on_roads_or_rail": int(on_lines), "in_water": int(in_water)}


def block_overlaps(city):
    shapes = [polygon_of(item) for item in city["blocks"]]
    tree = STRtree(shapes)
    return int(sum(1 for index, shape in enumerate(shapes) for other in tree.query(shape, predicate="intersects") if other > index and shape.intersection(shapes[other]).area > 1.0))


def blocks_on_roads(city):
    paved = [corridor(road["points"], road["width_m"] / 2.0) for road in city["roads"] if road["vertical"] in ("surface", "unknown")]
    if not paved:
        return 0
    tree = STRtree(paved)
    total = 0.0
    for item in city["blocks"]:
        shape = polygon_of(item)
        total += sum(shape.intersection(paved[index]).area for index in tree.query(shape, predicate="intersects"))
    return int(round(total))


def roads_over_water(city):
    water = [polygon_of(item) for item in city["water"]]
    if not water:
        return 0
    tree = STRtree(water)
    total = 0.0
    for road in city["roads"]:
        if road["vertical"] not in ("surface", "unknown"):
            continue
        line = LineString(road["points"])
        wet = sum(line.intersection(water[index]).length for index in tree.query(line, predicate="intersects"))
        total += max(wet - sum(end - start for start, end in road.get("bridges", [])), 0.0)
    return int(round(total))


def graph_problems(city, reach=0.5):
    nodes = {item["id"]: np.asarray(item["xy"], dtype=np.float64) for item in city["nodes"]}
    missing = 0
    loose = 0
    degree = {key: 0 for key in nodes}
    for edge in city["edges"]:
        if edge["from"] not in nodes or edge["to"] not in nodes:
            missing += 1
            continue
        degree[edge["from"]] += 1
        degree[edge["to"]] += 1
        points = np.asarray(edge["points"], dtype=np.float64)
        ends = (np.linalg.norm(points[0] - nodes[edge["from"]]), np.linalg.norm(points[-1] - nodes[edge["to"]]))
        flipped = (np.linalg.norm(points[0] - nodes[edge["to"]]), np.linalg.norm(points[-1] - nodes[edge["from"]]))
        loose += int(max(ends) > reach and max(flipped) > reach)
    wrong_degree = sum(1 for item in city["nodes"] if item["degree"] != degree[item["id"]])
    return {"edges_to_missing_nodes": int(missing), "edges_not_ending_on_their_nodes": int(loose), "wrong_node_degree": int(wrong_degree), "duplicate_node_ids": len(city["nodes"]) - len(nodes)}


def unjoined_crossings(city, reach=1.0):
    by_level = {}
    for road in city["roads"]:
        by_level.setdefault(road["vertical"], []).append(LineString(road["points"]))
    node_points = shapely.points(np.asarray([item["xy"] for item in city["nodes"] if item["mode"] == "road"], dtype=np.float64).reshape(-1, 2))
    node_tree = STRtree(node_points) if len(node_points) else None
    missing = 0
    for level, lines in by_level.items():
        if level == "unknown":
            continue
        tree = STRtree(lines)
        for index, line in enumerate(lines):
            for other in tree.query(line, predicate="crosses"):
                if other <= index:
                    continue
                crossing = line.intersection(lines[other])
                for point in getattr(crossing, "geoms", [crossing]):
                    if point.geom_type != "Point":
                        continue
                    if node_tree is None or not len(node_tree.query(point, predicate="dwithin", distance=reach)):
                        missing += 1
    return int(missing)


def near_misses(city, reach=3.0):
    lines = [(road["vertical"], LineString(road["points"])) for road in city["roads"]]
    tree = STRtree([line for _level, line in lines])
    count = 0
    for index, (level, line) in enumerate(lines):
        for end in (Point(line.coords[0]), Point(line.coords[-1])):
            close = [other for other in tree.query(end, predicate="dwithin", distance=reach) if other != index and lines[other][0] == level]
            if close and min(lines[other][1].distance(end) for other in close) > 0.05:
                count += 1
    return count


def hanging_flyovers(city, reach=30.0):
    width, height = city["size_m"]
    ground = [LineString(road["points"]) for road in city["roads"] if road["vertical"] != "underground"]
    if not ground:
        return 0
    tree = STRtree(ground)
    count = 0
    for road in city["roads"]:
        if road["vertical"] != "elevated":
            continue
        line = LineString(road["points"])
        for x, y in (line.coords[0], line.coords[-1]):
            if min(x, y, width - x, height - y) < 5.0:
                continue
            point = Point(x, y)
            count += int(not any(ground[index].distance(point) < reach and not ground[index].equals(line) for index in tree.query(point, predicate="dwithin", distance=reach)))
    return count


def tile_seams(city, seam=4.0):
    degree = {item["id"]: item["degree"] for item in city["nodes"]}
    width, height = city["size_m"]
    count = 0
    for item in city["nodes"]:
        if item["mode"] != "road" or degree[item["id"]] != 1:
            continue
        x, y = item["xy"]
        on_x = abs(x / TILE - round(x / TILE)) * TILE <= seam and seam < x < width - seam
        on_y = abs(y / TILE - round(y / TILE)) * TILE <= seam and seam < y < height - seam
        count += int(on_x or on_y)
    return count


def rail_problems(city, reach=1.0):
    width, height = city["size_m"]
    runs = {}
    for item in city["rail"]:
        runs.setdefault(item.get("line"), []).append(np.asarray(item["points"], dtype=np.float64))
    broken = 0
    stops_inside = 0
    for number, pieces in runs.items():
        if number is None:
            continue
        for before, after in zip(pieces[:-1], pieces[1:]):
            broken += int(np.linalg.norm(before[-1] - after[0]) > reach)
        for x, y in (pieces[0][0], pieces[-1][-1]):
            stops_inside += int(min(x, y, width - x, height - y) > reach)
    lines = {number: LineString(np.vstack(pieces)) for number, pieces in runs.items() if number is not None}
    off_line = 0
    for station in city.get("stations", []):
        point = Point(station["xy"])
        off_line += int(any(number not in lines or lines[number].distance(point) > reach for number in station.get("lines", [station.get("line")])))
    return {"breaks_between_runs": int(broken), "lines_stopping_inside_the_map": int(stops_inside), "stations_off_their_line": int(off_line)}


def cover_problems(city):
    water = [polygon_of(item) for item in city["water"]]
    natural = [polygon_of(item) for item in city["green"] if item.get("kind") == "natural"]
    park = [polygon_of(item) for item in city["green"] if item.get("kind") != "natural"]

    def shared(first, second):
        if not first or not second:
            return 0.0
        tree = STRtree(second)
        return float(sum(shape.intersection(second[index]).area for shape in first for index in tree.query(shape, predicate="intersects")))

    return {"natural_on_water_m2": round(shared(natural, water)), "natural_on_park_m2": round(shared(natural, park))}


def check_city(city, plan=None):
    width, height = city["size_m"]
    km2 = width * height / 1e6
    report = {
        "size_m": [width, height],
        "frame": {"units": city.get("units"), "axes": city.get("axes")},
        "points_outside_the_map": points_outside(city),
        "bad_shapes": bad_shapes(city),
        "buildings": building_problems(city),
        "overlapping_blocks": block_overlaps(city),
        "blocks_on_roads_m2": blocks_on_roads(city),
        "roads_on_the_ground_over_water_m": roads_over_water(city),
        "graph": graph_problems(city),
        "crossings_without_a_junction": unjoined_crossings(city),
        "rail": rail_problems(city),
        "rules": city.get("rules", ["rail", "stations", "road joins"]),
        "cover": cover_problems(city),
        "tile_seams_per_km2": round(tile_seams(city) / km2, 2),
        "near_miss_ends_per_km2": round(near_misses(city) / km2, 2),
        "elevated_ends_with_no_road_near": hanging_flyovers(city),
    }
    if plan is not None:
        expected = [int(round(height / city["plan"]["cell_m"])), int(round(width / city["plan"]["cell_m"]))]
        report["plan_cells_match_size"] = list(plan.shape[1:]) == expected
        report["plan_orientation"] = plan_orientation(city, plan)
    return report


def failures(report):
    found = []
    learned = ("breaks_between_runs", "lines_stopping_inside_the_map") if "rail" not in report.get("rules", ["rail"]) else ()
    for name in ("points_outside_the_map", "bad_shapes", "buildings", "graph", "rail"):
        found += [f"{name}.{key} = {value}" for key, value in report[name].items() if value and key not in learned]
    for name in ("overlapping_blocks", "crossings_without_a_junction"):
        if report[name]:
            found.append(f"{name} = {report[name]}")
    if report["roads_on_the_ground_over_water_m"] > 50:
        found.append(f"roads_on_the_ground_over_water_m = {report['roads_on_the_ground_over_water_m']}")
    if report["blocks_on_roads_m2"] > 100:
        found.append(f"blocks_on_roads_m2 = {report['blocks_on_roads_m2']}")
    if report["cover"]["natural_on_water_m2"] > 1000 or report["cover"]["natural_on_park_m2"] > 1000:
        found.append(f"cover = {report['cover']}")
    if report["frame"] != {"units": "metres", "axes": "x east, y north, z up"}:
        found.append(f"frame = {report['frame']}")
    if "plan_orientation" in report:
        scores = report["plan_orientation"]
        if not report["plan_cells_match_size"]:
            found.append("plan cells do not match the city size")
        if max(scores, key=scores.get) != "as is":
            found.append(f"plan_orientation = {scores}")
    return found
