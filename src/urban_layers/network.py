from __future__ import annotations

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from shapely.strtree import STRtree

from urban_layers.audit import road_pieces
from urban_layers.data import CELL, CHANNELS, LINE_CLASSES, TILE, ROAD_CLASSES, line_class
from urban_layers.rail import smooth_line


def level(class_id):
    value = LINE_CLASSES[class_id][2]
    return "surface" if value == "unknown" else value


def rank(class_id):
    return ROAD_CLASSES.index(LINE_CLASSES[class_id][1])


def labels(lines):
    return road_pieces([{"points": points, "vertical": LINE_CLASSES[class_id][2]} for class_id, points in lines])


def end_of(points, slot):
    return points[0] if slot == 0 else points[-1]


def free_ends(lines, tree, touch=1.0):
    found = []
    for index, (class_id, points) in enumerate(lines):
        if np.linalg.norm(points[0] - points[-1]) <= touch:
            continue
        mine = level(class_id)
        for slot in (0, 1):
            point = end_of(points, slot)
            joined = False
            for other in tree.query(Point(point), predicate="dwithin", distance=touch):
                if other == index:
                    continue
                theirs = lines[other][1]
                if level(lines[other][0]) == mine or mine != "surface" or min(np.linalg.norm(theirs[0] - point), np.linalg.norm(theirs[-1] - point)) <= touch:
                    joined = True
                    break
            if not joined:
                found.append((index, slot))
    return found


def with_vertex(points, point):
    along = LineString(points).project(Point(point))
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    index = int(np.clip(np.searchsorted(cumulative, along), 1, len(points) - 1))
    if min(np.linalg.norm(points[index - 1] - point), np.linalg.norm(points[index] - point)) < 1e-6:
        return points
    return np.insert(points, index, point, axis=0)


def split_at(lines, cuts):
    out = []
    for index, (class_id, points) in enumerate(lines):
        if index not in cuts:
            out.append((class_id, points))
            continue
        for point in cuts[index]:
            points = with_vertex(points, point)
        marks = [k for k in range(1, len(points) - 1) if any(np.linalg.norm(points[k] - point) < 1e-6 for point in cuts[index])]
        start = 0
        for mark in marks:
            out.append((class_id, points[start : mark + 1]))
            start = mark
        out.append((class_id, points[start:]))
    return out


def extend(points, slot, target):
    if np.linalg.norm(end_of(points, slot) - target) < 1e-6:
        return points
    return np.vstack([target[None], points]) if slot == 0 else np.vstack([points, target[None]])


def onto(shape, points, point, touch=1.0):
    target = np.asarray(nearest_points(shape, Point(point))[0].coords[0])
    for end in (points[0], points[-1]):
        if np.linalg.norm(end - target) <= touch:
            return np.asarray(end, dtype=np.float64)
    return target


def land_ramps(lines, reach=40.0, touch=1.0):
    shapes = [LineString(points) for _class_id, points in lines]
    tree = STRtree(shapes)
    cuts = {}
    landed = 0
    for index, slot in free_ends(lines, tree, touch):
        class_id, points = lines[index]
        mine = level(class_id)
        point = end_of(points, slot)
        best = None
        for other in tree.query(Point(point), predicate="dwithin", distance=reach if mine != "surface" else 5.0):
            theirs = level(lines[other][0])
            if other == index or theirs == mine:
                continue
            if mine == "surface":
                target = min((lines[other][1][0], lines[other][1][-1]), key=lambda end: np.linalg.norm(end - point))
                if np.linalg.norm(target - point) > 5.0:
                    continue
            else:
                target = onto(shapes[other], lines[other][1], point, touch)
            order = (theirs != "surface", float(np.linalg.norm(target - point)))
            if best is None or order < best[0]:
                best = (order, other, np.asarray(target, dtype=np.float64))
        if best is None:
            continue
        _order, other, target = best
        lines[index] = (class_id, extend(points, slot, target))
        cuts.setdefault(other, []).append(target)
        landed += 1
    return split_at(lines, cuts), landed


def stitch(lines, size, reach=40.0, seam=4.0, touch=1.0):
    width, height = size
    shapes = [LineString(points) for _class_id, points in lines]
    tree = STRtree(shapes)
    stitched = 0
    for index, slot in free_ends(lines, tree, touch):
        class_id, points = lines[index]
        point = end_of(points, slot)
        x, y = point
        on_seam = (abs(x / TILE - round(x / TILE)) * TILE <= seam and seam < x < width - seam) or (abs(y / TILE - round(y / TILE)) * TILE <= seam and seam < y < height - seam)
        major = rank(class_id) == 0
        if not on_seam and not major:
            continue
        heading = point - (points[1] if slot == 0 else points[-2])
        heading = heading / max(np.linalg.norm(heading), 1e-9)
        best = None
        for other in tree.query(Point(point), predicate="dwithin", distance=reach):
            if other == index or level(lines[other][0]) != level(class_id):
                continue
            if not on_seam and rank(lines[other][0]) != 0:
                continue
            target = onto(shapes[other], lines[other][1], point, touch)
            step = target - point
            gap = float(np.linalg.norm(step))
            if gap < 1e-6 or np.dot(step / gap, heading) < 0.5:
                continue
            if best is None or gap < best[0]:
                best = (gap, other, target)
        if best is None:
            continue
        _gap, other, target = best
        bridge = LineString([point, target])
        if any(k not in (index, other) for k in tree.query(bridge, predicate="intersects")):
            continue
        lines[index] = (class_id, extend(points, slot, target))
        stitched += 1
    return lines, stitched


def drop_strays(lines, small=200.0, touch=30.0, fragment=1000.0, remote=300.0):
    shapes, label = labels(lines)
    size = np.bincount(label, weights=shapely.length(shapes))
    tree = STRtree(shapes)
    solid = size >= fragment
    solid[np.argmax(size)] = True
    drop = set()
    for piece in np.nonzero(~solid)[0]:
        members = np.nonzero(label == piece)[0]
        near = label[tree.query(shapes[members], predicate="dwithin", distance=remote)[1]]
        if not np.any(solid[near[near != piece]]):
            drop.add(int(piece))
            continue
        close = label[tree.query(shapes[members], predicate="dwithin", distance=touch)[1]]
        if size[piece] < small and not np.any(close != piece):
            drop.add(int(piece))
    return [line for line, piece in zip(lines, label) if int(piece) not in drop], len(drop)


def cost_grid(plan):
    cost = 1.0 + 40.0 * plan[CHANNELS.index("water")] + 0.3 * plan[CHANNELS.index("green")]
    rows, columns = cost.shape
    index = np.arange(rows * columns).reshape(rows, columns)
    first, second, weight = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = index[0 : rows - dy, max(0, -dx) : columns - max(0, dx)]
        b = index[dy:rows, max(0, -dx) + dx : columns - max(0, dx) + dx]
        step = np.hypot(dy, dx)
        first.append(a.ravel())
        second.append(b.ravel())
        weight.append(step * (cost.ravel()[a.ravel()] + cost.ravel()[b.ravel()]) / 2.0)
    first, second, weight = np.concatenate(first), np.concatenate(second), np.concatenate(weight)
    graph = coo_matrix((weight, (first, second)), shape=(rows * columns, rows * columns)).tocsr()
    return graph.maximum(graph.T), (rows, columns)


def route(start, end, grid):
    graph, (rows, columns) = grid
    cell = lambda point: int(np.clip(point[1] // CELL, 0, rows - 1)) * columns + int(np.clip(point[0] // CELL, 0, columns - 1))
    first, last = cell(start), cell(end)
    _distance, previous = dijkstra(graph, indices=first, return_predecessors=True)
    path = [last]
    while path[-1] != first and previous[path[-1]] >= 0:
        path.append(int(previous[path[-1]]))
    middle = [((number % columns + 0.5) * CELL, (number // columns + 0.5) * CELL) for number in reversed(path[1:-1])]
    points = np.asarray([start, *middle, end], dtype=np.float64)
    points = np.asarray(LineString(points).simplify(CELL / 2.0).coords)
    return smooth_line(points, 3)


def spanning_links(lines, members, water, big, reach, extra, apart, close):
    members = np.asarray(list(members))
    shapes, label = labels([lines[index] for index in members])
    size = np.bincount(label, weights=shapely.length(shapes))
    giant = int(np.argmax(size))
    surface = np.asarray([level(lines[index][0]) == "surface" for index in members])
    ranks = np.asarray([rank(lines[index][0]) for index in members])
    tree = STRtree(shapes)
    contacts = {}

    def gather(indices, distance):
        if not len(indices):
            return
        a, b = tree.query(shapes[indices], predicate="dwithin", distance=distance)
        a = indices[a]
        keep = surface[b] & (label[a] != label[b])
        a, b = a[keep], b[keep]
        gaps = shapely.distance(shapes[a], shapes[b])
        for i, j, gap in zip(a.tolist(), b.tolist(), gaps.tolist()):
            key = (min(label[i], label[j]), max(label[i], label[j]))
            contacts.setdefault(key, []).append((gap + 50.0 * (ranks[i] + ranks[j]), gap, i, j))

    candidates = np.nonzero(surface & (label != giant))[0]
    gather(candidates, reach[0])
    touched = {piece for key in contacts for piece in key}
    gather(candidates[np.asarray([label[i] not in touched for i in candidates], dtype=bool)], reach[1])

    parent = list(range(len(size)))

    def find(piece):
        while parent[piece] != piece:
            parent[piece] = parent[parent[piece]]
            piece = parent[piece]
        return piece

    def water_share(i, j):
        segment = LineString(nearest_points(shapes[i], shapes[j]))
        return segment.intersection(water).length / max(segment.length, 1e-9) if water is not None and not water.is_empty else 0.0

    best = {key: min(items) for key, items in contacts.items()}
    order = sorted(best.items(), key=lambda item: item[1][0] * (1.0 + 4.0 * (water_share(item[1][2], item[1][3]) if item[1][1] > 50.0 else 0.0)))
    chosen = []
    for (a, b), (_score, gap, i, j) in order:
        if find(a) != find(b):
            parent[find(a)] = find(b)
            chosen.append((gap, i, j, a, b))

    giant_lines = np.nonzero(surface & (label == giant))[0]
    giant_tree = STRtree(shapes[giant_lines])
    for piece in range(len(size)):
        if find(piece) == find(giant) or not np.any(surface & (label == piece)):
            continue
        group = [other for other in range(len(size)) if find(other) == find(piece)]
        inside = np.nonzero(surface & np.isin(label, group))[0]
        (k, m), gaps = giant_tree.query_nearest(shapes[inside], return_distance=True, all_matches=False)
        pick = int(np.argmin(gaps))
        i, j = int(inside[k[pick]]), int(giant_lines[m[pick]])
        parent[find(piece)] = find(giant)
        chosen.append((float(gaps[pick]), i, j, int(label[i]), giant))

    for (a, b), items in contacts.items():
        if size[a] < big or size[b] < big:
            continue
        spots = [np.asarray(LineString(nearest_points(shapes[i], shapes[j])).centroid.coords[0]) for _gap, i, j, pa, pb in chosen if {pa, pb} == {a, b}]
        added = 0
        for _score, gap, i, j in sorted(items):
            if added >= extra:
                break
            if gap > close:
                continue
            middle = np.asarray(LineString(nearest_points(shapes[i], shapes[j])).centroid.coords[0])
            if all(np.linalg.norm(middle - spot) >= apart for spot in spots):
                chosen.append((gap, i, j, a, b))
                spots.append(middle)
                added += 1
    return [(gap, int(members[i]), int(members[j]), bool(size[a] >= big and size[b] >= big)) for gap, i, j, a, b in chosen]


def draw_links(lines, chosen, plan, name):
    grid = None
    new = []
    for gap, i, j, big_pair in chosen:
        start, end = (np.asarray(point.coords[0]) for point in nearest_points(LineString(lines[i][1]), LineString(lines[j][1])))
        if gap <= 400.0:
            points = np.asarray([start, end], dtype=np.float64)
        else:
            grid = grid or cost_grid(plan)
            points = route(start, end, grid)
        if np.linalg.norm(points[0] - points[-1]) < 1e-6:
            continue
        new.append((line_class("road", name(i, j, gap, big_pair), "surface"), points))
    return new


def join_majors(lines, plan, water, shortest=600.0, alone=100.0):
    majors = [index for index, (class_id, _points) in enumerate(lines) if rank(class_id) == 0]
    if not majors:
        return lines, 0, []
    shapes, label = labels([lines[index] for index in majors])
    size = np.bincount(label, weights=shapely.length(shapes))
    tree = STRtree(shapes)
    lone = set()
    for piece in np.nonzero(size < shortest)[0]:
        near = tree.query(shapes[label == piece], predicate="dwithin", distance=alone)[1]
        if not np.any(label[near] != piece):
            lone.add(int(piece))
    for local, index in enumerate(majors):
        if int(label[local]) in lone:
            class_id, points = lines[index]
            lines[index] = (line_class("road", "secondary", LINE_CLASSES[class_id][2]), points)
    kept = [index for local, index in enumerate(majors) if int(label[local]) not in lone]
    if len(kept) < 2:
        return lines, len(lone), []
    chosen = spanning_links(lines, kept, water, big=5000.0, reach=(300.0, 3000.0), extra=2, apart=2000.0, close=600.0)
    new = draw_links(lines, chosen, plan, lambda i, j, gap, big_pair: "major")
    return lines + new, len(lone), new


def link_pieces(lines, plan, water):
    chosen = spanning_links(lines, range(len(lines)), water, big=5000.0, reach=(150.0, 1500.0), extra=3, apart=1500.0, close=300.0)
    name = lambda i, j, gap, big_pair: ("major" if big_pair else "secondary") if gap > 400.0 else ROAD_CLASSES[max(rank(lines[i][0]), rank(lines[j][0]))]
    new = draw_links(lines, chosen, plan, name)
    return lines + new, new


def join_network(lines, plan, size, water=None):
    lines = [(class_id, np.asarray(points, dtype=np.float64)) for class_id, points in lines]
    lines = [(class_id, points) for class_id, points in lines if len(points) >= 2 and LineString(points).length > 0.5]
    if not lines:
        return lines, {}
    lines, landed = land_ramps(lines)
    lines, stitched = stitch(lines, size)
    lines, dropped = drop_strays(lines)
    lines, downgraded, major_links = join_majors(lines, plan, water)
    lines, links = link_pieces(lines, plan, water)
    km = lambda new: sum(LineString(points).length for _class_id, points in new) / 1000.0
    return lines, {"ramps": landed, "stitched": stitched, "dropped": dropped, "downgraded": downgraded, "major_links": len(major_links), "major_link_km": km(major_links), "links": len(links), "link_km": km(links)}
