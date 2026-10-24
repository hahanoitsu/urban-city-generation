from __future__ import annotations

import math

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import LineString, box
from shapely.ops import substring

from urban_layers.data import CELL, CHANNELS, LINE_CLASSES

# rail is drawn for the whole city at once. the plan only says roughly where rail is (blobs of a few cells),
# the street layer turned that into short stubs, real lines run right across the map
TRACKS = 2
RAIL_WIDTH = 10.0


def plan_grid(plan, name, sigma=1.0):
    return ndimage.gaussian_filter(plan[CHANNELS.index(name)].astype(np.float64), sigma)


def lookup(grid, points):
    # bilinear value at points in metres, cell centres are at (index + 0.5) * CELL
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    return ndimage.map_coordinates(grid, [points[:, 1] / CELL - 0.5, points[:, 0] / CELL - 0.5], order=1, mode="nearest")


def line_density(points, shape, tracks=TRACKS):
    # the plan's rail value for a line, track length in a cell over 2 cells
    grid = np.zeros(shape, dtype=np.float64)
    dense = np.asarray(shapely.segmentize(LineString(points), 8.0).coords)
    middle = (dense[1:] + dense[:-1]) / 2.0
    length = np.linalg.norm(dense[1:] - dense[:-1], axis=1)
    rows = np.clip((middle[:, 1] // CELL).astype(int), 0, shape[0] - 1)
    columns = np.clip((middle[:, 0] // CELL).astype(int), 0, shape[1] - 1)
    np.add.at(grid, (rows, columns), length * tracks / (2.0 * CELL))
    return grid


def best_heading(centre, pull, reach=1024.0):
    # the straight line through the centre that collects the most pull
    distances = np.concatenate([-np.arange(64.0, reach + 1, 64.0), np.arange(64.0, reach + 1, 64.0)])[:, None]
    options = np.arange(0.0, math.pi, math.pi / 24)
    scores = [lookup(pull, centre + distances * np.array([math.cos(option), math.sin(option)])).sum() for option in options]
    return float(options[int(np.argmax(scores))])


def walk(start, heading, pull, water, size, step=64.0, turn=0.08, limit=200):
    # follow the pull until the line leaves the map. the heading moves at most turn per step (a curve of
    # about 800 m radius) and never more than 90 degrees from where it set off, so it cannot loop back
    points = [np.asarray(start, dtype=np.float64)]
    first = heading
    ahead = np.array([128.0, 256.0, 384.0, 512.0])[:, None]
    area = box(0.0, 0.0, *size)
    for _step in range(limit):
        best = None
        for option in heading + np.linspace(-turn, turn, 5):
            if abs((option - first + math.pi) % (2 * math.pi) - math.pi) > math.pi / 2:
                continue
            spots = points[-1] + ahead * np.array([math.cos(option), math.sin(option)])
            score = lookup(pull, spots).sum() - 3.0 * lookup(water, spots).sum() - 2.0 * abs(option - heading)
            if best is None or score > best[0]:
                best = (score, option)
        heading = best[1]
        point = points[-1] + step * np.array([math.cos(heading), math.sin(heading)])
        if 0.0 <= point[0] <= size[0] and 0.0 <= point[1] <= size[1]:
            points.append(point)
            continue
        piece = LineString([points[-1], point]).intersection(area)
        coords = np.asarray(piece.coords) if not piece.is_empty and piece.geom_type == "LineString" else np.zeros((0, 2))
        if len(coords):
            far = coords[np.argmax(np.linalg.norm(coords - points[-1], axis=1))]
            if np.linalg.norm(far - points[-1]) > 1.0:
                points.append(far)
        break
    return np.asarray(points)


def smooth_line(points, rounds=2):
    # chaikin corner cutting that keeps both ends where they are
    points = np.asarray(points, dtype=np.float64)
    for _round in range(rounds):
        if len(points) < 3:
            break
        cut = np.stack([0.75 * points[:-1] + 0.25 * points[1:], 0.25 * points[:-1] + 0.75 * points[1:]], axis=1).reshape(-1, 2)
        points = np.vstack([points[:1], cut[1:-1], points[-1:]])
    return points


def rail_corridors(plan, size, threshold=0.08, smallest=400.0, most=4):
    # take the strongest rail blob, run a line through it to both map edges, take away what that line
    # explains and go again, until the lines carry the track the plan asks for. a hub with two lines
    # crossing leaves a blob after the first line, so it gets a second one
    residual = plan_grid(plan, "rail", 0.7)
    # away from the blobs a line leans towards big roads and busy ground, that is where stations go
    leaning = 0.3 * plan_grid(plan, "major", 1.0) + 0.3 * (plan_grid(plan, "coverage", 2.0) + plan_grid(plan, "commercial", 2.0))
    water = plan_grid(plan, "water", 1.0)
    wanted = float(plan[CHANNELS.index("rail")].sum()) * 2.0 * CELL / TRACKS
    lines = []
    total = 0.0
    while len(lines) < most and total < 0.8 * wanted:
        labels, count = ndimage.label(residual > threshold, structure=np.ones((3, 3)))
        if count == 0:
            break
        masses = ndimage.sum(residual, labels, range(1, count + 1))
        label = int(np.argmax(masses)) + 1
        if masses[label - 1] * 2.0 * CELL < smallest:
            break
        rows, columns = np.nonzero(labels == label)
        weights = residual[rows, columns]
        centre = np.array([((columns + 0.5) * CELL * weights).sum(), ((rows + 0.5) * CELL * weights).sum()]) / weights.sum()
        pull = residual + leaning
        heading = best_heading(centre, pull)
        forward = walk(centre, heading, pull, water, size)
        backward = walk(centre, heading + math.pi, pull, water, size)
        points = smooth_line(np.vstack([backward[::-1], forward[1:]]))
        line = LineString(points)
        near = sum(line.intersection(LineString(other).buffer(150.0)).length for other in lines)
        if line.length < 500.0 or near > 0.5 * line.length:
            # mostly runs next to a line already there, this blob is used up
            residual[labels == label] = 0.0
            continue
        residual = np.clip(residual - ndimage.gaussian_filter(line_density(points, residual.shape), 0.7), 0.0, None)
        lines.append(points)
        total += line.length
    return lines


def rail_runs(lines, plan, dense=0.35, step=64.0, hold=9):
    # elevated, and underground where the plan is dense (most of singapore's lines go under the centre).
    # a level has to hold for about 600 m so a line does not keep popping up and down
    density = plan_grid(plan, "coverage", 1.0) + plan_grid(plan, "commercial", 1.0)
    runs = []
    for number, points in enumerate(lines):
        line = LineString(points)
        spots = np.minimum(np.arange(0.0, line.length + step, step), line.length)
        xy = shapely.get_coordinates(shapely.line_interpolate_point(line, spots))
        under = ndimage.median_filter((lookup(density, xy) > dense).astype(int), size=hold, mode="nearest") > 0
        changes = [index for index in range(1, len(under)) if under[index] != under[index - 1]]
        cuts = [0.0] + [float(spots[index] + spots[index - 1]) / 2.0 for index in changes] + [line.length]
        levels = [under[0]] + [under[index] for index in changes]
        for low, high, down in zip(cuts[:-1], cuts[1:], levels):
            piece = substring(line, low, high)
            if piece.geom_type == "LineString" and piece.length > 1.0:
                runs.append((LINE_CLASSES.index(("rail", "subway", "underground" if down else "elevated")), np.asarray(piece.coords), number))
    return runs


def rail_stations(lines, plan, spacing=1100.0, search=300.0):
    # one stop about every 1.1 km, moved to the busiest spot nearby
    density = plan_grid(plan, "coverage", 1.0) + plan_grid(plan, "commercial", 1.0) + 0.5 * plan_grid(plan, "residential", 1.0)
    stations = []
    for number, points in enumerate(lines):
        line = LineString(points)
        position = spacing / 2.0
        while position < line.length - 200.0:
            options = np.arange(max(position - search, 150.0), min(position + search, line.length - 150.0) + 1.0, 25.0)
            if not len(options):
                break
            xy = shapely.get_coordinates(shapely.line_interpolate_point(line, options))
            best = float(options[int(np.argmax(lookup(density, xy)))])
            here = np.asarray(line.interpolate(best).coords[0])
            ahead = np.asarray(line.interpolate(min(best + 10.0, line.length)).coords[0])
            heading = math.degrees(math.atan2(ahead[1] - here[1], ahead[0] - here[0]))
            stations.append({"xy": here.tolist(), "line": number, "heading_deg": heading})
            position = best + spacing
    return stations
