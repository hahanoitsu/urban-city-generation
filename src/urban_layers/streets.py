from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from urban_layers.data import CELL, CHANNELS, LINE_CLASSES, MARGIN, TILE, boundary_side, touch_lists
from urban_layers.decoder import TokenDecoder, fourier
from urban_layers.tokens import StreetGrammar, StreetTokens

PLAN_CELLS = 8
CONTEXT_POINTS = 256
PORTS = 64
D4 = [
    np.array([[1, 0], [0, 1]]),
    np.array([[0, -1], [1, 0]]),
    np.array([[-1, 0], [0, -1]]),
    np.array([[0, 1], [-1, 0]]),
    np.array([[-1, 0], [0, 1]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[1, 0], [0, -1]]),
    np.array([[0, -1], [-1, 0]]),
]
SIDES = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}
PATTERNS = [
    set(),
    {(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)},
    {(-1, 0), (-1, -1), (0, -1)},
    {(-1, 0), (-1, -1), (0, -1), (1, -1)},
    {(-1, 0)},
    {(0, -1)},
    {(0, -1), (1, -1)},
]
PRIORITY = {"major": 0, "secondary": 1, "local": 2}


def plan_crop(plan, row, column, size=PLAN_CELLS):
    border = (size - int(TILE / CELL)) // 2
    crop = np.zeros((plan.shape[0], size, size), dtype=np.float32)
    top, left = row - border, column - border
    rows = slice(max(top, 0), min(top + size, plan.shape[1]))
    columns = slice(max(left, 0), min(left + size, plan.shape[2]))
    crop[:, rows.start - top : rows.stop - top, columns.start - left : columns.stop - left] = plan[:, rows, columns]
    return crop


def move_points(points, matrix):
    centre = TILE / 2.0
    return ((np.asarray(points, dtype=np.float64) - centre) @ matrix.T + centre).astype(np.float32)


def move_grid(grid, matrix):
    size = grid.shape[1]
    index = np.arange(size)
    rows, columns = np.meshgrid(index, index, indexing="ij")
    centres = np.stack([columns + 0.5, rows + 0.5], axis=-1) - size / 2.0
    source = centres @ matrix + size / 2.0 - 0.5
    source = np.rint(source).astype(int)
    return grid[:, source[..., 1], source[..., 0]]


def region(points):
    middle = np.asarray(points).mean(axis=0)
    dx = -1 if middle[0] < 0 else (1 if middle[0] > TILE else 0)
    dy = -1 if middle[1] < 0 else (1 if middle[1] > TILE else 0)
    return dx, dy


def heading(points, start=True):
    points = np.asarray(points, dtype=np.float64)
    a, b = (points[0], points[1]) if start else (points[-1], points[-2])
    step = b - a
    length = np.linalg.norm(step)
    return step / length if length > 1e-6 else np.zeros(2)


def port_key(port):
    side = boundary_side((port[0], port[1]))
    return side, round(port[1] if side in (0, 2) else port[0], 3)


def find_ports(lines, visible):
    ports = []
    for class_id, points in lines:
        for start in (True, False):
            point = points[0] if start else points[-1]
            side = boundary_side(point)
            if side >= 0 and SIDES[side] in visible:
                ports.append((float(point[0]), float(point[1]), *heading(points, start), class_id))
    ports.sort(key=port_key)
    return ports


def rank(class_id):
    mode, name, _level = LINE_CLASSES[class_id]
    return PRIORITY.get(name, 3) if mode == "road" else 3


def order_lines(lines, visible, ports=None, touches=None):
    # lines that continue a port come first, in port order, so generation can force their start.
    # then the network grows: the next line starts on one already drawn if it can, bigger roads first.
    # an end that touches another line inside the tile is marked as a join
    ports = find_ports(lines, visible) if ports is None else ports
    touches = touch_lists(lines) if touches is None else touches
    spots = np.asarray([(port[0], port[1]) for port in ports], dtype=np.float64).reshape(-1, 2)

    def port_index(point):
        if not len(spots):
            return None
        distance = np.linalg.norm(spots - np.asarray(point, dtype=np.float64), axis=1)
        index = int(np.argmin(distance))
        return index if distance[index] < 0.5 else None

    order = []
    placed = set()
    starts = []
    for index, (_class_id, points) in enumerate(lines):
        first, last = port_index(points[0]), port_index(points[-1])
        if first is not None or last is not None:
            flip = first is None or (last is not None and last < first)
            starts.append((last if flip else first, index, flip))
    for _port, index, flip in sorted(starts):
        order.append((index, flip, True))
        placed.add(index)

    while len(placed) < len(lines):
        best = None
        for index, (class_id, points) in enumerate(lines):
            if index in placed:
                continue
            start_on = any(other in placed for other in touches[index][0])
            end_on = any(other in placed for other in touches[index][1])
            if start_on or end_on:
                flip = not start_on
            else:
                flip = (round(float(points[-1][1])), round(float(points[-1][0]))) < (round(float(points[0][1])), round(float(points[0][0])))
            start = points[-1] if flip else points[0]
            key = (rank(class_id), 0 if start_on or end_on else 1, round(float(start[1])), round(float(start[0])))
            if best is None or key < best[0]:
                best = (key, index, flip)
        _key, index, flip = best
        order.append((index, flip, False))
        placed.add(index)

    result = []
    for index, flip, from_port in order:
        class_id, points = lines[index]
        points = np.asarray(points, dtype=np.float32)
        start_touch, end_touch = touches[index]
        if flip:
            points = points[::-1]
            start_touch, end_touch = end_touch, start_touch
        start_join = bool(start_touch) and not from_port and boundary_side(points[0]) < 0
        end_join = bool(end_touch) and boundary_side(points[-1]) < 0
        result.append((class_id, points, (start_join, end_join)))
    return result


def resample(points, spacing=16.0):
    points = np.asarray(points, dtype=np.float64)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    if cumulative[-1] < 1e-6:
        return points[:1], np.zeros((1, 2))
    count = max(2, int(np.ceil(cumulative[-1] / spacing)) + 1)
    distances = np.linspace(0.0, cumulative[-1], count)
    xs = np.interp(distances, cumulative, points[:, 0])
    ys = np.interp(distances, cumulative, points[:, 1])
    result = np.stack([xs, ys], axis=1)
    direction = np.gradient(result, axis=0)
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
    return result, direction


def condition_arrays(crop, context, ports, city):
    plan = crop.reshape(crop.shape[0], -1).T.astype(np.float32)
    index = np.arange(PLAN_CELLS)
    rows, columns = np.meshgrid(index, index, indexing="ij")
    border = (PLAN_CELLS - int(TILE / CELL)) // 2
    plan_xy = np.stack([(columns - border + 0.5) * CELL, (rows - border + 0.5) * CELL], axis=-1).reshape(-1, 2) / TILE

    context_xy = np.zeros((CONTEXT_POINTS, 2), dtype=np.float32)
    context_direction = np.zeros((CONTEXT_POINTS, 2), dtype=np.float32)
    context_class = np.zeros(CONTEXT_POINTS, dtype=np.int64)
    context_mask = np.zeros(CONTEXT_POINTS, dtype=bool)
    pieces = []
    for class_id, points in context:
        xy, direction = resample(points)
        distance = np.abs(xy - TILE / 2.0).max(axis=1)
        pieces.append((distance.min(), class_id, xy, direction))
    pieces.sort(key=lambda item: item[0])
    count = 0
    for _distance, class_id, xy, direction in pieces:
        take = min(len(xy), CONTEXT_POINTS - count)
        if take <= 0:
            break
        context_xy[count : count + take] = xy[:take] / TILE
        context_direction[count : count + take] = direction[:take]
        context_class[count : count + take] = class_id
        context_mask[count : count + take] = True
        count += take

    port_xy = np.zeros((PORTS, 2), dtype=np.float32)
    port_heading = np.zeros((PORTS, 2), dtype=np.float32)
    port_class = np.zeros(PORTS, dtype=np.int64)
    port_mask = np.zeros(PORTS, dtype=bool)
    for index, (x, y, hx, hy, class_id) in enumerate(ports[:PORTS]):
        port_xy[index] = (x / TILE, y / TILE)
        port_heading[index] = (hx, hy)
        port_class[index] = class_id
        port_mask[index] = True
    return {
        "plan": plan,
        "plan_xy": plan_xy.astype(np.float32),
        "context_xy": context_xy,
        "context_direction": context_direction,
        "context_class": context_class,
        "context_mask": context_mask,
        "port_xy": port_xy,
        "port_heading": port_heading,
        "port_class": port_class,
        "port_mask": port_mask,
        "city": np.asarray(city, dtype=np.float32),
    }


def collate(items):
    result = {}
    for name in items[0]:
        if name == "tokens":
            continue
        result[name] = torch.from_numpy(np.stack([item[name] for item in items]))
    if "tokens" in items[0]:
        length = max(len(item["tokens"]) for item in items)
        tokens = np.zeros((len(items), length), dtype=np.int64)
        for index, item in enumerate(items):
            tokens[index, : len(item["tokens"])] = item["tokens"]
        result["tokens"] = torch.from_numpy(tokens)
    return result


class StreetDataset(torch.utils.data.Dataset):
    def __init__(self, tiles, plans, cities, scale, augment=True, max_len=1536, city_dropout=0.1):
        # tiles: list of (city index, tile record), plans: list of normalised plan arrays
        self.tiles = tiles
        self.plans = plans
        self.cities = cities
        self.scale = scale
        self.augment = augment
        self.max_len = max_len
        self.city_dropout = city_dropout
        self.tokens = StreetTokens()
        # older data has no touch lists, work them out once here
        for _city_index, tile in tiles:
            if "touches" not in tile:
                tile["touches"] = touch_lists(tile["targets"])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        city_index, tile = self.tiles[index]
        matrix = random.choice(D4) if self.augment else D4[0]
        visible = random.choice(PATTERNS) if self.augment else PATTERNS[3]
        targets = [(class_id, move_points(points, matrix)) for class_id, points in tile["targets"]]
        context = []
        for class_id, points, _ring in tile["context"]:
            moved = move_points(points, matrix)
            if region(moved) in visible:
                context.append((class_id, moved))
        crop = move_grid(plan_crop(self.plans[city_index], *tile["cell"]), matrix) * self.scale[:, None, None]
        ports = find_ports(targets, visible)[:PORTS]
        city = np.zeros(self.cities, dtype=np.float32)
        if not self.augment or random.random() > self.city_dropout:
            city[city_index] = 1.0
        item = condition_arrays(crop, context, ports, city)
        item["tokens"] = np.asarray(self.tokens.encode(order_lines(targets, visible, ports, tile["touches"]), self.max_len), dtype=np.int64)
        return item


class StreetModel(nn.Module):
    def __init__(self, cities, d=256, layers=6, heads=8, max_len=1536, bands=8, dropout=0.0):
        super().__init__()
        self.config = {"cities": cities, "d": d, "layers": layers, "heads": heads, "max_len": max_len, "bands": bands, "dropout": dropout}
        self.tokens = StreetTokens()
        self.bands = bands
        self.place = nn.Linear(4 * bands, d)
        self.plan = nn.Linear(len(CHANNELS), d)
        self.line_class = nn.Embedding(len(LINE_CLASSES), d)
        self.direction = nn.Linear(2, d)
        self.kind = nn.Embedding(4, d)
        self.city = nn.Linear(cities, d)
        self.decoder = TokenDecoder(self.tokens.vocab, self.tokens.features(), d, layers, heads, max_len, dropout)

    def condition(self, batch):
        kinds = self.kind.weight
        plan = self.plan(batch["plan"]) + self.place(fourier(batch["plan_xy"], self.bands)) + kinds[0]
        context = (
            self.place(fourier(batch["context_xy"], self.bands))
            + self.line_class(batch["context_class"])
            + self.direction(batch["context_direction"])
            + kinds[1]
        )
        ports = self.place(fourier(batch["port_xy"], self.bands)) + self.line_class(batch["port_class"]) + self.direction(batch["port_heading"]) + kinds[2]
        city = (self.city(batch["city"]) + kinds[3])[:, None]
        memory = torch.cat([city, plan, context, ports], dim=1)
        batch_size = memory.shape[0]
        fixed = torch.zeros(batch_size, 1 + plan.shape[1], dtype=torch.bool, device=memory.device)
        padding = torch.cat([fixed, ~batch["context_mask"], ~batch["port_mask"]], dim=1)
        return memory, padding

    def forward(self, batch):
        memory, padding = self.condition(batch)
        return self.decoder(batch["tokens"], memory, padding)

    @torch.no_grad()
    def generate(self, batch, temperature=1.0, top_p=0.95, generator=None, max_points=128, max_lines=400, force_ports=True):
        memory, padding = self.condition(batch)
        ports = None
        if force_ports:
            spots = torch.round(batch["port_xy"] * TILE).clamp(0, TILE).long()
            ports = torch.cat([batch["port_class"][..., None], spots], dim=-1).masked_fill(~batch["port_mask"][..., None], -1)
        grammar = StreetGrammar(self.tokens, max_points, max_lines, ports, max_len=self.decoder.max_len, sides=batch.get("sides"))
        tokens = self.decoder.generate(memory, padding, grammar, temperature=temperature, top_p=top_p, generator=generator)
        return [self.tokens.decode(row.tolist()) for row in tokens.cpu()]


def snap_to_ports(lines, ports, reach=8.0):
    # lines are (class, points, joins), an end moved onto a port is attached so it is not a join any more
    if not ports:
        return lines
    targets = np.asarray([(x, y) for x, y, *_rest in ports], dtype=np.float32)
    result = []
    for class_id, points, joins in lines:
        points = np.asarray(points, dtype=np.float32).copy()
        joins = list(joins)
        for slot, end in ((0, 0), (1, -1)):
            distance = np.linalg.norm(targets - points[end], axis=1)
            nearest = int(np.argmin(distance))
            if distance[nearest] <= reach:
                points[end] = targets[nearest]
                joins[slot] = False
        result.append((class_id, points, tuple(joins)))
    return result
