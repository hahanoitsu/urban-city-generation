from __future__ import annotations

import math
import random

import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.ops import nearest_points, unary_union

from urban_layers.data import CHANNELS, box_polygon, from_frame
from urban_layers.decoder import TokenDecoder, fourier
from urban_layers.tokens import BuildingGrammar, BuildingTokens

RING = 48


def block_arrays(block, scale, city):
    ring = np.asarray(block["ring"], dtype=np.float32)
    tangent = np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    features = np.concatenate(
        [
            np.asarray(block["plan"], dtype=np.float32) * scale,
            np.asarray(block["frontage"], dtype=np.float32),
            [math.log1p(block["area"]) / 10.0],
        ]
    ).astype(np.float32)
    return {
        "ring": ring / 256.0,
        "tangent": tangent.astype(np.float32),
        "features": features,
        "city": np.asarray(city, dtype=np.float32),
    }


def flip(block, flip_x, flip_y):
    ring = np.asarray(block["ring"], dtype=np.float32).copy()
    items = np.asarray(block["buildings"], dtype=np.float32).copy()
    if flip_x:
        ring[:, 0] *= -1
        items[:, 1] *= -1
        items[:, 5] *= -1
    if flip_y:
        ring[:, 1] *= -1
        items[:, 2] *= -1
        items[:, 5] *= -1
    return {**block, "ring": ring, "buildings": items}


class BuildingDataset(torch.utils.data.Dataset):
    def __init__(self, blocks, cities, scale, augment=True, max_len=1024, city_dropout=0.1):
        self.blocks = blocks
        self.cities = cities
        self.scale = scale
        self.augment = augment
        self.max_len = max_len
        self.city_dropout = city_dropout
        self.tokens = BuildingTokens()

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        city_index, block = self.blocks[index]
        if self.augment:
            block = flip(block, random.random() < 0.5, random.random() < 0.5)
        city = np.zeros(self.cities, dtype=np.float32)
        if not self.augment or random.random() > self.city_dropout:
            city[city_index] = 1.0
        item = block_arrays(block, self.scale, city)
        item["tokens"] = np.asarray(self.tokens.encode(block["landuse"], block["buildings"], self.max_len), dtype=np.int64)
        return item


class BuildingModel(torch.nn.Module):
    def __init__(self, cities, d=256, layers=6, heads=8, max_len=1024, bands=8, dropout=0.0):
        super().__init__()
        self.config = {"cities": cities, "d": d, "layers": layers, "heads": heads, "max_len": max_len, "bands": bands, "dropout": dropout}
        self.tokens = BuildingTokens()
        self.bands = bands
        self.place = torch.nn.Linear(4 * bands, d)
        self.tangent = torch.nn.Linear(2, d)
        self.features = torch.nn.Sequential(torch.nn.Linear(len(CHANNELS) + 4, d), torch.nn.GELU(), torch.nn.Linear(d, d))
        self.kind = torch.nn.Embedding(3, d)
        self.city = torch.nn.Linear(cities, d)
        self.decoder = TokenDecoder(self.tokens.vocab, self.tokens.features(), d, layers, heads, max_len, dropout)

    def condition(self, batch):
        kinds = self.kind.weight
        ring = self.place(fourier(batch["ring"], self.bands)) + self.tangent(batch["tangent"]) + kinds[0]
        features = (self.features(batch["features"]) + kinds[1])[:, None]
        city = (self.city(batch["city"]) + kinds[2])[:, None]
        memory = torch.cat([city, features, ring], dim=1)
        padding = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
        return memory, padding

    def forward(self, batch):
        memory, padding = self.condition(batch)
        return self.decoder(batch["tokens"], memory, padding)

    @torch.no_grad()
    def generate(self, batch, temperature=1.0, top_p=0.95, generator=None, max_buildings=1000, landuse=None):
        memory, padding = self.condition(batch)
        grammar = BuildingGrammar(self.tokens, max_buildings, landuse)
        tokens = self.decoder.generate(memory, padding, grammar, temperature=temperature, top_p=top_p, generator=generator)
        return [self.tokens.decode(row.tolist()) for row in tokens.cpu()]


def face_road(corners, roads, reach=60.0, limit=45.0):
    if roads is None or not len(roads):
        return corners
    middle = corners.mean(axis=0)
    start, step = roads[:, 0], roads[:, 1] - roads[:, 0]
    along = np.clip(((middle - start) * step).sum(axis=1) / np.maximum((step * step).sum(axis=1), 1e-9), 0.0, 1.0)
    distance = np.linalg.norm(start + step * along[:, None] - middle, axis=1)
    nearest = int(np.argmin(distance))
    if distance[nearest] > reach:
        return corners
    edge = corners[1] - corners[0]
    turn = (math.atan2(step[nearest, 1], step[nearest, 0]) - math.atan2(edge[1], edge[0])) % (math.pi / 2)
    if turn > math.pi / 4:
        turn -= math.pi / 2
    if abs(math.degrees(turn)) > limit:
        return corners
    cos, sin = math.cos(turn), math.sin(turn)
    return (corners - middle) @ np.array([[cos, sin], [-sin, cos]]) + middle


def near_boxes(shape, boxes):
    minx, miny, maxx, maxy = shape.bounds
    return np.nonzero((boxes[:, 0] < maxx) & (boxes[:, 2] > minx) & (boxes[:, 1] < maxy) & (boxes[:, 3] > miny))[0] if len(boxes) else []


def clashes(shape, blocked, shapes, boxes, tolerance):
    if blocked is not None and blocked.intersects(shape):
        return True
    return any(shape.intersection(shapes[index]).area > tolerance for index in near_boxes(shape, boxes))


def way_out(corners, block_polygon, blocked, shapes, boxes, inside, limit, tolerance):
    shape = Polygon(corners)
    if not clashes(shape, blocked, shapes, boxes, tolerance):
        return corners
    hit = [shape.intersection(blocked)] if blocked is not None and blocked.intersects(shape) else []
    hit += [shape.intersection(shapes[index]) for index in near_boxes(shape, boxes)]
    along, across = corners[1] - corners[0], corners[3] - corners[0]
    limit = max(limit, 0.4 * max(np.linalg.norm(along), np.linalg.norm(across)))
    steps = np.linspace(limit / 12.0, limit, 12)
    for direction in (np.subtract(shape.centroid.coords[0], unary_union(hit).centroid.coords[0]), along, -along, across, -across):
        if np.linalg.norm(direction) < 1e-6:
            continue
        direction = direction / np.linalg.norm(direction)
        for step in steps:
            moved = corners + direction * step
            shape = Polygon(moved)
            if clashes(shape, blocked, shapes, boxes, tolerance):
                continue
            if shape.intersection(block_polygon).area >= inside * shape.area:
                return moved
            break
    return None


def trim(corners, block_polygon, blocked, shapes, boxes, inside, tolerance, most=0.75, least=4.0, steps=7):
    best, loss = None, most
    for side in range(4):
        inward = corners[(side + 3) % 4] - corners[side]
        moving = np.zeros((4, 1))
        moving[[side, (side + 1) % 4]] = 1.0
        cut = lambda share: corners + moving * inward * share
        deepest = min(loss, 1.0 - least / max(np.linalg.norm(inward), 1e-9))
        if deepest <= 0.0 or clashes(Polygon(cut(deepest)), blocked, shapes, boxes, tolerance):
            continue
        low, high = 0.0, deepest
        for _step in range(steps):
            middle = (low + high) / 2.0
            low, high = (middle, high) if clashes(Polygon(cut(middle)), blocked, shapes, boxes, tolerance) else (low, middle)
        shape = Polygon(cut(high))
        if shape.intersection(block_polygon).area >= inside * shape.area:
            best, loss = cut(high), high
    return best


def place_buildings(block_polygon, centre, angle, items, inside=0.7, nudge=15.0, blocked=(), roads=None, push=12.0, tolerance=1.0):
    placed = []
    shapes = []
    boxes = np.zeros((0, 4))
    for kind, x, y, width, depth, turn, height in items:
        corners = from_frame(box_polygon(x, y, width, depth, turn), centre, angle)
        shape = Polygon(corners)
        if not shape.is_valid or shape.area < 4.0:
            continue
        if shape.intersection(block_polygon).area < inside * shape.area:
            room = block_polygon.buffer(-min(width, depth) / 2.0)
            if room.is_empty:
                continue
            middle = shape.centroid
            target = nearest_points(room, middle)[0]
            if target.distance(middle) > nudge:
                continue
            corners = corners + (target.x - middle.x, target.y - middle.y)
            shape = Polygon(corners)
            if shape.intersection(block_polygon).area < inside * shape.area:
                continue
        turned = face_road(corners, roads)
        if turned is not corners and Polygon(turned).intersection(block_polygon).area >= inside * Polygon(turned).area:
            corners = turned
        moved = None
        middle = corners.mean(axis=0)
        offset = corners - middle
        sides = [corners[1] - corners[0], corners[3] - corners[0]]
        sides = [side / max(np.linalg.norm(side), 1e-9) for side in sides]
        lengths = [np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[3] - corners[0])]
        for ground in blocked or (None,):
            moved = way_out(corners, block_polygon, ground, shapes, boxes, inside, push, tolerance)
            if moved is not None:
                break
            moved = trim(corners, block_polygon, ground, shapes, boxes, inside, tolerance)
            kept = Polygon(moved).area / Polygon(corners).area if moved is not None else 0.0
            for first, second in ((1.0, 0.8), (0.8, 1.0), (0.8, 0.8), (1.0, 0.6), (0.6, 1.0)):
                if first * second <= kept:
                    break
                if min(lengths[0] * first, lengths[1] * second) < min(4.0, *lengths):
                    continue
                scaled = middle + np.outer(offset @ sides[0], sides[0]) * first + np.outer(offset @ sides[1], sides[1]) * second
                shrunk = way_out(scaled, block_polygon, ground, shapes, boxes, inside, push, tolerance)
                if shrunk is not None:
                    moved = shrunk
                    break
            if moved is not None:
                break
        if moved is None:
            continue
        shape = Polygon(moved)
        shapes.append(shape)
        boxes = np.vstack([boxes, shape.bounds])
        placed.append({"kind": int(kind), "height_m": float(height), "footprint": moved.tolist()})
    return placed
