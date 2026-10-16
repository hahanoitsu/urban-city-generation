from __future__ import annotations

import math
import random

import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.ops import nearest_points

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
        # blocks: list of (city index, block record)
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


def place_buildings(block_polygon, centre, angle, items, inside=0.7, overlap=0.15, nudge=15.0):
    placed = []
    shapes = []
    for kind, x, y, width, depth, turn, height in items:
        corners = from_frame(box_polygon(x, y, width, depth, turn), centre, angle)
        shape = Polygon(corners)
        if not shape.is_valid or shape.area < 4.0:
            continue
        if shape.intersection(block_polygon).area < inside * shape.area:
            # the model puts some boxes a few metres off the block, real ones almost never are,
            # so move the box just inside instead of dropping it
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
        if any(shape.intersection(other).area > overlap * min(shape.area, other.area) for other in shapes if shape.distance(other) < 1.0):
            continue
        shapes.append(shape)
        placed.append({"kind": int(kind), "height_m": float(height), "footprint": corners.tolist()})
    return placed
