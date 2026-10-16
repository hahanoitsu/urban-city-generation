from __future__ import annotations

import math

import numpy as np
import torch

from urban_layers.data import LANDUSE, LINE_CLASSES, TILE
from urban_model.structured_city_data import BUILDING_KINDS

PAD, BOS, EOS = 0, 1, 2


def fourier(values, bands=8):
    values = np.asarray(values, dtype=np.float32)[..., None]
    scales = math.pi * 2.0 ** np.arange(bands, dtype=np.float32)
    return np.concatenate([np.sin(values * scales), np.cos(values * scales)], axis=-1)


class StreetTokens:
    # BOS, then per line: class, x, y, x, y, ..., END, then EOS
    def __init__(self, size=int(TILE)):
        self.size = size
        self.coords = size + 1
        self.end = 3
        self.first_class = 4
        self.first_x = self.first_class + len(LINE_CLASSES)
        self.first_y = self.first_x + self.coords
        self.vocab = self.first_y + self.coords
        kinds = np.zeros(self.vocab, dtype=np.int64)
        kinds[self.end] = 3
        kinds[self.first_class : self.first_x] = 4
        kinds[self.first_x : self.first_y] = 5
        kinds[self.first_y :] = 6
        kinds[EOS] = 2
        kinds[BOS] = 1
        self.kinds = kinds

    def features(self):
        table = np.zeros((self.vocab, 7 + 16), dtype=np.float32)
        table[np.arange(self.vocab), self.kinds] = 1.0
        values = np.arange(self.coords) / self.size
        table[self.first_x : self.first_y, 7:] = fourier(values)
        table[self.first_y :, 7:] = fourier(values)
        return torch.from_numpy(table)

    def quantise(self, points):
        points = np.clip(np.rint(np.asarray(points, dtype=np.float64)), 0, self.size).astype(np.int64)
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.any(points[1:] != points[:-1], axis=1)
        return points[keep]

    def encode(self, lines, max_len=None):
        tokens = [BOS]
        for class_id, points in lines:
            points = self.quantise(points)
            if len(points) < 2:
                continue
            piece = [self.first_class + int(class_id)]
            for x, y in points:
                piece += [self.first_x + int(x), self.first_y + int(y)]
            piece.append(self.end)
            if max_len is not None and len(tokens) + len(piece) + 1 > max_len:
                break
            tokens += piece
        tokens.append(EOS)
        return tokens

    def decode(self, tokens):
        lines = []
        class_id = None
        points = []
        pending_x = None
        for token in list(tokens):
            token = int(token)
            kind = self.kinds[token]
            if kind == 4:
                class_id, points, pending_x = token - self.first_class, [], None
            elif kind == 5:
                pending_x = token - self.first_x
            elif kind == 6 and pending_x is not None:
                points.append((pending_x, token - self.first_y))
                pending_x = None
            elif kind == 3:
                if class_id is not None and len(points) >= 2:
                    lines.append((class_id, np.asarray(points, dtype=np.float32)))
                class_id, points = None, []
            elif token == EOS:
                break
        return lines


class StreetGrammar:
    # ports: optional (batch, ports, 3) long tensor of class, x, y with -1 for padding.
    # every port gets a line that starts exactly there before any free lines.
    # with max_len set, a line is ended early if the ports still waiting would not fit.
    # sides: optional (batch, 4) bool, left bottom right top neighbour already generated.
    # like in training, a line can only start or end within margin of those sides where a line comes in
    def __init__(self, tokens, max_points=48, max_lines=160, ports=None, reach=8, max_len=None, sides=None, margin=4):
        self.tokens = tokens
        self.max_points = max_points
        self.max_lines = max_lines
        self.ports = ports
        self.reach = reach
        self.max_len = max_len
        self.sides = sides
        self.margin = margin
        kinds = torch.from_numpy(tokens.kinds)
        self.masks = {name: kinds == value for name, value in (("eos", 2), ("end", 3), ("class", 4), ("x", 5), ("y", 6))}
        self.masks["pad"] = torch.zeros(tokens.vocab, dtype=torch.bool)
        self.masks["pad"][PAD] = True

    def start(self, batch, device):
        ports = self.ports.to(device) if self.ports is not None else torch.full((batch, 1, 3), -1, dtype=torch.long, device=device)
        zeros = torch.zeros(batch, dtype=torch.long, device=device)
        return {
            "phase": zeros.clone(),
            "points": zeros.clone(),
            "lines": zeros.clone(),
            "ports": ports,
            "used": ports[..., 0] < 0,
            "current": torch.full((batch,), -1, dtype=torch.long, device=device),
            "last_x": zeros.clone(),
            "last": torch.zeros(batch, 2, dtype=torch.long, device=device),
            "length": torch.ones(batch, dtype=torch.long, device=device),
            "sides": self.sides.to(device) if self.sides is not None else torch.zeros(batch, 4, dtype=torch.bool, device=device),
        }

    def pending(self, state):
        free = ~state["used"]
        return torch.where(free.any(dim=1), free.long().argmax(dim=1), torch.full_like(state["phase"], -1))

    def allowed(self, state):
        masks = {name: value.to(state["phase"].device) for name, value in self.masks.items()}
        phase, points, lines = state["phase"], state["points"], state["lines"]
        batch = len(phase)
        result = torch.zeros(batch, self.tokens.vocab, dtype=torch.bool, device=phase.device)
        open_line = (phase == 0)[:, None]
        result |= open_line & masks["eos"]
        result |= (open_line & (lines < self.max_lines)[:, None]) & masks["class"]
        result |= (phase == 1)[:, None] & masks["x"]
        result |= (phase == 2)[:, None] & masks["y"]
        sides, size, margin = state["sides"], self.tokens.size, self.margin
        edges = torch.zeros_like(result)
        first_x, first_y = self.tokens.first_x, self.tokens.first_y
        edges[:, first_x : first_x + margin + 1] = sides[:, 0:1]
        edges[:, first_y : first_y + margin + 1] = sides[:, 1:2]
        edges[:, first_x + size - margin : first_x + size + 1] = sides[:, 2:3]
        edges[:, first_y + size - margin : first_y + size + 1] = sides[:, 3:4]
        first_point = (points == 0) & (state["current"] < 0) & ((phase == 1) | (phase == 2))
        result &= ~(edges & first_point[:, None])

        after_point = phase == 3
        room = points < self.max_points
        if self.max_len is not None:
            # a waiting port needs class, two points and end (6), this line needs a point and end, then eos
            waiting = (~state["used"]).sum(dim=1) - (state["current"] >= 0).long()
            room &= self.max_len - state["length"] > 6 * waiting + 6
        room |= points < 2
        last = state["last"]
        on_side = (
            (sides[:, 0] & (last[:, 0] <= margin))
            | (sides[:, 1] & (last[:, 1] <= margin))
            | (sides[:, 2] & (last[:, 0] >= size - margin))
            | (sides[:, 3] & (last[:, 1] >= size - margin))
        )
        spots = state["ports"]
        at_port = (((spots[..., 1:] - last[:, None, :]).abs().max(dim=-1).values <= self.reach) & (spots[..., 0] >= 0)).any(dim=1)
        stuck = on_side & ~at_port & room
        result |= (after_point & room)[:, None] & masks["x"]
        result |= (after_point & (points >= 2) & ~stuck)[:, None] & masks["end"]
        result |= (phase == 4)[:, None] & masks["pad"]

        rows = torch.arange(batch, device=phase.device)
        waiting = self.pending(state)
        port = state["ports"][rows, waiting.clamp_min(0)]
        current = state["ports"][rows, state["current"].clamp_min(0)]
        for condition, token in (
            ((phase == 0) & (waiting >= 0), self.tokens.first_class + port[:, 0]),
            ((phase == 1) & (state["current"] >= 0), self.tokens.first_x + current[:, 1]),
            ((phase == 2) & (state["current"] >= 0), self.tokens.first_y + current[:, 2]),
        ):
            forced = torch.zeros_like(result)
            forced[rows, token.clamp(0, self.tokens.vocab - 1)] = True
            result = torch.where(condition[:, None], forced, result)
        return result

    def advance(self, state, token):
        kinds = torch.from_numpy(self.tokens.kinds).to(token.device)[token]
        phase = state["phase"].clone()
        points = state["points"].clone()
        lines = state["lines"].clone()
        used = state["used"].clone()
        current = state["current"].clone()
        last_x = state["last_x"].clone()
        last = state["last"].clone()
        rows = torch.arange(len(phase), device=token.device)
        finished = phase == 4

        current = torch.where((kinds == 4) & (phase == 0), self.pending(state), current)
        last_x = torch.where(kinds == 5, token - self.tokens.first_x, last_x)
        new_point = kinds == 6
        last = torch.where(new_point[:, None], torch.stack([last_x, token - self.tokens.first_y], dim=1), last)
        started = new_point & (current >= 0)
        used[rows[started], current[started]] = True
        current = torch.where(started, torch.full_like(current, -1), current)
        ended = kinds == 3
        if bool(ended.any()):
            distance = (state["ports"][..., 1:] - last[:, None, :]).abs().max(dim=-1).values
            used |= ended[:, None] & (distance <= self.reach) & (state["ports"][..., 0] >= 0)

        phase = torch.where(kinds == 4, torch.ones_like(phase), phase)
        points = torch.where(kinds == 4, torch.zeros_like(points), points)
        phase = torch.where(kinds == 5, torch.full_like(phase, 2), phase)
        points = torch.where(new_point, points + 1, points)
        phase = torch.where(new_point, torch.full_like(phase, 3), phase)
        lines = torch.where(ended, lines + 1, lines)
        phase = torch.where(ended, torch.zeros_like(phase), phase)
        phase = torch.where((kinds == 2) | finished, torch.full_like(phase, 4), phase)
        length = state["length"] + 1
        return {**state, "phase": phase, "points": points, "lines": lines, "used": used, "current": current, "last_x": last_x, "last": last, "length": length}

    def done(self, state):
        return state["phase"] == 4


class BuildingTokens:
    # BOS, landuse, then per building: kind, x, y, width, depth, angle, height, then EOS
    fields = ("x", "y", "width", "depth", "angle", "height")
    bins = {"x": 512, "y": 512, "width": 128, "depth": 128, "angle": 60, "height": 100}

    def __init__(self):
        self.first_landuse = 3
        self.first_kind = self.first_landuse + len(LANDUSE)
        start = self.first_kind + len(BUILDING_KINDS)
        self.first = {}
        for name in self.fields:
            self.first[name] = start
            start += self.bins[name]
        self.vocab = start
        kinds = np.zeros(self.vocab, dtype=np.int64)
        kinds[BOS] = 1
        kinds[EOS] = 2
        kinds[self.first_landuse : self.first_kind] = 3
        kinds[self.first_kind : self.first["x"]] = 4
        for index, name in enumerate(self.fields):
            kinds[self.first[name] : self.first[name] + self.bins[name]] = 5 + index
        self.kinds = kinds

    def features(self):
        table = np.zeros((self.vocab, 11 + 16), dtype=np.float32)
        table[np.arange(self.vocab), self.kinds] = 1.0
        for name in self.fields:
            count = self.bins[name]
            table[self.first[name] : self.first[name] + count, 11:] = fourier(np.arange(count) / count)
        return torch.from_numpy(table)

    def values(self, building):
        _kind, x, y, width, depth, turn, height = building
        return {
            "x": int(np.clip(np.floor(x + 256), 0, 511)),
            "y": int(np.clip(np.floor(y + 256), 0, 511)),
            "width": int(np.clip(np.rint(width) - 1, 0, 127)),
            "depth": int(np.clip(np.rint(depth) - 1, 0, 127)),
            "angle": int(np.clip(np.floor((math.degrees(turn) + 90.0) / 3.0), 0, 59)),
            "height": int(np.clip(np.rint(height / 3.0), 0, 99)),
        }

    def encode(self, landuse, buildings, max_len=None):
        tokens = [BOS, self.first_landuse + int(landuse)]
        order = sorted(range(len(buildings)), key=lambda index: (round(float(buildings[index][2]) / 12.0), float(buildings[index][1])))
        for index in order:
            building = buildings[index]
            if max_len is not None and len(tokens) + 8 > max_len:
                break
            tokens.append(self.first_kind + int(building[0]))
            values = self.values(building)
            tokens += [self.first[name] + values[name] for name in self.fields]
        tokens.append(EOS)
        return tokens

    def decode(self, tokens):
        landuse = 0
        buildings = []
        current = None
        for token in list(tokens):
            token = int(token)
            kind = self.kinds[token]
            if kind == 3:
                landuse = token - self.first_landuse
            elif kind == 4:
                current = {"kind": token - self.first_kind}
            elif kind >= 5 and current is not None:
                name = self.fields[kind - 5]
                current[name] = token - self.first[name]
                if name == "height":
                    buildings.append(
                        (
                            current["kind"],
                            current["x"] - 256 + 0.5,
                            current["y"] - 256 + 0.5,
                            current["width"] + 1.0,
                            current["depth"] + 1.0,
                            math.radians(current["angle"] * 3.0 + 1.5 - 90.0),
                            current["height"] * 3.0,
                        )
                    )
                    current = None
            elif token == EOS:
                break
        return landuse, buildings


class BuildingGrammar:
    def __init__(self, tokens, max_buildings=120):
        self.tokens = tokens
        self.max_buildings = max_buildings
        self.kinds = torch.from_numpy(tokens.kinds)

    def start(self, batch, device):
        return {"phase": torch.zeros(batch, dtype=torch.long, device=device), "count": torch.zeros(batch, dtype=torch.long, device=device)}

    def allowed(self, state):
        kinds = self.kinds.to(state["phase"].device)
        phase, count = state["phase"], state["count"]
        result = torch.zeros(len(phase), self.tokens.vocab, dtype=torch.bool, device=phase.device)
        result |= (phase == 0)[:, None] & (kinds == 3)[None]
        building = (phase == 1)[:, None]
        result |= building & (kinds == 2)[None]
        result |= (building & (count < self.max_buildings)[:, None]) & (kinds == 4)[None]
        for offset in range(6):
            result |= (phase == 2 + offset)[:, None] & (kinds == 5 + offset)[None]
        pad = torch.zeros(self.tokens.vocab, dtype=torch.bool, device=phase.device)
        pad[PAD] = True
        result |= (phase == 8)[:, None] & pad[None]
        return result

    def advance(self, state, token):
        kinds = self.kinds.to(token.device)[token]
        phase = state["phase"].clone()
        count = state["count"].clone()
        finished = phase == 8
        phase = torch.where(kinds == 3, torch.ones_like(phase), phase)
        phase = torch.where(kinds == 4, torch.full_like(phase, 2), phase)
        moved = (kinds >= 5) & (kinds <= 9)
        phase = torch.where(moved, kinds - 2, phase)
        count = torch.where(kinds == 10, count + 1, count)
        phase = torch.where(kinds == 10, torch.ones_like(phase), phase)
        phase = torch.where((kinds == 2) | finished, torch.full_like(phase, 8), phase)
        return {"phase": phase, "count": count}

    def done(self, state):
        return state["phase"] == 8
