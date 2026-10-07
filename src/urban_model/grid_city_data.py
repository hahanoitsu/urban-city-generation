from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import torch

from urban_model.structured_city_data import AREA_KINDS, BUILDING_KINDS, CLASSES, VERTICAL


@dataclass(frozen=True)
class GridConfig:
    target_size_m: float = 512.0
    node_grid: int = 16
    node_slots: int = 10
    edge_slots: int = 4
    building_grid: int = 16
    building_slots: int = 8
    area_grid: int = 8
    area_slots: int = 8
    segment_m: float = 28.0
    curve_points: int = 2
    building_points: int = 24
    area_points: int = 48
    width_scale_m: float = 32.0
    height_scale_m: float = 100.0

    @property
    def candidates(self) -> int:
        return 9 * self.node_slots

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]

# the eight rotations and flips of the square tile
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


def candidate_table(config: GridConfig) -> np.ndarray:
    grid = config.node_grid
    slots = config.node_slots
    table = np.full((grid * grid * slots, 9 * slots), -1, dtype=np.int64)
    for row in range(grid):
        for column in range(grid):
            for slot in range(slots):
                node = (row * grid + column) * slots + slot
                for neighbour, (dy, dx) in enumerate(OFFSETS):
                    other_row = row + dy
                    other_column = column + dx
                    if not (0 <= other_row < grid and 0 <= other_column < grid):
                        continue
                    base = (other_row * grid + other_column) * slots
                    table[node, neighbour * slots : (neighbour + 1) * slots] = np.arange(base, base + slots)
    return table


def _cell(point, grid, size):
    width = size / grid
    column = min(max(int(math.floor(point[0] / width)), 0), grid - 1)
    row = min(max(int(math.floor(point[1] / width)), 0), grid - 1)
    return row, column


def _cell_centre(row, column, grid, size):
    width = size / grid
    return np.array([(column + 0.5) * width, (row + 0.5) * width])


def _arc(points):
    points = np.asarray(points, dtype=np.float64)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return points, np.concatenate([[0.0], np.cumsum(steps)])


def _sample(points, cumulative, distances):
    return np.stack(
        [np.interp(distances, cumulative, points[:, 0]), np.interp(distances, cumulative, points[:, 1])],
        axis=1,
    )


def resample_ring(points, count):
    points = np.asarray(points, dtype=np.float64)
    closed = np.concatenate([points, points[:1]])
    closed, cumulative = _arc(closed)
    if cumulative[-1] <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    distances = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    return _sample(closed, cumulative, distances)


def canonical_ring(points):
    points = np.asarray(points, dtype=np.float64)
    area = 0.5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1])
    if area < 0:
        points = points[::-1]
    keys = [(round(float(x), 3), round(float(y), 3)) for x, y in points]
    start = min(range(len(points)), key=lambda index: keys[index])
    return np.roll(points, -start, axis=0)


def _chord(start, end, minimum):
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= minimum:
        along = np.array([1.0, 0.0]) if length <= 1e-9 else delta / length
        length = minimum
    else:
        along = delta / length
    across = np.array([-along[1], along[0]])
    return length, along, across


def split_edges(plain, segment_m):
    nodes = [np.asarray(point, dtype=np.float64) for point in plain["nodes"]]
    edges = []
    for edge in plain["edges"]:
        if edge["from"] == edge["to"]:
            continue
        points = np.asarray(edge["points"], dtype=np.float64).copy()
        points[0] = nodes[edge["from"]]
        points[-1] = nodes[edge["to"]]
        points, cumulative = _arc(points)
        length = cumulative[-1]
        count = max(1, int(math.ceil(length / segment_m)))
        cuts = [length * index / count for index in range(count + 1)]
        ids = [edge["from"]]
        for distance in cuts[1:-1]:
            nodes.append(_sample(points, cumulative, np.array([distance]))[0])
            ids.append(len(nodes) - 1)
        ids.append(edge["to"])
        for index in range(count):
            distances = np.linspace(cuts[index], cuts[index + 1], 8)
            edges.append({**edge, "from": ids[index], "to": ids[index + 1], "points": _sample(points, cumulative, distances)})
    return {**plain, "nodes": nodes, "edges": edges}


def encode_grid(plain, config: GridConfig):
    size = config.target_size_m
    plain = split_edges(plain, config.segment_m)
    grid = config.node_grid
    slots = config.node_slots
    half = size / grid / 2.0
    total = grid * grid * slots

    cells = {}
    for index, point in enumerate(plain["nodes"]):
        cells.setdefault(_cell(point, grid, size), []).append(index)
    node_slot = {}
    node_pos = np.zeros((total, 2), dtype=np.float32)
    node_presence = np.zeros(total, dtype=np.int64)
    for (row, column), members in cells.items():
        if len(members) > slots:
            raise ValueError("grid nodes exceeded")
        members.sort(key=lambda index: (plain["nodes"][index][1], plain["nodes"][index][0]))
        centre = _cell_centre(row, column, grid, size)
        for slot, index in enumerate(members):
            global_slot = (row * grid + column) * slots + slot
            node_slot[index] = (global_slot, row, column, slot)
            node_pos[global_slot] = (plain["nodes"][index] - centre) / half
            node_presence[global_slot] = 1

    edge_count = config.edge_slots
    curve = config.curve_points
    edge_target = np.zeros((total, edge_count), dtype=np.int64)
    edge_mode = np.zeros((total, edge_count), dtype=np.int64)
    edge_class = np.zeros((total, edge_count), dtype=np.int64)
    edge_vertical = np.zeros((total, edge_count), dtype=np.int64)
    edge_width = np.zeros((total, edge_count, 1), dtype=np.float32)
    edge_width_valid = np.zeros((total, edge_count), dtype=bool)
    edge_width_weight = np.zeros((total, edge_count), dtype=np.float32)
    edge_curve = np.zeros((total, edge_count, curve, 2), dtype=np.float32)
    used = np.zeros(total, dtype=np.int64)
    for edge in plain["edges"]:
        first = node_slot[edge["from"]]
        second = node_slot[edge["to"]]
        points = np.asarray(edge["points"], dtype=np.float64)
        if first[0] == second[0]:
            continue
        if first[0] > second[0]:
            first, second = second, first
            points = points[::-1]
        low, row, column, _slot = first
        high, other_row, other_column, other_slot = second
        dy = other_row - row
        dx = other_column - column
        if abs(dy) > 1 or abs(dx) > 1:
            raise ValueError("grid edge too long")
        sub = used[low]
        if sub >= edge_count:
            raise ValueError("grid edges exceeded")
        used[low] += 1
        neighbour = OFFSETS.index((dy, dx))
        edge_target[low, sub] = neighbour * slots + other_slot + 1
        edge_mode[low, sub] = int(edge["mode"])
        edge_class[low, sub] = int(edge["class"])
        edge_vertical[low, sub] = int(edge["vertical"])
        if edge.get("width") is not None:
            edge_width[low, sub, 0] = float(edge["width"]) / config.width_scale_m
            edge_width_valid[low, sub] = True
            edge_width_weight[low, sub] = float(edge.get("width_weight", 1.0))
        start = np.asarray(plain["nodes"][edge["from"] if node_slot[edge["from"]][0] == low else edge["to"]])
        end = np.asarray(plain["nodes"][edge["to"] if node_slot[edge["to"]][0] == high else edge["from"]])
        points, cumulative = _arc(points)
        inner = _sample(points, cumulative, np.linspace(0.0, cumulative[-1], curve + 2))[1:-1]
        straight = start + (end - start) * (np.arange(1, curve + 1)[:, None] / (curve + 1))
        length, along, across = _chord(start, end, 5.0)
        offset = inner - straight
        edge_curve[low, sub, :, 0] = offset @ along / length
        edge_curve[low, sub, :, 1] = offset @ across / length

    result = {
        "node_pos": node_pos,
        "node_presence": node_presence,
        "edge_target": edge_target,
        "edge_mode": edge_mode,
        "edge_class": edge_class,
        "edge_vertical": edge_vertical,
        "edge_width": edge_width,
        "edge_width_valid": edge_width_valid,
        "edge_width_weight": edge_width_weight,
        "edge_curve": edge_curve,
    }
    dropped = Counter()
    for name, grid, slots, count, items in (
        ("building", config.building_grid, config.building_slots, config.building_points, plain["buildings"]),
        ("area", config.area_grid, config.area_slots, config.area_points, plain["areas"]),
    ):
        half = size / grid / 2.0
        total = grid * grid * slots
        frame = np.zeros((total, 3), dtype=np.float32)
        shape = np.zeros((total, count, 2), dtype=np.float32)
        presence = np.zeros(total, dtype=np.int64)
        kind = np.zeros(total, dtype=np.int64)
        height = np.zeros((total, 1), dtype=np.float32)
        height_valid = np.zeros(total, dtype=bool)
        height_weight = np.zeros(total, dtype=np.float32)
        cells = {}
        for item in items:
            ring = canonical_ring(resample_ring(item["points"], count))
            centre = ring.mean(axis=0)
            offset = ring - centre
            radius = max(float(np.sqrt((offset**2).sum(axis=1).mean())), 0.5)
            cells.setdefault(_cell(centre, grid, size), []).append((item, centre, offset / radius, radius))
        for (row, column), members in cells.items():
            if len(members) > slots:
                members.sort(key=lambda member: -member[3])
                dropped[name] += len(members) - slots
                members = members[:slots]
            members.sort(key=lambda member: (member[1][1], member[1][0]))
            cell_centre = _cell_centre(row, column, grid, size)
            for slot, (item, centre, local, radius) in enumerate(members):
                index = (row * grid + column) * slots + slot
                presence[index] = 1
                frame[index, :2] = (centre - cell_centre) / half
                frame[index, 2] = math.log(radius / 10.0)
                shape[index] = local
                kind[index] = int(item["kind"])
                if name == "building" and item.get("height") is not None:
                    height[index, 0] = float(item["height"]) / config.height_scale_m
                    height_valid[index] = True
                    height_weight[index] = float(item.get("height_weight", 1.0))
        result[f"{name}_frame"] = frame
        result[f"{name}_shape"] = shape
        result[f"{name}_presence"] = presence
        result[f"{name}_kind"] = kind
        if name == "building":
            result["building_height"] = height
            result["building_height_valid"] = height_valid
            result["building_height_weight"] = height_weight
    return result, dropped


def decode_grid(arrays, config: GridConfig):
    size = config.target_size_m
    grid = config.node_grid
    slots = config.node_slots
    half = size / grid / 2.0
    table = candidate_table(config)
    presence = np.asarray(arrays["node_presence"])
    pos = np.asarray(arrays["node_pos"], dtype=np.float64)
    nodes = []
    lookup = {}
    for index in np.flatnonzero(presence == 1):
        cell = index // slots
        centre = _cell_centre(cell // grid, cell % grid, grid, size)
        lookup[int(index)] = len(nodes)
        nodes.append(centre + pos[index] * half)

    edges = []
    pairs = set()
    target = np.asarray(arrays["edge_target"])
    curve = np.asarray(arrays["edge_curve"], dtype=np.float64)
    count = curve.shape[2]
    for low in lookup:
        for sub in range(target.shape[1]):
            value = int(target[low, sub])
            if value <= 0 or value - 1 >= table.shape[1]:
                continue
            high = int(table[low, value - 1])
            if high < 0 or high not in lookup or high == low:
                continue
            pair = (min(low, high), max(low, high))
            if pair in pairs:
                continue
            pairs.add(pair)
            start = nodes[lookup[low]]
            end = nodes[lookup[high]]
            length, along, across = _chord(start, end, 5.0)
            straight = start + (end - start) * (np.arange(1, count + 1)[:, None] / (count + 1))
            local = curve[low, sub]
            inner = straight + length * (local[:, :1] * along + local[:, 1:2] * across)
            width = float(arrays["edge_width"][low, sub, 0]) * config.width_scale_m
            edges.append(
                {
                    "from": lookup[low],
                    "to": lookup[high],
                    "points": np.concatenate([start[None], inner, end[None]]),
                    "mode": int(arrays["edge_mode"][low, sub]),
                    "class": int(arrays["edge_class"][low, sub]),
                    "vertical": int(arrays["edge_vertical"][low, sub]),
                    "width": width if width > 0 else None,
                }
            )

    result = {"nodes": nodes, "edges": edges}
    for name, grid, slots in (
        ("building", config.building_grid, config.building_slots),
        ("area", config.area_grid, config.area_slots),
    ):
        half = size / grid / 2.0
        frame = np.asarray(arrays[f"{name}_frame"], dtype=np.float64)
        shape = np.asarray(arrays[f"{name}_shape"], dtype=np.float64)
        items = []
        for index in np.flatnonzero(np.asarray(arrays[f"{name}_presence"]) == 1):
            cell = index // slots
            cell_centre = _cell_centre(cell // grid, cell % grid, grid, size)
            centre = cell_centre + frame[index, :2] * half
            radius = math.exp(float(frame[index, 2])) * 10.0
            item = {"points": centre + shape[index] * radius, "kind": int(arrays[f"{name}_kind"][index])}
            if name == "building":
                item["height"] = float(arrays["building_height"][index, 0]) * config.height_scale_m
            items.append(item)
        result[f"{name}s"] = items
    return result


def merge_chains(plain):
    nodes = plain["nodes"]
    edges = plain["edges"]
    touching = {}
    for index, edge in enumerate(edges):
        touching.setdefault(edge["from"], []).append(index)
        touching.setdefault(edge["to"], []).append(index)

    def key(edge):
        return (edge["mode"], edge["class"], edge["vertical"])

    def joinable(node):
        found = touching.get(node, [])
        return len(found) == 2 and found[0] != found[1] and key(edges[found[0]]) == key(edges[found[1]])

    seen = set()
    merged = []
    for start in range(len(edges)):
        if start in seen:
            continue
        chain = [start]
        seen.add(start)
        for direction in (0, 1):
            current = start
            node = edges[start]["to"] if direction == 0 else edges[start]["from"]
            while joinable(node):
                other = [index for index in touching[node] if index != current][0]
                if other in seen:
                    break
                seen.add(other)
                if direction == 0:
                    chain.append(other)
                else:
                    chain.insert(0, other)
                current = other
                edge = edges[other]
                node = edge["to"] if edge["from"] == node else edge["from"]
        points = []
        node = None
        if len(chain) == 1:
            edge = edges[chain[0]]
            merged.append(dict(edge))
            continue
        first, second = edges[chain[0]], edges[chain[1]]
        node = first["from"] if first["from"] not in (second["from"], second["to"]) else first["to"]
        start_node = node
        widths = []
        for index in chain:
            edge = edges[index]
            part = np.asarray(edge["points"])
            if edge["from"] != node:
                part = part[::-1]
                node = edge["from"]
            else:
                node = edge["to"]
            points.extend(part if not points else part[1:])
            if edge.get("width") is not None:
                widths.append(edge["width"])
        merged.append(
            {
                **edges[chain[0]],
                "from": start_node,
                "to": node,
                "points": np.asarray(points),
                "width": float(np.mean(widths)) if widths else None,
            }
        )

    used = sorted({edge["from"] for edge in merged} | {edge["to"] for edge in merged})
    remap = {old: new for new, old in enumerate(used)}
    keep = [nodes[index] for index in used]
    for index in range(len(nodes)):
        if index not in touching:
            remap[index] = len(keep)
            keep.append(nodes[index])
    for edge in merged:
        edge["from"] = remap[edge["from"]]
        edge["to"] = remap[edge["to"]]
    return {**plain, "nodes": keep, "edges": merged}


def plain_from_json(payload):
    nodes = [np.asarray(node["xyz_m"][:2], dtype=np.float64) for node in payload["nodes"]]
    ids = {node["id"]: index for index, node in enumerate(payload["nodes"])}
    edges = []
    for edge in payload["edges"]:
        mode = 0 if edge["mode"] == "road" else 1
        width = edge.get("width_m")
        edges.append(
            {
                "from": ids[edge["from"]],
                "to": ids[edge["to"]],
                "points": np.asarray([point[:2] for point in edge["spline_xyz_m"]], dtype=np.float64),
                "mode": mode,
                "class": CLASSES.index(edge["class"]) if edge["class"] in CLASSES else 0,
                "vertical": VERTICAL.index(edge["vertical_mode"]) if edge["vertical_mode"] in VERTICAL else 3,
                "width": float(width) if width and mode == 0 and float(width) > 0 else None,
            }
        )
    buildings = [
        {
            "points": np.asarray(item["footprint_xy_m"], dtype=np.float64),
            "height": float(item["height_m"]) if item.get("height_m") and float(item["height_m"]) > 0 else None,
            "kind": BUILDING_KINDS.index(item["type"]) if item.get("type") in BUILDING_KINDS else 0,
        }
        for item in payload["buildings"]
    ]
    areas = [
        {
            "points": np.asarray(item["polygon_xy_m"], dtype=np.float64),
            "kind": AREA_KINDS.index(item["kind"]),
        }
        for item in payload["areas"]
    ]
    return {"nodes": nodes, "edges": edges, "buildings": buildings, "areas": areas}


def json_from_plain(plain):
    return {
        "nodes": [{"id": index, "xyz_m": [float(point[0]), float(point[1]), None]} for index, point in enumerate(plain["nodes"])],
        "edges": [
            {
                "from": edge["from"],
                "to": edge["to"],
                "mode": "road" if edge["mode"] == 0 else "rail",
                "class": CLASSES[edge["class"]] if 0 <= edge["class"] < len(CLASSES) else "unknown",
                "vertical_mode": VERTICAL[edge["vertical"]] if 0 <= edge["vertical"] < len(VERTICAL) else "unknown",
                "width_m": float(edge["width"]) if edge.get("width") else 0.0,
                "spline_xyz_m": [[float(x), float(y), None] for x, y in edge["points"]],
            }
            for edge in plain["edges"]
        ],
        "buildings": [
            {
                "footprint_xy_m": [[float(x), float(y)] for x, y in item["points"]],
                "base_z_m": None,
                "height_m": float(item.get("height") or 0.0),
                "type": BUILDING_KINDS[item["kind"]] if 0 <= item["kind"] < len(BUILDING_KINDS) else "generic",
            }
            for item in plain["buildings"]
        ],
        "areas": [
            {"kind": AREA_KINDS[item["kind"]], "polygon_xy_m": [[float(x), float(y)] for x, y in item["points"]]}
            for item in plain["areas"]
            if 0 <= item["kind"] < len(AREA_KINDS)
        ],
    }


def plain_from_structured(item, scene_config):
    size = scene_config.target_size_m

    def metres(value):
        return (np.asarray(value, dtype=np.float64) + 1.0) * 0.5 * size

    presence = item["node_presence"].numpy()
    positions = item["node_position"].numpy()[:, :2]
    lookup = {}
    nodes = []
    for index in np.flatnonzero(presence == 1):
        lookup[int(index)] = len(nodes)
        nodes.append(metres(positions[index]))
    edges = []
    count = item["edge_shape"].shape[1]
    for index in np.flatnonzero(item["edge_presence"].numpy() == 1):
        left = int(item["edge_from"][index])
        right = int(item["edge_to"][index])
        if left not in lookup or right not in lookup or left == right:
            continue
        start = positions[left]
        end = positions[right]
        straight = start + (end - start) * np.linspace(0.0, 1.0, count)[:, None]
        line = straight + item["edge_shape"][index, :, :2].numpy()
        valid = bool(item["edge_width_valid"][index])
        edges.append(
            {
                "from": lookup[left],
                "to": lookup[right],
                "points": metres(line),
                "mode": int(item["edge_mode"][index]),
                "class": int(item["edge_class"][index]),
                "vertical": int(item["edge_vertical"][index]),
                "width": float(item["edge_width"][index, 0]) * scene_config.width_scale_m if valid else None,
                "width_weight": float(item["edge_width_weight"][index]),
            }
        )
    buildings = []
    for index in np.flatnonzero(item["building_presence"].numpy() == 1):
        valid = bool(item["building_height_valid"][index])
        buildings.append(
            {
                "points": metres(item["building_shape"][index].numpy()),
                "height": float(item["building_height"][index, 0]) * scene_config.height_scale_m if valid else None,
                "height_weight": float(item["building_height_weight"][index]),
                "kind": int(item["building_kind"][index]),
            }
        )
    areas = [
        {"points": metres(item["area_shape"][index].numpy()), "kind": int(item["area_kind"][index])}
        for index in np.flatnonzero(item["area_presence"].numpy() == 1)
    ]
    return {"nodes": nodes, "edges": edges, "buildings": buildings, "areas": areas}


def transform_plain(plain, matrix, size):
    centre = np.array([size / 2.0, size / 2.0])

    def move(points):
        return (np.asarray(points, dtype=np.float64) - centre) @ matrix.T + centre

    return {
        "nodes": [move(point) for point in plain["nodes"]],
        "edges": [{**edge, "points": move(edge["points"])} for edge in plain["edges"]],
        "buildings": [{**item, "points": move(item["points"])} for item in plain["buildings"]],
        "areas": [{**item, "points": move(item["points"])} for item in plain["areas"]],
    }


def transform_inputs(ports, context, matrix, feature_count):
    matrix_t = torch.tensor(matrix.T, dtype=ports.dtype)
    ports = ports.clone()
    context = context.clone()
    ports[:, 0:2] = ports[:, 0:2] @ matrix_t
    ports[:, 2:4] = ports[:, 2:4] @ matrix_t
    normals = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=ports.dtype)
    sides = ports[:, -4:].clone()
    moved = normals @ matrix_t
    for old in range(4):
        new = int(torch.argmax(normals @ moved[old]))
        ports[:, -4 + new] = sides[:, old]
    context[:, feature_count : feature_count + 2] = context[:, feature_count : feature_count + 2] @ matrix_t
    return ports, context


def to_tensors(arrays):
    return {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in arrays.items()}


class GridCityDataset(torch.utils.data.Dataset):
    def __init__(self, base, indexes, config: GridConfig, augment=False, show_progress=True):
        self.base = base
        self.config = config
        self.augment = augment
        self.feature_count = len(base.feature_names)
        self.items = []
        self.rejected = Counter()
        self.dropped = Counter()
        for position, index in enumerate(indexes, start=1):
            plain = plain_from_structured(base[index], base.config)
            try:
                _arrays, dropped = encode_grid(plain, config)
            except ValueError as error:
                self.rejected[str(error)] += 1
                continue
            self.dropped.update(dropped)
            self.items.append((index, plain))
            if show_progress and (position % 500 == 0 or position == len(indexes)):
                print(f"grid encode: {position}/{len(indexes)} kept={len(self.items)}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        base_index, plain = self.items[index]
        item = self.base[base_index]
        ports = item["ports"]
        context = item["context"]
        arrays = None
        if self.augment:
            choice = random.randrange(len(D4))
            if choice:
                matrix = D4[choice]
                try:
                    arrays, _dropped = encode_grid(transform_plain(plain, matrix, self.config.target_size_m), self.config)
                    ports, context = transform_inputs(ports, context, matrix, self.feature_count)
                except ValueError:
                    arrays = None
                    ports = item["ports"]
                    context = item["context"]
        if arrays is None:
            arrays, _dropped = encode_grid(plain, self.config)
        return {
            **to_tensors(arrays),
            "context": context,
            "context_padding": item["context_padding"],
            "relations": item["relations"],
            "ports": ports,
            "port_padding": item["port_padding"],
            "sample_id": item["sample_id"],
        }
