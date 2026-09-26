from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from shapely.geometry import LineString


ROAD_CLASSES = ("major", "secondary", "local")
RAIL_CLASSES = ("rail", "subway", "light_rail", "tram", "monorail")
TRANSPORT_CLASSES = (*ROAD_CLASSES, *RAIL_CLASSES)
VERTICAL_MODES = ("surface", "underground", "elevated", "unknown")
TRANSPORT_MODES = ("road", "rail")


@dataclass(frozen=True)
class SpatialTensorConfig:
    target_size_m: float = 1024.0
    context_size_m: float = 5120.0
    local_vector_size_m: float = 2560.0
    context_line_points: int = 6
    edge_shape_points: int = 8
    max_context_lines: int = 512
    max_ports: int = 128
    max_nodes: int = 256
    max_edges: int = 512
    width_scale_m: float = 32.0


def _class_index(mode: str, value: str) -> int:
    if mode == "road":
        value = value if value in ROAD_CLASSES else "local"
    else:
        value = value if value in RAIL_CLASSES else "rail"
    return TRANSPORT_CLASSES.index(value)


def _vertical_index(value: str) -> int:
    return VERTICAL_MODES.index(value) if value in VERTICAL_MODES else 3


def _mode_index(value: str) -> int:
    return 0 if value == "road" else 1


def _resample_line(points: list[list[float]], count: int) -> np.ndarray:
    line = LineString([(float(point[0]), float(point[1])) for point in points])
    if line.length <= 1e-8:
        value = np.asarray(line.coords[0], dtype=np.float32)
        return np.repeat(value[None], count, axis=0)
    distances = np.linspace(0.0, float(line.length), count)
    return np.asarray(
        [
            [line.interpolate(float(distance)).x, line.interpolate(float(distance)).y]
            for distance in distances
        ],
        dtype=np.float32,
    )


def _normalise_target_xy(points: np.ndarray, size: float) -> np.ndarray:
    return points / size * 2.0 - 1.0


def _normalise_context_xy(points: np.ndarray, target_size: float, context_size: float) -> np.ndarray:
    center = target_size / 2.0
    return (points - center) / (context_size / 2.0)


def _edge_geometry_target(
    geometry: list[list[float]],
    start: np.ndarray,
    end: np.ndarray,
    count: int,
    target_size: float,
) -> np.ndarray:
    points = _normalise_target_xy(_resample_line(geometry, count + 2), target_size)
    straight = np.linspace(start, end, count + 2, dtype=np.float32)
    return (points - straight)[1:-1]


def _line_distance(record: dict[str, Any], target_size: float) -> float:
    points = np.asarray(record["geometry_local_m"], dtype=np.float32)
    center = np.asarray([target_size / 2.0, target_size / 2.0], dtype=np.float32)
    return float(np.linalg.norm(points.mean(axis=0) - center))


def _prepare_context_lines(
    payload: dict[str, Any],
    config: SpatialTensorConfig,
) -> dict[str, torch.Tensor]:
    records = [
        *payload["input"]["visible_transport"].get("roads", []),
        *payload["input"]["visible_transport"].get("rail", []),
    ]
    records.sort(key=lambda value: _line_distance(value, config.target_size_m))
    records = records[: config.max_context_lines]

    points = np.zeros(
        (config.max_context_lines, config.context_line_points, 2),
        dtype=np.float32,
    )
    mode = np.zeros(config.max_context_lines, dtype=np.int64)
    class_index = np.zeros(config.max_context_lines, dtype=np.int64)
    vertical = np.zeros(config.max_context_lines, dtype=np.int64)
    width = np.zeros((config.max_context_lines, 1), dtype=np.float32)
    length = np.zeros((config.max_context_lines, 1), dtype=np.float32)
    padding = np.ones(config.max_context_lines, dtype=bool)

    for index, record in enumerate(records):
        values = _resample_line(record["geometry_local_m"], config.context_line_points)
        points[index] = _normalise_context_xy(
            values,
            config.target_size_m,
            config.local_vector_size_m,
        )
        current_mode = str(record["mode"])
        mode[index] = _mode_index(current_mode)
        class_index[index] = _class_index(current_mode, str(record["class"]))
        vertical[index] = _vertical_index(str(record["vertical_mode"]))
        width[index, 0] = float(record.get("width_m", 0.0)) / config.width_scale_m
        length[index, 0] = math.log1p(float(record["length_m"])) / math.log1p(
            config.local_vector_size_m
        )
        padding[index] = False

    return {
        "context_line_points": torch.from_numpy(points),
        "context_line_mode": torch.from_numpy(mode),
        "context_line_class": torch.from_numpy(class_index),
        "context_line_vertical": torch.from_numpy(vertical),
        "context_line_width": torch.from_numpy(width),
        "context_line_length": torch.from_numpy(length),
        "context_line_padding": torch.from_numpy(padding),
    }


def _prepare_ports(
    payload: dict[str, Any],
    config: SpatialTensorConfig,
) -> dict[str, torch.Tensor]:
    records = payload["input"].get("boundary_ports", [])[: config.max_ports]
    continuous = np.zeros((config.max_ports, 5), dtype=np.float32)
    mode = np.zeros(config.max_ports, dtype=np.int64)
    class_index = np.zeros(config.max_ports, dtype=np.int64)
    vertical = np.zeros(config.max_ports, dtype=np.int64)
    padding = np.ones(config.max_ports, dtype=bool)

    for index, record in enumerate(records):
        x, y = [float(value) for value in record["position_local_m"]]
        hx, hy = [float(value) for value in record["heading"]]
        width = float(record.get("width_m", 0.0))
        continuous[index] = [
            x / config.target_size_m * 2.0 - 1.0,
            y / config.target_size_m * 2.0 - 1.0,
            hx,
            hy,
            width / config.width_scale_m,
        ]
        current_mode = str(record["mode"])
        mode[index] = _mode_index(current_mode)
        class_index[index] = _class_index(current_mode, str(record["class"]))
        vertical[index] = _vertical_index(str(record["vertical_mode"]))
        padding[index] = False

    return {
        "ports": torch.from_numpy(continuous),
        "port_mode": torch.from_numpy(mode),
        "port_class": torch.from_numpy(class_index),
        "port_vertical": torch.from_numpy(vertical),
        "port_padding": torch.from_numpy(padding),
    }


def _prepare_target_graph(
    payload: dict[str, Any],
    config: SpatialTensorConfig,
) -> dict[str, torch.Tensor]:
    graph = payload["target"]["transport_graph"]
    nodes = list(graph["nodes"])
    nodes.sort(
        key=lambda value: (
            round(float(value["position_local_m"][1]), 3),
            round(float(value["position_local_m"][0]), 3),
            str(value["transport_mode"]),
            str(value["vertical_mode"]),
            str(value["id"]),
        )
    )
    if len(nodes) > config.max_nodes:
        raise ValueError(f"nodes:{len(nodes)}")

    node_lookup = {str(node["id"]): index for index, node in enumerate(nodes)}
    node_xy = np.zeros((config.max_nodes, 2), dtype=np.float32)
    node_mode = np.zeros(config.max_nodes, dtype=np.int64)
    node_vertical = np.zeros(config.max_nodes, dtype=np.int64)
    node_boundary = np.zeros(config.max_nodes, dtype=np.float32)

    for index, node in enumerate(nodes):
        xy = np.asarray(node["position_local_m"][:2], dtype=np.float32)
        node_xy[index] = _normalise_target_xy(xy[None], config.target_size_m)[0]
        node_mode[index] = _mode_index(str(node["transport_mode"]))
        node_vertical[index] = _vertical_index(str(node["vertical_mode"]))
        node_boundary[index] = float(node.get("boundary_port_key") is not None)

    edges = []
    for edge in graph["edges"]:
        left = node_lookup.get(str(edge["from_node"]))
        right = node_lookup.get(str(edge["to_node"]))
        if left is None or right is None:
            continue
        if left > right:
            left, right = right, left
            edge = dict(edge)
            edge["geometry_local_m"] = list(reversed(edge["geometry_local_m"]))
        edges.append((left, right, edge))

    edges.sort(
        key=lambda value: (
            value[0],
            value[1],
            str(value[2]["transport_mode"]),
            str(value[2]["class"]),
            str(value[2]["vertical_mode"]),
            str(value[2]["id"]),
        )
    )
    if len(edges) > config.max_edges:
        raise ValueError(f"edges:{len(edges)}")

    edge_from = np.zeros(config.max_edges, dtype=np.int64)
    edge_to = np.zeros(config.max_edges, dtype=np.int64)
    edge_mode = np.zeros(config.max_edges, dtype=np.int64)
    edge_class = np.zeros(config.max_edges, dtype=np.int64)
    edge_vertical = np.zeros(config.max_edges, dtype=np.int64)
    edge_width = np.zeros((config.max_edges, 1), dtype=np.float32)
    edge_shape = np.zeros(
        (config.max_edges, config.edge_shape_points, 2),
        dtype=np.float32,
    )

    for index, (left, right, edge) in enumerate(edges):
        edge_from[index] = left
        edge_to[index] = right
        current_mode = str(edge["transport_mode"])
        edge_mode[index] = _mode_index(current_mode)
        edge_class[index] = _class_index(current_mode, str(edge["class"]))
        edge_vertical[index] = _vertical_index(str(edge["vertical_mode"]))
        edge_width[index, 0] = float(edge.get("width_m", 0.0)) / config.width_scale_m
        edge_shape[index] = _edge_geometry_target(
            edge["geometry_local_m"],
            node_xy[left],
            node_xy[right],
            config.edge_shape_points,
            config.target_size_m,
        )

    return {
        "node_count": torch.tensor(len(nodes), dtype=torch.long),
        "node_xy": torch.from_numpy(node_xy),
        "node_mode": torch.from_numpy(node_mode),
        "node_vertical": torch.from_numpy(node_vertical),
        "node_boundary": torch.from_numpy(node_boundary),
        "edge_count": torch.tensor(len(edges), dtype=torch.long),
        "edge_from": torch.from_numpy(edge_from),
        "edge_to": torch.from_numpy(edge_to),
        "edge_mode": torch.from_numpy(edge_mode),
        "edge_class": torch.from_numpy(edge_class),
        "edge_vertical": torch.from_numpy(edge_vertical),
        "edge_width": torch.from_numpy(edge_width),
        "edge_shape": torch.from_numpy(edge_shape),
    }


def geographic_split(row: dict[str, Any], group_size: int = 5) -> str:
    sample_id = str(row["id"])
    parts = sample_id.rsplit("_", 2)
    grid_row = int(parts[-2])
    grid_column = int(parts[-1])
    parent = f"{grid_row // group_size}:{grid_column // group_size}"
    value = int.from_bytes(
        hashlib.sha1(parent.encode("utf-8"), usedforsecurity=False).digest()[:4],
        "little",
    ) % 100
    if value < 75:
        return "train"
    if value < 88:
        return "validation"
    return "test"


class SpatialWorldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        config: SpatialTensorConfig | None = None,
        maximum_samples: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.config = config or SpatialTensorConfig()
        rows = [
            json.loads(line)
            for line in (self.root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if maximum_samples is not None:
            rows.sort(
                key=lambda value: hashlib.sha1(
                    str(value["id"]).encode("utf-8"),
                    usedforsecurity=False,
                ).digest()
            )
            rows = rows[:maximum_samples]

        payloads = []
        feature_values = []
        for row in rows:
            with gzip.open(self.root / row["sample_path"], "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            payloads.append((row, payload))
            for cell in payload["input"]["context_cells"]:
                if cell["features"]:
                    feature_values.append(cell["features"])

        if not payloads:
            raise RuntimeError("No spatial world samples found")
        self.feature_names = sorted(payloads[0][1]["style"])
        feature_array = np.asarray(
            [
                [float(values.get(name, 0.0)) for name in self.feature_names]
                for values in feature_values
            ],
            dtype=np.float32,
        )
        self.feature_mean = feature_array.mean(axis=0)
        self.feature_std = feature_array.std(axis=0)
        self.feature_std[self.feature_std < 1e-6] = 1.0

        self.samples = []
        self.rejected = {"nodes": 0, "edges": 0}
        for row, payload in payloads:
            try:
                graph = _prepare_target_graph(payload, self.config)
            except ValueError as error:
                name = str(error).split(":", 1)[0]
                self.rejected[name] += 1
                continue

            cells = payload["input"]["context_cells"]
            context = np.zeros(
                (len(cells), len(self.feature_names) + 3),
                dtype=np.float32,
            )
            for index, cell in enumerate(cells):
                raw = np.asarray(
                    [float(cell["features"].get(name, 0.0)) for name in self.feature_names],
                    dtype=np.float32,
                )
                center = np.asarray(cell["center_local_m"], dtype=np.float32)
                center = _normalise_context_xy(
                    center[None],
                    self.config.target_size_m,
                    self.config.context_size_m,
                )[0]
                context[index, : len(self.feature_names)] = (
                    raw - self.feature_mean
                ) / self.feature_std
                context[index, -3:-1] = center
                context[index, -1] = float(cell["masked_fraction"])

            style = np.asarray(
                [float(payload["style"].get(name, 0.0)) for name in self.feature_names],
                dtype=np.float32,
            )
            style = (style - self.feature_mean) / self.feature_std

            controls = np.asarray(
                [float(payload["controls"].get(name, 0.0)) for name in self.feature_names],
                dtype=np.float32,
            )
            controls = (controls - self.feature_mean) / self.feature_std

            prepared = {
                "context_cells": torch.from_numpy(context),
                "style": torch.from_numpy(style),
                "controls": torch.from_numpy(controls),
                **_prepare_context_lines(payload, self.config),
                **_prepare_ports(payload, self.config),
                **graph,
                "sample_id": row["id"],
                "split": geographic_split(row),
            }
            self.samples.append(prepared)

        if not self.samples:
            raise RuntimeError("All spatial world samples exceeded configured graph limits")

    @property
    def context_dimensions(self) -> int:
        return len(self.feature_names) + 3

    @property
    def style_dimensions(self) -> int:
        return len(self.feature_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
