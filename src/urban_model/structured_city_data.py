from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Polygon,
    shape,
)
from shapely.geometry.polygon import orient

from urban_dataset.city_state import build_transport_graph
from urban_dataset.tile import TileSpec


ROAD_CLASSES = ("major", "secondary", "local")
RAIL_CLASSES = ("rail", "subway", "light_rail", "tram")
CLASSES = (*ROAD_CLASSES, *RAIL_CLASSES)
VERTICAL = ("surface", "underground", "elevated", "unknown")
AREA_KINDS = (
    "green",
    "water",
    "residential",
    "commercial_mixed",
    "industrial",
    "civic",
)
RELATIONS = (
    "spatial",
    "road_major",
    "road_secondary",
    "road_local",
    "rail",
    "surface",
    "underground",
    "elevated",
)


@dataclass(frozen=True)
class SceneTensorConfig:
    target_size_m: float = 512.0
    region_size_m: float = 2048.0
    node_slots: int = 384
    edge_slots: int = 640
    building_slots: int = 384
    area_slots: int = 96
    edge_shape_points: int = 8
    building_points: int = 24
    area_points: int = 32
    maximum_ports: int = 96
    width_scale_m: float = 32.0
    height_scale_m: float = 100.0
    context_radius_regions: int = 1


def _one_hot(value: str, values: tuple[str, ...]) -> list[float]:
    result = [0.0] * len(values)
    if value in values:
        result[values.index(value)] = 1.0
    return result


def _frame(records: list[dict[str, Any]], mode: str) -> gpd.GeoDataFrame:
    rows = []
    for record in records:
        row = {
            "id": record["id"],
            "vertical_mode": record["vertical_mode"],
            "geometry": LineString(record["geometry_local_m"]),
        }
        if mode == "road":
            row["road_class"] = record["class"]
            width = record.get("width_m")
            if width is not None:
                row["estimated_width_m"] = float(width)
        else:
            row["railway"] = record["class"]
        rows.append(row)
    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")


def _transport_graph(payload: dict[str, Any], size: float) -> dict[str, Any]:
    roads = _frame(payload["target"].get("roads", []), "road")
    rail = _frame(payload["target"].get("rail", []), "rail")
    tile = TileSpec(
        city_id=str(payload["city_id"]),
        column=0,
        row=0,
        minx=0.0,
        miny=0.0,
        maxx=size,
        maxy=size,
    )
    return build_transport_graph(roads, rail, tile)


def _iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon | GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_polygons(part)


def _resample_line(points: list[list[float]], count: int) -> np.ndarray:
    line = LineString([(float(point[0]), float(point[1])) for point in points])
    if line.length <= 1e-8:
        value = np.asarray(line.coords[0], dtype=np.float32)
        return np.repeat(value[None], count, axis=0)
    distances = np.linspace(0.0, float(line.length), count)
    return np.asarray(
        [[line.interpolate(float(distance)).x, line.interpolate(float(distance)).y] for distance in distances],
        dtype=np.float32,
    )


def _resample_ring(polygon: Polygon, count: int) -> np.ndarray:
    oriented = orient(polygon, sign=1.0)
    coordinates = list(oriented.exterior.coords[:-1])
    if not coordinates:
        coordinates = list(oriented.exterior.coords)
    start = min(
        range(len(coordinates)),
        key=lambda index: (
            round(float(coordinates[index][0]), 6),
            round(float(coordinates[index][1]), 6),
        ),
    )
    coordinates = coordinates[start:] + coordinates[:start]
    coordinates.append(coordinates[0])
    ring = LineString(coordinates)
    if ring.length <= 1e-8:
        value = np.asarray(ring.coords[0], dtype=np.float32)
        return np.repeat(value[None], count, axis=0)
    distances = np.linspace(0.0, float(ring.length), count, endpoint=False)
    return np.asarray(
        [[ring.interpolate(float(distance)).x, ring.interpolate(float(distance)).y] for distance in distances],
        dtype=np.float32,
    )


def _normalise_xy(points: np.ndarray, size: float) -> np.ndarray:
    return points / size * 2.0 - 1.0


def _port_vector(port: dict[str, Any], config: SceneTensorConfig) -> list[float]:
    x, y = [float(value) for value in port["position_local_m"]]
    hx, hy = [float(value) for value in port["heading"]]
    width = port.get("width_m")
    width_valid = width is not None and float(width) > 0
    x_norm = x / config.target_size_m * 2.0 - 1.0
    y_norm = y / config.target_size_m * 2.0 - 1.0
    epsilon = 1e-3
    side = "left"
    if abs(x - config.target_size_m) <= epsilon:
        side = "right"
    elif abs(y) <= epsilon:
        side = "bottom"
    elif abs(y - config.target_size_m) <= epsilon:
        side = "top"
    return [
        x_norm,
        y_norm,
        hx,
        hy,
        float(width) / config.width_scale_m if width_valid else 0.0,
        float(width_valid),
        *_one_hot(str(port["mode"]), ("road", "rail")),
        *_one_hot(str(port["class"]), CLASSES),
        *_one_hot(str(port["vertical_mode"]), VERTICAL),
        *_one_hot(side, ("left", "right", "bottom", "top")),
    ]


def _edge_class_index(mode: str, value: str) -> int:
    if mode == "road":
        value = value if value in ROAD_CLASSES else "local"
    else:
        value = value if value in RAIL_CLASSES else "rail"
    return CLASSES.index(value)


def _vertical_index(value: str) -> int:
    return VERTICAL.index(value) if value in VERTICAL else VERTICAL.index("unknown")


def _polygon_records(payload: dict[str, Any]) -> list[tuple[str, Polygon]]:
    result = []
    for record in payload["target"].get("landuse", []):
        kind = str(record.get("class") or "")
        if kind not in AREA_KINDS or kind == "water":
            continue
        geometry_payload = record.get("geometry_local_m")
        if not geometry_payload:
            continue
        geometry = shape(geometry_payload)
        for polygon in _iter_polygons(geometry):
            if polygon.area > 1e-6:
                result.append((kind, polygon))

    for record in payload["target"].get("water", []):
        geometry_payload = record.get("geometry_local_m")
        if not geometry_payload:
            continue
        geometry = shape(geometry_payload)
        for polygon in _iter_polygons(geometry):
            if polygon.area > 1e-6:
                result.append(("water", polygon))
    return result


def scene_counts(payload: dict[str, Any], config: SceneTensorConfig) -> dict[str, int]:
    graph = _transport_graph(payload, config.target_size_m)
    buildings = 0
    for record in payload["target"].get("buildings", []):
        geometry_payload = record.get("footprint_local_m")
        if not geometry_payload:
            continue
        buildings += sum(1 for polygon in _iter_polygons(shape(geometry_payload)) if polygon.area > 1e-6)
    return {
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "buildings": buildings,
        "areas": len(_polygon_records(payload)),
        "ports": len(payload["input"].get("boundary_ports", [])),
    }


def encode_scene(payload: dict[str, Any], config: SceneTensorConfig) -> dict[str, torch.Tensor]:
    graph = _transport_graph(payload, config.target_size_m)
    nodes = list(graph["nodes"])
    nodes.sort(
        key=lambda node: (
            round(float(node["position_local_m"][1]), 3),
            round(float(node["position_local_m"][0]), 3),
            str(node["transport_mode"]),
            str(node["vertical_mode"]),
            str(node["id"]),
        )
    )
    if len(nodes) > config.node_slots:
        raise ValueError(f"node slots exceeded: {len(nodes)} > {config.node_slots}")

    node_lookup = {str(node["id"]): index for index, node in enumerate(nodes)}
    node_position = np.zeros((config.node_slots, 3), dtype=np.float32)
    node_presence = np.zeros(config.node_slots, dtype=np.int64)
    node_z_valid = np.zeros(config.node_slots, dtype=bool)
    for index, node in enumerate(nodes):
        x, y = [float(value) for value in node["position_local_m"][:2]]
        node_position[index, :2] = [x / config.target_size_m * 2.0 - 1.0, y / config.target_size_m * 2.0 - 1.0]
        node_presence[index] = 1

    edges = []
    for edge in graph["edges"]:
        left = node_lookup.get(str(edge["from_node"]))
        right = node_lookup.get(str(edge["to_node"]))
        if left is None or right is None or left == right:
            continue
        if left > right:
            left, right = right, left
            edge = dict(edge)
            edge["geometry_local_m"] = list(reversed(edge["geometry_local_m"]))
        edges.append((left, right, edge))
    edges.sort(
        key=lambda item: (
            min(item[0], item[1]),
            max(item[0], item[1]),
            str(item[2]["transport_mode"]),
            str(item[2]["class"]),
            str(item[2]["id"]),
        )
    )
    if len(edges) > config.edge_slots:
        raise ValueError(f"edge slots exceeded: {len(edges)} > {config.edge_slots}")

    edge_presence = np.zeros(config.edge_slots, dtype=np.int64)
    edge_mode = np.zeros(config.edge_slots, dtype=np.int64)
    edge_class = np.zeros(config.edge_slots, dtype=np.int64)
    edge_vertical = np.zeros(config.edge_slots, dtype=np.int64)
    edge_from = np.zeros(config.edge_slots, dtype=np.int64)
    edge_to = np.zeros(config.edge_slots, dtype=np.int64)
    edge_width = np.zeros((config.edge_slots, 1), dtype=np.float32)
    edge_width_valid = np.zeros(config.edge_slots, dtype=bool)
    edge_width_weight = np.zeros(config.edge_slots, dtype=np.float32)
    edge_shape = np.zeros((config.edge_slots, config.edge_shape_points, 3), dtype=np.float32)
    edge_z_valid = np.zeros((config.edge_slots, config.edge_shape_points), dtype=bool)

    for index, (left, right, edge) in enumerate(edges):
        edge_presence[index] = 1
        mode = str(edge["transport_mode"])
        edge_mode[index] = 0 if mode == "road" else 1
        edge_class[index] = _edge_class_index(mode, str(edge["class"]))
        edge_vertical[index] = _vertical_index(str(edge["vertical_mode"]))
        edge_from[index] = left
        edge_to[index] = right
        width_source = str(edge.get("width_source", ""))
        width_valid = mode == "road" and width_source == "estimated_width_m"
        edge_width[index, 0] = float(edge["width_m"]) / config.width_scale_m if width_valid else 0.0
        edge_width_valid[index] = width_valid
        if width_valid:
            source_id = str(edge.get("source_id"))
            confidence = 0.25
            for record in payload["target"].get("roads", []):
                if str(record.get("id")) == source_id:
                    confidence = float(record.get("width_confidence", 0.25))
                    break
            edge_width_weight[index] = confidence
        points = _resample_line(edge["geometry_local_m"], config.edge_shape_points)
        points = _normalise_xy(points, config.target_size_m)
        start = node_position[left, :2]
        end = node_position[right, :2]
        straight = np.linspace(start, end, config.edge_shape_points, dtype=np.float32)
        edge_shape[index, :, :2] = points - straight

    building_records = []
    for record in payload["target"].get("buildings", []):
        geometry_payload = record.get("footprint_local_m")
        if not geometry_payload:
            continue
        for polygon in _iter_polygons(shape(geometry_payload)):
            if polygon.area <= 1e-6:
                continue
            building_records.append((polygon.centroid.y, polygon.centroid.x, str(record["id"]), polygon, record))
    building_records.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(building_records) > config.building_slots:
        raise ValueError(
            f"building slots exceeded: {len(building_records)} > {config.building_slots}"
        )

    building_presence = np.zeros(config.building_slots, dtype=np.int64)
    building_shape = np.zeros(
        (config.building_slots, config.building_points, 2),
        dtype=np.float32,
    )
    building_height = np.zeros((config.building_slots, 1), dtype=np.float32)
    building_height_valid = np.zeros(config.building_slots, dtype=bool)
    building_height_weight = np.zeros(config.building_slots, dtype=np.float32)
    building_base_z = np.zeros((config.building_slots, 1), dtype=np.float32)
    building_base_z_valid = np.zeros(config.building_slots, dtype=bool)
    for index, (_y, _x, _id, polygon, record) in enumerate(building_records):
        building_presence[index] = 1
        building_shape[index] = _normalise_xy(
            _resample_ring(polygon, config.building_points),
            config.target_size_m,
        )
        height = record.get("height_m")
        valid = bool(record.get("height_valid", height is not None and float(height or 0) > 0))
        if valid:
            building_height[index, 0] = float(height) / config.height_scale_m
            building_height_valid[index] = True
            confidence = int(record.get("height_confidence", 0))
            building_height_weight[index] = {
                3: 1.0,
                2: 0.8,
                1: 0.35,
                0: 0.1,
            }.get(confidence, 0.1)

    areas = _polygon_records(payload)
    areas.sort(key=lambda item: (item[1].centroid.y, item[1].centroid.x, item[0]))
    if len(areas) > config.area_slots:
        raise ValueError(f"area slots exceeded: {len(areas)} > {config.area_slots}")

    area_presence = np.zeros(config.area_slots, dtype=np.int64)
    area_kind = np.zeros(config.area_slots, dtype=np.int64)
    area_shape = np.zeros((config.area_slots, config.area_points, 2), dtype=np.float32)
    for index, (kind, polygon) in enumerate(areas):
        area_presence[index] = 1
        area_kind[index] = AREA_KINDS.index(kind)
        area_shape[index] = _normalise_xy(
            _resample_ring(polygon, config.area_points),
            config.target_size_m,
        )

    return {
        "node_position": torch.from_numpy(node_position),
        "node_presence": torch.from_numpy(node_presence),
        "node_z_valid": torch.from_numpy(node_z_valid),
        "edge_presence": torch.from_numpy(edge_presence),
        "edge_mode": torch.from_numpy(edge_mode),
        "edge_class": torch.from_numpy(edge_class),
        "edge_vertical": torch.from_numpy(edge_vertical),
        "edge_from": torch.from_numpy(edge_from),
        "edge_to": torch.from_numpy(edge_to),
        "edge_width": torch.from_numpy(edge_width),
        "edge_width_valid": torch.from_numpy(edge_width_valid),
        "edge_width_weight": torch.from_numpy(edge_width_weight),
        "edge_shape": torch.from_numpy(edge_shape),
        "edge_z_valid": torch.from_numpy(edge_z_valid),
        "building_presence": torch.from_numpy(building_presence),
        "building_shape": torch.from_numpy(building_shape),
        "building_height": torch.from_numpy(building_height),
        "building_height_valid": torch.from_numpy(building_height_valid),
        "building_height_weight": torch.from_numpy(building_height_weight),
        "building_base_z": torch.from_numpy(building_base_z),
        "building_base_z_valid": torch.from_numpy(building_base_z_valid),
        "area_presence": torch.from_numpy(area_presence),
        "area_kind": torch.from_numpy(area_kind),
        "area_shape": torch.from_numpy(area_shape),
    }


class StructuredCityDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        config: SceneTensorConfig | None = None,
        maximum_samples: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.config = config or SceneTensorConfig()
        graph = json.loads((self.root / "context-graph.json").read_text(encoding="utf-8"))
        self.feature_names = sorted(graph["nodes"][0]["features"])
        self.node_ids = [node["id"] for node in graph["nodes"]]
        self.node_index = {node_id: index for index, node_id in enumerate(self.node_ids)}
        self.node_rows = np.asarray([int(node["row"]) for node in graph["nodes"]], dtype=np.int64)
        self.node_columns = np.asarray([int(node["column"]) for node in graph["nodes"]], dtype=np.int64)
        features = np.asarray(
            [
                [float(node["features"][name]) for name in self.feature_names]
                for node in graph["nodes"]
            ],
            dtype=np.float32,
        )
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-6] = 1.0
        self.feature_mean = mean
        self.feature_std = std
        self.features = (features - mean) / std
        self.centers = np.asarray(
            [node["center_projected_m"] for node in graph["nodes"]],
            dtype=np.float32,
        )

        relations = np.zeros(
            (len(RELATIONS), len(self.node_ids), len(self.node_ids)),
            dtype=np.float32,
        )
        np.fill_diagonal(relations[RELATIONS.index("spatial")], 1.0)
        for edge in graph["edges"]:
            left = self.node_index[edge["from"]]
            right = self.node_index[edge["to"]]
            relations[RELATIONS.index("spatial"), left, right] = 1.0
            relations[RELATIONS.index("spatial"), right, left] = 1.0
            for port in edge["transport_ports"]:
                names = []
                mode = str(port["mode"])
                edge_class = str(port["class"])
                vertical = str(port["vertical_mode"])
                if mode == "road" and edge_class in ROAD_CLASSES:
                    names.append(f"road_{edge_class}")
                if mode == "rail":
                    names.append("rail")
                if vertical in {"surface", "underground", "elevated"}:
                    names.append(vertical)
                for name in names:
                    relation = RELATIONS.index(name)
                    relations[relation, left, right] += 1.0
                    relations[relation, right, left] += 1.0
        for index in range(len(RELATIONS)):
            degree = relations[index].sum(axis=1, keepdims=True)
            relations[index] /= np.maximum(degree, 1.0)
        self.global_relations = relations

        rows = [
            json.loads(line)
            for line in (self.root / "targets.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        accepted = []
        for row in rows:
            if row["boundary_ports"] > self.config.maximum_ports:
                continue
            path = self.root / row["sample_path"]
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            counts = scene_counts(payload, self.config)
            if (
                counts["nodes"] > self.config.node_slots
                or counts["edges"] > self.config.edge_slots
                or counts["buildings"] > self.config.building_slots
                or counts["areas"] > self.config.area_slots
            ):
                continue
            accepted.append((row, payload))
            if maximum_samples is not None and len(accepted) >= maximum_samples:
                break
        if not accepted:
            raise RuntimeError("No structured city samples fit the configured slots")
        self.samples = accepted
        self.port_dimensions = 23
        self.context_dimensions = len(self.feature_names) + 3
        self.context_slots = (self.config.context_radius_regions * 2 + 1) ** 2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, payload = self.samples[index]
        target_bounds = np.asarray(payload["target_bounds_projected_m"], dtype=np.float32)
        target_center = np.asarray(
            [
                (target_bounds[0] + target_bounds[2]) / 2.0,
                (target_bounds[1] + target_bounds[3]) / 2.0,
            ],
            dtype=np.float32,
        )
        parent = self.node_index[payload["parent_region_id"]]
        parent_row = self.node_rows[parent]
        parent_column = self.node_columns[parent]
        radius = self.config.context_radius_regions
        indexes = [
            global_index
            for global_index in range(len(self.node_ids))
            if abs(int(self.node_rows[global_index] - parent_row)) <= radius
            and abs(int(self.node_columns[global_index] - parent_column)) <= radius
        ]
        indexes.sort(
            key=lambda global_index: (
                int(self.node_rows[global_index] - parent_row),
                int(self.node_columns[global_index] - parent_column),
            )
        )

        context = np.zeros((self.context_slots, self.context_dimensions), dtype=np.float32)
        context_padding = np.ones(self.context_slots, dtype=bool)
        relations = np.zeros(
            (len(RELATIONS), self.context_slots, self.context_slots),
            dtype=np.float32,
        )

        for local_index, global_index in enumerate(indexes):
            context_padding[local_index] = False
            relative = (
                self.centers[global_index] - target_center
            ) / max(self.config.region_size_m, 1.0)
            context[local_index, : len(self.feature_names)] = self.features[global_index]
            context[local_index, len(self.feature_names) : len(self.feature_names) + 2] = relative
            if global_index == parent:
                context[local_index, : len(self.feature_names)] = 0.0
                context[local_index, -1] = 1.0

        if indexes:
            global_indexes = np.asarray(indexes, dtype=np.int64)
            local_relations = self.global_relations[:, global_indexes][:, :, global_indexes]
            count = len(indexes)
            relations[:, :count, :count] = local_relations

        ports = np.zeros((self.config.maximum_ports, self.port_dimensions), dtype=np.float32)
        port_padding = np.ones(self.config.maximum_ports, dtype=bool)
        vectors = [
            _port_vector(port, self.config)
            for port in payload["input"].get("boundary_ports", [])[: self.config.maximum_ports]
        ]
        if vectors:
            ports[: len(vectors)] = np.asarray(vectors, dtype=np.float32)
            port_padding[: len(vectors)] = False

        scene = encode_scene(payload, self.config)
        return {
            **scene,
            "context": torch.from_numpy(context),
            "context_padding": torch.from_numpy(context_padding),
            "relations": torch.from_numpy(relations),
            "ports": torch.from_numpy(ports),
            "port_padding": torch.from_numpy(port_padding),
            "sample_id": row["id"],
        }
