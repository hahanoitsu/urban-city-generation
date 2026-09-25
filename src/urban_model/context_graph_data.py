from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import LineString

from urban_ai.codec import CommandCodecConfig, FIELDS, encode_program
from urban_ai.conversion import city_state_to_program
from urban_ai.schema import ProgramConfig
from urban_dataset.city_state import build_transport_graph
from urban_dataset.tile import TileSpec


ROAD_CLASSES = ("major", "secondary", "local")
RAIL_CLASSES = ("rail", "subway", "light_rail", "tram", "monorail")
VERTICAL = ("surface", "underground", "elevated", "unknown")


def _one_hot(value: str, values: tuple[str, ...]) -> list[float]:
    result = [0.0] * len(values)
    if value in values:
        result[values.index(value)] = 1.0
    return result


def _port_vector(port: dict[str, Any]) -> list[float]:
    x, y = port["position_local_m"]
    hx, hy = port["heading"]
    mode = str(port["mode"])
    edge_class = str(port["class"])
    vertical = str(port["vertical_mode"])
    return [
        float(x) / 512.0 * 2.0 - 1.0,
        float(y) / 512.0 * 2.0 - 1.0,
        float(hx),
        float(hy),
        float(port.get("width_m", 0.0)) / 32.0,
        1.0 if mode == "road" else 0.0,
        1.0 if mode == "rail" else 0.0,
        *_one_hot(edge_class, ROAD_CLASSES),
        *_one_hot(edge_class, RAIL_CLASSES),
        *_one_hot(vertical, VERTICAL),
    ]


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
            row["estimated_width_m"] = float(record.get("width_m", 5.0))
        else:
            row["railway"] = record["class"]
        rows.append(row)
    if not rows:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")


def _program(payload: dict[str, Any], config: ProgramConfig) -> dict[str, Any]:
    roads = _frame(payload["target"]["roads"], "road")
    rail = _frame(payload["target"]["rail"], "rail")
    tile = TileSpec(
        city_id=str(payload["city_id"]),
        column=0,
        row=0,
        minx=0.0,
        miny=0.0,
        maxx=512.0,
        maxy=512.0,
    )
    graph = build_transport_graph(roads, rail, tile)
    state = {
        "tile": {
            "tile_id": payload["id"],
            "city_id": payload["city_id"],
            "area_id": None,
        },
        "coordinate_system": {
            "local_bounds": [0.0, 0.0, 512.0, 512.0],
        },
        "transport_graph": graph,
        "building_solids": payload["target"]["buildings"],
        "water": [],
        "green": [],
    }
    return city_state_to_program(state, config)


class ContextGraphProgramDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        maximum_samples: int = 32,
        maximum_commands: int = 768,
        maximum_ports: int = 96,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        graph = json.loads((self.root / "context-graph.json").read_text(encoding="utf-8"))
        self.feature_names = sorted(graph["nodes"][0]["features"])
        self.node_ids = [node["id"] for node in graph["nodes"]]
        self.node_index = {node_id: index for index, node_id in enumerate(self.node_ids)}
        values = []
        for node in graph["nodes"]:
            values.append(
                [
                    *[float(node["features"][name]) for name in self.feature_names],
                    *[float(value) for value in node["center_city_normalized"]],
                ]
            )
        self.base_context = np.asarray(values, dtype=np.float32)
        adjacency = np.eye(len(self.node_ids), dtype=np.float32)
        for edge in graph["edges"]:
            left = self.node_index[edge["from"]]
            right = self.node_index[edge["to"]]
            adjacency[left, right] = 1.0
            adjacency[right, left] = 1.0
        degree = adjacency.sum(axis=1, keepdims=True)
        self.adjacency = torch.from_numpy(adjacency / np.maximum(degree, 1.0))
        self.program_config = ProgramConfig(
            coordinate_bins=256,
            simplify_tolerance_m=2.0,
        )
        self.codec = CommandCodecConfig(
            program=self.program_config,
            maximum_nodes=512,
        )
        self.maximum_commands = int(maximum_commands)
        self.maximum_ports = int(maximum_ports)
        rows = [
            json.loads(line)
            for line in (self.root / "targets.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows.sort(
            key=lambda row: (
                row["rail"] > 0,
                min(row["boundary_ports"], 32),
                row["transport_length_m"],
            ),
            reverse=True,
        )
        candidates = []
        for row in rows:
            if row["boundary_ports"] < 2 or row["boundary_ports"] > maximum_ports:
                continue
            with gzip.open(self.root / row["sample_path"], "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            program = _program(payload, self.program_config)
            encoded = encode_program(program, self.codec)
            if len(encoded["op"]) > self.maximum_commands:
                continue
            candidates.append((row, payload, encoded))
            if len(candidates) >= maximum_samples:
                break
        if not candidates:
            raise RuntimeError("No context graph samples fit the overfit model limits")
        self.samples = candidates
        self.port_dimensions = len(_port_vector(self.samples[0][1]["input"]["boundary_ports"][0]))
        self.context_dimensions = self.base_context.shape[1] + 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, payload, encoded = self.samples[index]
        context = np.concatenate(
            [
                self.base_context.copy(),
                np.zeros((self.base_context.shape[0], 1), dtype=np.float32),
            ],
            axis=1,
        )
        parent = self.node_index[payload["parent_region_id"]]
        context[parent, : len(self.feature_names)] = 0.0
        context[parent, -1] = 1.0
        ports = np.zeros((self.maximum_ports, self.port_dimensions), dtype=np.float32)
        port_padding = np.ones(self.maximum_ports, dtype=bool)
        values = [_port_vector(port) for port in payload["input"]["boundary_ports"][: self.maximum_ports]]
        if values:
            ports[: len(values)] = np.asarray(values, dtype=np.float32)
            port_padding[: len(values)] = False
        commands = {}
        for field in FIELDS:
            data = np.zeros(self.maximum_commands, dtype=np.int64)
            data[: len(encoded[field])] = np.asarray(encoded[field], dtype=np.int64)
            commands[field] = torch.from_numpy(data)
        return {
            **commands,
            "context": torch.from_numpy(context),
            "ports": torch.from_numpy(ports),
            "port_padding": torch.from_numpy(port_padding),
            "sample_id": row["id"],
            "commands": len(encoded["op"]),
            "raw_ports": row["boundary_ports"],
            "rail_edges": row["rail"],
        }
