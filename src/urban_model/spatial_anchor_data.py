from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from urban_model.spatial_world_data import SpatialTensorConfig, SpatialWorldDataset


@dataclass(frozen=True)
class SpatialAnchorConfig:
    grid_size: int = 32
    slots_per_cell: int = 12
    max_active_nodes: int = 384
    max_edges: int = 512


def _subanchor_xy(slots: int) -> np.ndarray:
    columns = 4
    rows = int(np.ceil(slots / columns))
    values = []
    for row in range(rows):
        for column in range(columns):
            if len(values) == slots:
                break
            values.append(
                [
                    (column + 0.5) / columns,
                    (row + 0.5) / rows,
                ]
            )
    return np.asarray(values, dtype=np.float32)


def _assign_group(
    positions: np.ndarray,
    anchors: np.ndarray,
) -> list[int]:
    distances = np.square(
        positions[:, None, :] - anchors[None, :, :]
    ).sum(axis=-1)
    pairs = [
        (float(distances[node, slot]), node, slot)
        for node in range(len(positions))
        for slot in range(len(anchors))
    ]
    pairs.sort()
    result = [-1] * len(positions)
    used = set()
    assigned = 0
    for _distance, node, slot in pairs:
        if result[node] >= 0 or slot in used:
            continue
        result[node] = slot
        used.add(slot)
        assigned += 1
        if assigned == len(positions):
            break
    return result


class AnchoredSpatialWorldDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        *,
        tensor_config: SpatialTensorConfig | None = None,
        anchor_config: SpatialAnchorConfig | None = None,
        maximum_samples: int | None = None,
    ) -> None:
        self.tensor_config = tensor_config or SpatialTensorConfig()
        self.anchor_config = anchor_config or SpatialAnchorConfig()
        base = SpatialWorldDataset(
            root,
            config=self.tensor_config,
            maximum_samples=maximum_samples,
        )
        self.feature_names = base.feature_names
        self.feature_mean = base.feature_mean
        self.feature_std = base.feature_std
        self.context_dimensions = base.context_dimensions
        self.style_dimensions = base.style_dimensions
        self.base_rejected = dict(base.rejected)
        self.rejected = {
            "anchor_cell_overflow": 0,
            "active_nodes": 0,
            "edges": 0,
        }
        self.duplicate_edges = 0
        self.samples = []
        self.subanchors = _subanchor_xy(self.anchor_config.slots_per_cell)

        for sample in base.samples:
            prepared = self._prepare(sample)
            if prepared is not None:
                self.samples.append(prepared)

        if not self.samples:
            raise RuntimeError("All anchored spatial samples were rejected")

    def _prepare(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        grid = self.anchor_config.grid_size
        slots = self.anchor_config.slots_per_cell
        node_count = int(sample["node_count"])
        edge_count = int(sample["edge_count"])
        if node_count > self.anchor_config.max_active_nodes:
            self.rejected["active_nodes"] += 1
            return None
        if edge_count > self.anchor_config.max_edges:
            self.rejected["edges"] += 1
            return None

        xy = sample["node_xy"][:node_count].numpy()
        unit = np.clip((xy + 1.0) * 0.5, 0.0, 1.0 - 1e-7)
        columns = np.floor(unit[:, 0] * grid).astype(np.int64)
        rows = np.floor(unit[:, 1] * grid).astype(np.int64)
        local = unit * grid - np.stack([columns, rows], axis=-1)

        groups: dict[int, list[int]] = {}
        for index in range(node_count):
            cell = int(rows[index] * grid + columns[index])
            groups.setdefault(cell, []).append(index)

        if any(len(values) > slots for values in groups.values()):
            self.rejected["anchor_cell_overflow"] += 1
            return None

        cells = grid * grid
        cell_count = np.zeros(cells, dtype=np.int64)
        slot_presence = np.zeros((cells, slots), dtype=np.float32)
        node_offset = np.zeros((cells, slots, 2), dtype=np.float32)
        node_mode = np.zeros((cells, slots), dtype=np.int64)
        node_vertical = np.zeros((cells, slots), dtype=np.int64)
        node_boundary = np.zeros((cells, slots), dtype=np.float32)
        original_to_anchor = {}
        anchor_to_original = {}

        for cell, indexes in groups.items():
            assignments = _assign_group(local[indexes], self.subanchors)
            cell_count[cell] = len(indexes)
            for group_index, original_index in enumerate(indexes):
                slot = assignments[group_index]
                flat = cell * slots + slot
                original_to_anchor[original_index] = flat
                anchor_to_original[flat] = original_index
                slot_presence[cell, slot] = 1.0
                node_offset[cell, slot] = (
                    local[original_index] - self.subanchors[slot]
                )
                node_mode[cell, slot] = int(sample["node_mode"][original_index])
                node_vertical[cell, slot] = int(
                    sample["node_vertical"][original_index]
                )
                node_boundary[cell, slot] = float(
                    sample["node_boundary"][original_index]
                )

        active_anchor_ids = sorted(anchor_to_original)
        active_lookup = {
            anchor: index for index, anchor in enumerate(active_anchor_ids)
        }
        active_count = len(active_anchor_ids)
        active_ids = np.zeros(
            self.anchor_config.max_active_nodes,
            dtype=np.int64,
        )
        active_ids[:active_count] = np.asarray(
            active_anchor_ids,
            dtype=np.int64,
        )

        kept_edges = {}
        for edge_index in range(edge_count):
            left_original = int(sample["edge_from"][edge_index])
            right_original = int(sample["edge_to"][edge_index])
            left_anchor = original_to_anchor.get(left_original)
            right_anchor = original_to_anchor.get(right_original)
            if left_anchor is None or right_anchor is None or left_anchor == right_anchor:
                continue
            left = active_lookup[left_anchor]
            right = active_lookup[right_anchor]
            shape = sample["edge_shape"][edge_index].clone()
            if left > right:
                left, right = right, left
                shape = torch.flip(shape, dims=[0])
            key = (left, right)
            if key in kept_edges:
                self.duplicate_edges += 1
                continue
            kept_edges[key] = {
                "left": left,
                "right": right,
                "class": int(sample["edge_class"][edge_index]),
                "vertical": int(sample["edge_vertical"][edge_index]),
                "width": float(sample["edge_width"][edge_index, 0]),
                "shape": shape,
            }

        if len(kept_edges) > self.anchor_config.max_edges:
            self.rejected["edges"] += 1
            return None

        node_degree = np.zeros(
            self.anchor_config.max_active_nodes,
            dtype=np.int64,
        )
        edge_pairs = np.zeros(
            (self.anchor_config.max_edges, 2),
            dtype=np.int64,
        )
        edge_class = np.zeros(self.anchor_config.max_edges, dtype=np.int64)
        edge_vertical = np.zeros(self.anchor_config.max_edges, dtype=np.int64)
        edge_width = np.zeros(
            (self.anchor_config.max_edges, 1),
            dtype=np.float32,
        )
        edge_shape = np.zeros(
            (
                self.anchor_config.max_edges,
                self.tensor_config.edge_shape_points,
                2,
            ),
            dtype=np.float32,
        )
        for index, value in enumerate(
            sorted(kept_edges.values(), key=lambda item: (item["left"], item["right"]))
        ):
            edge_pairs[index] = [value["left"], value["right"]]
            node_degree[value["left"]] += 1
            node_degree[value["right"]] += 1
            edge_class[index] = value["class"]
            edge_vertical[index] = value["vertical"]
            edge_width[index, 0] = value["width"]
            edge_shape[index] = value["shape"].numpy()

        keep = {
            key: value
            for key, value in sample.items()
            if key
            in {
                "context_cells",
                "style",
                "controls",
                "context_line_points",
                "context_line_mode",
                "context_line_class",
                "context_line_vertical",
                "context_line_width",
                "context_line_length",
                "context_line_padding",
                "ports",
                "port_mode",
                "port_class",
                "port_vertical",
                "port_padding",
                "sample_id",
                "split",
            }
        }
        keep.update(
            {
                "cell_count": torch.from_numpy(cell_count),
                "slot_presence": torch.from_numpy(slot_presence),
                "node_offset": torch.from_numpy(node_offset),
                "node_mode": torch.from_numpy(node_mode),
                "node_vertical": torch.from_numpy(node_vertical),
                "node_boundary": torch.from_numpy(node_boundary),
                "active_count": torch.tensor(active_count, dtype=torch.long),
                "active_anchor_ids": torch.from_numpy(active_ids),
                "node_degree": torch.from_numpy(node_degree),
                "edge_count": torch.tensor(len(kept_edges), dtype=torch.long),
                "edge_pairs": torch.from_numpy(edge_pairs),
                "edge_class": torch.from_numpy(edge_class),
                "edge_vertical": torch.from_numpy(edge_vertical),
                "edge_width": torch.from_numpy(edge_width),
                "edge_shape": torch.from_numpy(edge_shape),
            }
        )
        return keep

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
