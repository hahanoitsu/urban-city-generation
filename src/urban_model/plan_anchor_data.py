from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from urban_model.city_plan_data import CityPlanConfig, CityPlanDataset
from urban_model.spatial_world_data import SpatialTensorConfig


@dataclass(frozen=True)
class PlanAnchorConfig:
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


def prepare_plan_anchor(
    sample: dict[str, Any],
    tensor_config: SpatialTensorConfig,
    anchor_config: PlanAnchorConfig,
    subanchors: np.ndarray,
) -> dict[str, torch.Tensor] | None:
    grid = anchor_config.grid_size
    slots = anchor_config.slots_per_cell
    node_count = int(sample["node_count"])
    edge_count = int(sample["edge_count"])
    if node_count > anchor_config.max_active_nodes:
        return None
    if edge_count > anchor_config.max_edges:
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
        assignments = _assign_group(local[indexes], subanchors)
        cell_count[cell] = len(indexes)
        for group_index, original_index in enumerate(indexes):
            slot = assignments[group_index]
            flat = cell * slots + slot
            original_to_anchor[original_index] = flat
            anchor_to_original[flat] = original_index
            slot_presence[cell, slot] = 1.0
            node_offset[cell, slot] = (
                local[original_index] - subanchors[slot]
            )
            node_mode[cell, slot] = int(
                sample["node_mode"][original_index]
            )
            node_vertical[cell, slot] = int(
                sample["node_vertical"][original_index]
            )
            node_boundary[cell, slot] = float(
                sample["node_boundary"][original_index]
            )

    active_anchor_ids = sorted(anchor_to_original)
    active_lookup = {
        anchor: index
        for index, anchor in enumerate(active_anchor_ids)
    }
    active_count = len(active_anchor_ids)
    active_ids = np.zeros(
        anchor_config.max_active_nodes,
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
        if (
            left_anchor is None
            or right_anchor is None
            or left_anchor == right_anchor
        ):
            continue

        left = active_lookup[left_anchor]
        right = active_lookup[right_anchor]
        shape = sample["edge_shape"][edge_index].clone()
        start = sample["node_xy"][left_original].clone()
        end = sample["node_xy"][right_original].clone()
        if left > right:
            left, right = right, left
            start, end = end, start
            shape = torch.flip(shape, dims=[0])

        chord = end - start
        chord_length = torch.linalg.vector_norm(
            chord
        ).clamp_min(1e-4)
        normal = torch.stack(
            [-chord[1], chord[0]]
        ) / chord_length
        curve = (
            shape * normal[None]
        ).sum(dim=-1) / chord_length
        key = (left, right)
        if key in kept_edges:
            continue
        kept_edges[key] = {
            "left": left,
            "right": right,
            "class": int(sample["edge_class"][edge_index]),
            "vertical": int(
                sample["edge_vertical"][edge_index]
            ),
            "width": float(
                sample["edge_width"][edge_index, 0]
            ),
            "curve": curve,
        }

    if len(kept_edges) > anchor_config.max_edges:
        return None

    node_degree = np.zeros(
        anchor_config.max_active_nodes,
        dtype=np.int64,
    )
    edge_pairs = np.zeros(
        (anchor_config.max_edges, 2),
        dtype=np.int64,
    )
    edge_class = np.zeros(
        anchor_config.max_edges,
        dtype=np.int64,
    )
    edge_vertical = np.zeros(
        anchor_config.max_edges,
        dtype=np.int64,
    )
    edge_width = np.zeros(
        (anchor_config.max_edges, 1),
        dtype=np.float32,
    )
    edge_curve = np.zeros(
        (
            anchor_config.max_edges,
            tensor_config.edge_shape_points,
        ),
        dtype=np.float32,
    )

    for index, value in enumerate(
        sorted(
            kept_edges.values(),
            key=lambda item: (
                item["left"],
                item["right"],
            ),
        )
    ):
        edge_pairs[index] = [
            value["left"],
            value["right"],
        ]
        node_degree[value["left"]] += 1
        node_degree[value["right"]] += 1
        edge_class[index] = value["class"]
        edge_vertical[index] = value["vertical"]
        edge_width[index, 0] = value["width"]
        edge_curve[index] = value["curve"].numpy()

    return {
        "cell_count": torch.from_numpy(cell_count),
        "slot_presence": torch.from_numpy(slot_presence),
        "node_offset": torch.from_numpy(node_offset),
        "node_mode": torch.from_numpy(node_mode),
        "node_vertical": torch.from_numpy(node_vertical),
        "node_boundary": torch.from_numpy(node_boundary),
        "active_count": torch.tensor(
            active_count,
            dtype=torch.long,
        ),
        "active_anchor_ids": torch.from_numpy(active_ids),
        "node_degree": torch.from_numpy(node_degree),
        "edge_count": torch.tensor(
            len(kept_edges),
            dtype=torch.long,
        ),
        "edge_pairs": torch.from_numpy(edge_pairs),
        "edge_class": torch.from_numpy(edge_class),
        "edge_vertical": torch.from_numpy(edge_vertical),
        "edge_width": torch.from_numpy(edge_width),
        "edge_curve": torch.from_numpy(edge_curve),
    }


class PlanAnchorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        *,
        tensor_config: SpatialTensorConfig | None = None,
        plan_config: CityPlanConfig | None = None,
        anchor_config: PlanAnchorConfig | None = None,
        maximum_samples: int | None = None,
    ) -> None:
        self.tensor_config = tensor_config or SpatialTensorConfig()
        self.plan_config = plan_config or CityPlanConfig()
        self.anchor_config = anchor_config or PlanAnchorConfig()
        base = CityPlanDataset(
            root,
            tensor_config=self.tensor_config,
            plan_config=self.plan_config,
            maximum_samples=maximum_samples,
        )
        self.feature_names = base.feature_names
        self.context_dimensions = base.context_dimensions
        self.style_dimensions = base.style_dimensions
        self.plan_dimensions = base.plan_dimensions
        self.orientation_dimensions = base.orientation_dimensions
        self.global_dimensions = base.global_dimensions
        self.global_mean = base.global_mean
        self.global_std = base.global_std
        self.base_rejected = dict(base.base_rejected)
        self.rejected = {
            "anchor": 0,
        }
        self.subanchors = _subanchor_xy(
            self.anchor_config.slots_per_cell
        )
        self.samples = []

        keep = {
            "plan_presence",
            "plan_log_counts",
            "plan_counts",
            "plan_orientation",
            "plan_orientation_mask",
            "plan_global",
            "plan_global_raw",
            "sample_id",
            "split",
        }

        for sample in base.samples:
            anchored = prepare_plan_anchor(
                sample,
                self.tensor_config,
                self.anchor_config,
                self.subanchors,
            )
            if anchored is None:
                self.rejected["anchor"] += 1
                continue
            prepared = {
                key: value
                for key, value in sample.items()
                if key in keep
            }
            prepared.update(anchored)
            self.samples.append(prepared)

        if not self.samples:
            raise RuntimeError(
                "All plan anchor samples were rejected"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
