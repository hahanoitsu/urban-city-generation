from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from urban_model.spatial_world_data import SpatialTensorConfig, SpatialWorldDataset


PLAN_CHANNELS = (
    "junctions",
    "road_nodes",
    "rail_nodes",
    "major_corridor",
    "secondary_corridor",
    "local_corridor",
    "rail_corridor",
    "boundary_nodes",
)

ORIENTATION_CHANNELS = (
    "major_corridor",
    "secondary_corridor",
    "local_corridor",
    "rail_corridor",
)

GLOBAL_CHANNELS = (
    "nodes",
    "edges",
    "components",
    "rail_share",
    "major_share",
    "secondary_share",
    "local_share",
    "boundary_share",
)


@dataclass(frozen=True)
class CityPlanConfig:
    grid_size: int = 16
    corridor_samples_per_cell: int = 4


def _cell_index(xy: torch.Tensor, grid_size: int) -> tuple[int, int]:
    unit = ((xy + 1.0) * 0.5).clamp(0.0, 1.0 - 1e-7)
    column = int(unit[0] * grid_size)
    row = int(unit[1] * grid_size)
    return row, column


def _components(
    node_count: int,
    edge_from: torch.Tensor,
    edge_to: torch.Tensor,
    edge_count: int,
) -> int:
    adjacency = [[] for _ in range(node_count)]
    for index in range(edge_count):
        left = int(edge_from[index])
        right = int(edge_to[index])
        if left == right or left >= node_count or right >= node_count:
            continue
        adjacency[left].append(right)
        adjacency[right].append(left)
    seen = set()
    components = 0
    for start in range(node_count):
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
    return components


def _edge_polyline(
    sample: dict[str, Any],
    edge_index: int,
) -> torch.Tensor:
    left = int(sample["edge_from"][edge_index])
    right = int(sample["edge_to"][edge_index])
    start = sample["node_xy"][left]
    end = sample["node_xy"][right]
    residual = sample["edge_shape"][edge_index]
    points = [start]
    for point_index in range(residual.shape[0]):
        fraction = (point_index + 1) / (residual.shape[0] + 1)
        base = start + (end - start) * fraction
        points.append(base + residual[point_index])
    points.append(end)
    return torch.stack(points)


def _corridor_cells(
    polyline: torch.Tensor,
    grid_size: int,
    samples_per_cell: int,
) -> dict[tuple[int, int], list[torch.Tensor]]:
    cells: dict[tuple[int, int], list[torch.Tensor]] = {}
    for index in range(polyline.shape[0] - 1):
        start = polyline[index]
        end = polyline[index + 1]
        delta = end - start
        span_cells = float(delta.abs().max()) * 0.5 * grid_size
        steps = max(
            2,
            int(math.ceil(span_cells * samples_per_cell)) + 1,
        )
        angle = torch.atan2(delta[1], delta[0])
        orientation = torch.stack(
            [
                torch.cos(angle * 2.0),
                torch.sin(angle * 2.0),
            ]
        )
        for step in range(steps):
            fraction = step / (steps - 1)
            point = start + delta * fraction
            cell = _cell_index(point, grid_size)
            cells.setdefault(cell, []).append(orientation)
    return cells


def build_city_plan(
    sample: dict[str, Any],
    config: CityPlanConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    grid = config.grid_size
    counts = torch.zeros(
        grid,
        grid,
        len(PLAN_CHANNELS),
        dtype=torch.float32,
    )
    orientation_sum = torch.zeros(
        grid,
        grid,
        len(ORIENTATION_CHANNELS),
        2,
        dtype=torch.float32,
    )
    orientation_weight = torch.zeros(
        grid,
        grid,
        len(ORIENTATION_CHANNELS),
        dtype=torch.float32,
    )
    node_count = int(sample["node_count"])
    edge_count = int(sample["edge_count"])

    for index in range(node_count):
        row, column = _cell_index(sample["node_xy"][index], grid)
        counts[row, column, 0] += 1.0
        if int(sample["node_mode"][index]) == 0:
            counts[row, column, 1] += 1.0
        else:
            counts[row, column, 2] += 1.0
        if float(sample["node_boundary"][index]) > 0.5:
            counts[row, column, 7] += 1.0

    road_edges = 0
    rail_edges = 0
    class_counts = [0, 0, 0]
    for index in range(edge_count):
        edge_class = int(sample["edge_class"][index])
        if edge_class < 3:
            plan_index = 3 + edge_class
            orientation_index = edge_class
            road_edges += 1
            class_counts[edge_class] += 1
        else:
            plan_index = 6
            orientation_index = 3
            rail_edges += 1

        cells = _corridor_cells(
            _edge_polyline(sample, index),
            grid,
            config.corridor_samples_per_cell,
        )
        for (row, column), orientations in cells.items():
            counts[row, column, plan_index] += 1.0
            values = torch.stack(orientations)
            orientation_sum[
                row,
                column,
                orientation_index,
            ] += values.mean(dim=0)
            orientation_weight[
                row,
                column,
                orientation_index,
            ] += 1.0

    orientation = orientation_sum / orientation_weight[
        ..., None
    ].clamp_min(1.0)
    orientation_mask = orientation_weight > 0

    components = _components(
        node_count,
        sample["edge_from"],
        sample["edge_to"],
        edge_count,
    )
    boundary_nodes = int(
        (sample["node_boundary"][:node_count] > 0.5).sum()
    )
    total_edges = max(edge_count, 1)
    total_nodes = max(node_count, 1)
    global_values = torch.tensor(
        [
            float(node_count),
            float(edge_count),
            float(components),
            rail_edges / total_edges,
            class_counts[0] / total_edges,
            class_counts[1] / total_edges,
            class_counts[2] / total_edges,
            boundary_nodes / total_nodes,
        ],
        dtype=torch.float32,
    )
    return (
        counts.reshape(grid * grid, -1),
        orientation.reshape(
            grid * grid,
            len(ORIENTATION_CHANNELS),
            2,
        ),
        orientation_mask.reshape(
            grid * grid,
            len(ORIENTATION_CHANNELS),
        ),
        global_values,
    )


class CityPlanDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        *,
        tensor_config: SpatialTensorConfig | None = None,
        plan_config: CityPlanConfig | None = None,
        maximum_samples: int | None = None,
    ) -> None:
        self.tensor_config = tensor_config or SpatialTensorConfig()
        self.plan_config = plan_config or CityPlanConfig()
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
        self.samples = []

        counts = []
        orientations = []
        orientation_masks = []
        globals_ = []
        for sample in base.samples:
            (
                plan_counts,
                plan_orientation,
                plan_orientation_mask,
                global_values,
            ) = build_city_plan(
                sample,
                self.plan_config,
            )
            counts.append(plan_counts)
            orientations.append(plan_orientation)
            orientation_masks.append(plan_orientation_mask)
            globals_.append(global_values)

        count_stack = torch.stack(counts)
        global_stack = torch.stack(globals_)
        presence = count_stack > 0
        positives = presence.sum(dim=(0, 1)).to(torch.float32)
        total = float(
            presence.shape[0] * presence.shape[1]
        )
        negatives = total - positives
        self.presence_pos_weight = (
            negatives / positives.clamp_min(1.0)
        ).clamp(1.0, 30.0)
        self.global_mean = global_stack.mean(dim=0)
        self.global_std = global_stack.std(dim=0).clamp_min(0.05)

        for (
            sample,
            plan_counts,
            plan_orientation,
            plan_orientation_mask,
            global_values,
        ) in zip(
            base.samples,
            counts,
            orientations,
            orientation_masks,
            globals_,
            strict=True,
        ):
            prepared = dict(sample)
            prepared["plan_counts"] = plan_counts
            prepared["plan_presence"] = (
                plan_counts > 0
            ).to(torch.float32)
            prepared["plan_log_counts"] = torch.log1p(
                plan_counts
            )
            prepared["plan_orientation"] = plan_orientation
            prepared["plan_orientation_mask"] = (
                plan_orientation_mask
            )
            prepared["plan_global_raw"] = global_values
            prepared["plan_global"] = (
                global_values - self.global_mean
            ) / self.global_std
            self.samples.append(prepared)

    @property
    def plan_dimensions(self) -> int:
        return len(PLAN_CHANNELS)

    @property
    def orientation_dimensions(self) -> int:
        return len(ORIENTATION_CHANNELS)

    @property
    def global_dimensions(self) -> int:
        return len(GLOBAL_CHANNELS)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
