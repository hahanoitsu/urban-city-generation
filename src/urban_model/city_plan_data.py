from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from urban_model.spatial_world_data import SpatialTensorConfig, SpatialWorldDataset


PLAN_CHANNELS = (
    "junctions",
    "road_nodes",
    "rail_nodes",
    "major_edges",
    "secondary_edges",
    "local_edges",
    "rail_edges",
    "boundary_nodes",
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


def build_city_plan(
    sample: dict[str, Any],
    config: CityPlanConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = config.grid_size
    plan = torch.zeros(
        grid,
        grid,
        len(PLAN_CHANNELS),
        dtype=torch.float32,
    )
    node_count = int(sample["node_count"])
    edge_count = int(sample["edge_count"])

    for index in range(node_count):
        row, column = _cell_index(sample["node_xy"][index], grid)
        plan[row, column, 0] += 1.0
        if int(sample["node_mode"][index]) == 0:
            plan[row, column, 1] += 1.0
        else:
            plan[row, column, 2] += 1.0
        if float(sample["node_boundary"][index]) > 0.5:
            plan[row, column, 7] += 1.0

    edge_classes = sample["edge_class"][:edge_count]
    road_edges = 0
    rail_edges = 0
    class_counts = [0, 0, 0]
    for index in range(edge_count):
        left = int(sample["edge_from"][index])
        right = int(sample["edge_to"][index])
        midpoint = (
            sample["node_xy"][left] + sample["node_xy"][right]
        ) * 0.5
        row, column = _cell_index(midpoint, grid)
        edge_class = int(edge_classes[index])
        if edge_class < 3:
            plan[row, column, 3 + edge_class] += 1.0
            road_edges += 1
            class_counts[edge_class] += 1
        else:
            plan[row, column, 6] += 1.0
            rail_edges += 1

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
    return plan.reshape(grid * grid, -1), global_values


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

        plans = []
        globals_ = []
        for sample in base.samples:
            plan, global_values = build_city_plan(
                sample,
                self.plan_config,
            )
            plans.append(plan)
            globals_.append(global_values)

        plan_stack = torch.stack(plans)
        global_stack = torch.stack(globals_)
        self.plan_mean = plan_stack.mean(dim=(0, 1))
        self.plan_std = plan_stack.std(dim=(0, 1)).clamp_min(0.25)
        self.global_mean = global_stack.mean(dim=0)
        self.global_std = global_stack.std(dim=0).clamp_min(0.05)

        for sample, plan, global_values in zip(
            base.samples,
            plans,
            globals_,
            strict=True,
        ):
            prepared = dict(sample)
            prepared["plan_grid_raw"] = plan
            prepared["plan_global_raw"] = global_values
            prepared["plan_grid"] = (
                plan - self.plan_mean
            ) / self.plan_std
            prepared["plan_global"] = (
                global_values - self.global_mean
            ) / self.global_std
            self.samples.append(prepared)

    @property
    def plan_dimensions(self) -> int:
        return len(PLAN_CHANNELS)

    @property
    def global_dimensions(self) -> int:
        return len(GLOBAL_CHANNELS)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
