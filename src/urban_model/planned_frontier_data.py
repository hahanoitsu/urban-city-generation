from __future__ import annotations

from typing import Any

import torch

from urban_model.city_plan_data import CityPlanConfig, CityPlanDataset
from urban_model.frontier_data import (
    OP_GROW,
    OP_LINK,
    OP_ROOT,
    FrontierProgramConfig,
    build_frontier_program,
)
from urban_model.spatial_world_data import SpatialTensorConfig


PROGRESS_CHANNELS = (
    "nodes",
    "edges",
    "components",
    "major",
    "secondary",
    "local",
    "rail",
)


def build_program_progress(
    program: dict[str, torch.Tensor],
    plan_global_raw: torch.Tensor,
) -> torch.Tensor:
    length = int(program["program_length"])
    maximum = program["program_op"].shape[0]
    progress = torch.zeros(
        maximum,
        len(PROGRESS_CHANNELS),
        dtype=torch.float32,
    )

    planned_nodes = max(float(plan_global_raw[0]), 1.0)
    planned_edges = max(float(plan_global_raw[1]), 1.0)
    planned_components = max(float(plan_global_raw[2]), 1.0)
    class_budgets = [
        max(float(plan_global_raw[4]) * planned_edges, 1.0),
        max(float(plan_global_raw[5]) * planned_edges, 1.0),
        max(float(plan_global_raw[6]) * planned_edges, 1.0),
        max(float(plan_global_raw[3]) * planned_edges, 1.0),
    ]
    values = torch.zeros(len(PROGRESS_CHANNELS))

    for index in range(length):
        op = int(program["program_op"][index])
        if op == OP_ROOT:
            values[0] += 1.0 / planned_nodes
            values[2] += 1.0 / planned_components
        elif op == OP_GROW:
            values[0] += 1.0 / planned_nodes
            values[1] += 1.0 / planned_edges
            edge_class = int(program["program_edge_class"][index])
            group = edge_class if edge_class < 3 else 3
            values[3 + group] += 1.0 / class_budgets[group]
        elif op == OP_LINK:
            values[1] += 1.0 / planned_edges
            edge_class = int(program["program_edge_class"][index])
            group = edge_class if edge_class < 3 else 3
            values[3 + group] += 1.0 / class_budgets[group]
        progress[index] = values.clamp(0.0, 2.0)
    return progress


class PlannedFrontierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        *,
        tensor_config: SpatialTensorConfig | None = None,
        plan_config: CityPlanConfig | None = None,
        program_config: FrontierProgramConfig | None = None,
        maximum_samples: int | None = None,
    ) -> None:
        self.tensor_config = tensor_config or SpatialTensorConfig()
        self.plan_config = plan_config or CityPlanConfig()
        self.program_config = program_config or FrontierProgramConfig(
            curve_points=self.tensor_config.edge_shape_points,
        )
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
        self.presence_pos_weight = base.presence_pos_weight
        self.global_mean = base.global_mean
        self.global_std = base.global_std
        self.base_rejected = dict(base.base_rejected)
        self.rejected = {"steps": 0, "nodes": 0}
        self.samples: list[dict[str, Any]] = []

        for sample in base.samples:
            try:
                program = build_frontier_program(
                    sample,
                    self.program_config,
                )
            except ValueError as error:
                name = str(error).split(":", 1)[0]
                self.rejected[name] = self.rejected.get(name, 0) + 1
                continue
            prepared = dict(sample)
            prepared.update(program)
            prepared["program_progress"] = build_program_progress(
                program,
                sample["plan_global_raw"],
            )
            self.samples.append(prepared)

        if not self.samples:
            raise RuntimeError("All planned frontier programs were rejected")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
