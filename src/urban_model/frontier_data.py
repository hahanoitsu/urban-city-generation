from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from urban_model.spatial_world_data import SpatialTensorConfig, SpatialWorldDataset


OP_PAD = 0
OP_BOS = 1
OP_ROOT = 2
OP_GROW = 3
OP_LINK = 4
OP_CLOSE = 5
OP_EOS = 6
OP_NAMES = ("pad", "bos", "root", "grow", "link", "close", "eos")


@dataclass(frozen=True)
class FrontierProgramConfig:
    max_steps: int = 1024
    max_nodes: int = 384
    curve_points: int = 8


def _curve_from_shape(
    shape: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    reverse: bool,
) -> torch.Tensor:
    chord = end - start
    length = torch.linalg.vector_norm(chord).clamp_min(1e-5)
    normal = torch.stack([-chord[1], chord[0]]) / length
    curve = (shape * normal[None]).sum(dim=-1) / length
    if reverse:
        curve = -torch.flip(curve, dims=[0])
    return curve


def _edge_sort_key(
    active: int,
    neighbour: int,
    edge_index: int,
    sample: dict[str, Any],
):
    current = sample["node_xy"][active]
    other = sample["node_xy"][neighbour]
    delta = other - current
    angle = math.atan2(float(delta[1]), float(delta[0]))
    return (
        int(sample["edge_mode"][edge_index]),
        int(sample["edge_class"][edge_index]),
        angle,
        float(other[1]),
        float(other[0]),
        neighbour,
    )


def _component_nodes(
    adjacency: list[list[tuple[int, int]]],
) -> list[list[int]]:
    seen = set()
    components = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        values = []
        while queue:
            node = queue.pop()
            values.append(node)
            for neighbour, _edge in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(values)
    return components


def _root_key(node: int, sample: dict[str, Any]):
    return (
        -float(sample["node_boundary"][node]),
        int(sample["node_mode"][node]),
        float(sample["node_xy"][node, 1]),
        float(sample["node_xy"][node, 0]),
        node,
    )


def build_frontier_program(
    sample: dict[str, Any],
    config: FrontierProgramConfig,
) -> dict[str, torch.Tensor]:
    node_count = int(sample["node_count"])
    edge_count = int(sample["edge_count"])
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    for edge_index in range(edge_count):
        left = int(sample["edge_from"][edge_index])
        right = int(sample["edge_to"][edge_index])
        if left == right:
            continue
        adjacency[left].append((right, edge_index))
        adjacency[right].append((left, edge_index))

    components = _component_nodes(adjacency)
    components.sort(key=lambda values: min(_root_key(node, sample) for node in values))

    ops = [OP_BOS]
    xy = [[0.0, 0.0]]
    node_mode = [0]
    node_vertical = [0]
    node_boundary = [0.0]
    edge_class = [0]
    edge_vertical = [0]
    edge_width = [0.0]
    curve = [[0.0] * config.curve_points]
    pointer = [0]
    active_node = [-1]

    generated: dict[int, int] = {}
    emitted_edges = set()

    def append(
        op: int,
        *,
        xy_value=(0.0, 0.0),
        node_mode_value=0,
        node_vertical_value=0,
        node_boundary_value=0.0,
        edge_class_value=0,
        edge_vertical_value=0,
        edge_width_value=0.0,
        curve_value=None,
        pointer_value=0,
        active_value=-1,
    ):
        ops.append(op)
        xy.append([float(xy_value[0]), float(xy_value[1])])
        node_mode.append(int(node_mode_value))
        node_vertical.append(int(node_vertical_value))
        node_boundary.append(float(node_boundary_value))
        edge_class.append(int(edge_class_value))
        edge_vertical.append(int(edge_vertical_value))
        edge_width.append(float(edge_width_value))
        if curve_value is None:
            curve.append([0.0] * config.curve_points)
        else:
            values = curve_value.detach().cpu().tolist()
            if len(values) != config.curve_points:
                raise ValueError("curve_points")
            curve.append([float(value) for value in values])
        pointer.append(int(pointer_value))
        active_node.append(int(active_value))

    for component in components:
        root = min(component, key=lambda node: _root_key(node, sample))
        generated[root] = len(generated)
        append(
            OP_ROOT,
            xy_value=sample["node_xy"][root],
            node_mode_value=sample["node_mode"][root],
            node_vertical_value=sample["node_vertical"][root],
            node_boundary_value=sample["node_boundary"][root],
            active_value=generated[root],
        )
        queue = deque([root])

        while queue:
            active = queue[0]
            incidents = sorted(
                adjacency[active],
                key=lambda value: _edge_sort_key(
                    active,
                    value[0],
                    value[1],
                    sample,
                ),
            )
            for neighbour, edge_index in incidents:
                if edge_index in emitted_edges:
                    continue
                emitted_edges.add(edge_index)

                edge_left = int(sample["edge_from"][edge_index])
                edge_right = int(sample["edge_to"][edge_index])
                reverse = not (edge_left == active and edge_right == neighbour)
                edge_curve = _curve_from_shape(
                    sample["edge_shape"][edge_index],
                    sample["node_xy"][edge_left],
                    sample["node_xy"][edge_right],
                    reverse,
                )

                if neighbour not in generated:
                    generated[neighbour] = len(generated)
                    delta = (
                        sample["node_xy"][neighbour]
                        - sample["node_xy"][active]
                    ) / 2.0
                    append(
                        OP_GROW,
                        xy_value=delta,
                        node_mode_value=sample["node_mode"][neighbour],
                        node_vertical_value=sample["node_vertical"][neighbour],
                        node_boundary_value=sample["node_boundary"][neighbour],
                        edge_class_value=sample["edge_class"][edge_index],
                        edge_vertical_value=sample["edge_vertical"][edge_index],
                        edge_width_value=sample["edge_width"][edge_index, 0],
                        curve_value=edge_curve,
                        active_value=generated[active],
                    )
                    queue.append(neighbour)
                else:
                    append(
                        OP_LINK,
                        edge_class_value=sample["edge_class"][edge_index],
                        edge_vertical_value=sample["edge_vertical"][edge_index],
                        edge_width_value=sample["edge_width"][edge_index, 0],
                        curve_value=edge_curve,
                        pointer_value=generated[neighbour],
                        active_value=generated[active],
                    )

            queue.popleft()
            next_active = generated[queue[0]] if queue else -1
            append(OP_CLOSE, active_value=next_active)

    append(OP_EOS)

    length = len(ops)
    if length > config.max_steps:
        raise ValueError(f"steps:{length}")
    if len(generated) > config.max_nodes:
        raise ValueError(f"nodes:{len(generated)}")

    def padded_tensor(values, shape, dtype):
        result = torch.zeros(shape, dtype=dtype)
        if len(values):
            result[: len(values)] = torch.as_tensor(values, dtype=dtype)
        return result

    return {
        "program_length": torch.tensor(length, dtype=torch.long),
        "program_op": padded_tensor(
            ops,
            (config.max_steps,),
            torch.long,
        ),
        "program_xy": padded_tensor(
            xy,
            (config.max_steps, 2),
            torch.float32,
        ),
        "program_node_mode": padded_tensor(
            node_mode,
            (config.max_steps,),
            torch.long,
        ),
        "program_node_vertical": padded_tensor(
            node_vertical,
            (config.max_steps,),
            torch.long,
        ),
        "program_node_boundary": padded_tensor(
            node_boundary,
            (config.max_steps,),
            torch.float32,
        ),
        "program_edge_class": padded_tensor(
            edge_class,
            (config.max_steps,),
            torch.long,
        ),
        "program_edge_vertical": padded_tensor(
            edge_vertical,
            (config.max_steps,),
            torch.long,
        ),
        "program_edge_width": padded_tensor(
            edge_width,
            (config.max_steps, 1),
            torch.float32,
        ),
        "program_curve": padded_tensor(
            curve,
            (config.max_steps, config.curve_points),
            torch.float32,
        ),
        "program_pointer": padded_tensor(
            pointer,
            (config.max_steps,),
            torch.long,
        ),
        "program_active_node": padded_tensor(
            active_node,
            (config.max_steps,),
            torch.long,
        ),
        "program_nodes": torch.tensor(len(generated), dtype=torch.long),
        "program_edges": torch.tensor(len(emitted_edges), dtype=torch.long),
    }


class FrontierProgramDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        *,
        tensor_config: SpatialTensorConfig | None = None,
        program_config: FrontierProgramConfig | None = None,
        maximum_samples: int | None = None,
    ) -> None:
        self.tensor_config = tensor_config or SpatialTensorConfig()
        self.program_config = program_config or FrontierProgramConfig(
            curve_points=self.tensor_config.edge_shape_points,
        )
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
        self.rejected = {"steps": 0, "nodes": 0, "curve_points": 0}
        self.samples = []

        for sample in base.samples:
            try:
                program = build_frontier_program(sample, self.program_config)
            except ValueError as error:
                name = str(error).split(":", 1)[0]
                self.rejected[name] = self.rejected.get(name, 0) + 1
                continue
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
                    "node_count",
                    "node_xy",
                    "node_mode",
                    "node_vertical",
                    "node_boundary",
                    "edge_count",
                    "edge_from",
                    "edge_to",
                    "edge_mode",
                    "edge_class",
                    "edge_vertical",
                    "edge_width",
                    "edge_shape",
                }
            }
            keep.update(program)
            self.samples.append(keep)

        if not self.samples:
            raise RuntimeError("All frontier programs were rejected")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
