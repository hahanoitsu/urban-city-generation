import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from urban_model.frontier_data import (
    OP_CLOSE,
    OP_EOS,
    OP_GROW,
    OP_ROOT,
    FrontierProgramConfig,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sample_frontier_architect.py"
SPEC = importlib.util.spec_from_file_location("sample_frontier_architect", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeFrontierModel:
    def __init__(self):
        self.ops = [OP_ROOT, OP_GROW, OP_CLOSE, OP_CLOSE, OP_EOS]

    def encode_context(self, batch):
        device = batch["context_cells"].device
        return (
            torch.zeros(1, 1, 8, device=device),
            torch.zeros(1, 1, dtype=torch.bool, device=device),
        )

    def decode_program(self, batch, memory, memory_padding, *, input_length):
        device = memory.device
        steps = input_length
        op = torch.full((1, steps, 7), -20.0, device=device)
        next_op = self.ops[min(steps - 1, len(self.ops) - 1)]
        op[0, -1, next_op] = 20.0
        xy_mean = torch.zeros(1, steps, 2, device=device)
        if next_op == OP_GROW:
            xy_mean[0, -1] = torch.tensor([0.2, 0.0], device=device)
        return {
            "op": op,
            "xy_mean": xy_mean,
            "xy_logstd": torch.full((1, steps, 2), -5.0, device=device),
            "node_mode": torch.stack(
                [
                    torch.full((1, steps), 20.0, device=device),
                    torch.full((1, steps), -20.0, device=device),
                ],
                dim=-1,
            ),
            "node_vertical": torch.zeros(1, steps, 4, device=device),
            "node_boundary": torch.zeros(1, steps, device=device),
            "edge_class": torch.zeros(1, steps, 8, device=device),
            "edge_vertical": torch.zeros(1, steps, 4, device=device),
            "width_mean": torch.ones(1, steps, 1, device=device) * 0.2,
            "width_logstd": torch.full((1, steps, 1), -5.0, device=device),
            "curve_mean": torch.zeros(1, steps, 2, device=device),
            "curve_logstd": torch.full((1, steps, 2), -5.0, device=device),
            "pointer": torch.zeros(1, steps, 8, device=device),
        }


def make_sample():
    steps = 16
    return {
        "context_cells": torch.zeros(2, 4),
        "style": torch.zeros(1),
        "controls": torch.zeros(1),
        "context_line_points": torch.zeros(1, 2, 2),
        "context_line_mode": torch.zeros(1, dtype=torch.long),
        "context_line_class": torch.zeros(1, dtype=torch.long),
        "context_line_vertical": torch.zeros(1, dtype=torch.long),
        "context_line_width": torch.zeros(1, 1),
        "context_line_length": torch.zeros(1, 1),
        "context_line_padding": torch.ones(1, dtype=torch.bool),
        "ports": torch.zeros(1, 5),
        "port_mode": torch.zeros(1, dtype=torch.long),
        "port_class": torch.zeros(1, dtype=torch.long),
        "port_vertical": torch.zeros(1, dtype=torch.long),
        "port_padding": torch.ones(1, dtype=torch.bool),
        "program_length": torch.tensor(2),
        "program_op": torch.zeros(steps, dtype=torch.long),
        "program_xy": torch.zeros(steps, 2),
        "program_node_mode": torch.zeros(steps, dtype=torch.long),
        "program_node_vertical": torch.zeros(steps, dtype=torch.long),
        "program_node_boundary": torch.zeros(steps),
        "program_edge_class": torch.zeros(steps, dtype=torch.long),
        "program_edge_vertical": torch.zeros(steps, dtype=torch.long),
        "program_edge_width": torch.zeros(steps, 1),
        "program_curve": torch.zeros(steps, 2),
        "program_pointer": torch.zeros(steps, dtype=torch.long),
        "program_active_node": torch.zeros(steps, dtype=torch.long),
        "program_active_xy": torch.zeros(steps, 2),
    }


def test_frontier_rollout_builds_connected_growth_graph():
    sample = make_sample()
    tensor_config = SimpleNamespace(
        target_size_m=1024.0,
        width_scale_m=32.0,
        max_edges=8,
    )
    program_config = FrontierProgramConfig(
        max_steps=16,
        max_nodes=8,
        curve_points=2,
    )
    graph = MODULE.rollout(
        FakeFrontierModel(),
        sample,
        tensor_config,
        program_config,
        torch.device("cpu"),
        0.0,
    )
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["from_node"] == 0
    assert graph["edges"][0]["to_node"] == 1
    assert graph["roots"] == 1
