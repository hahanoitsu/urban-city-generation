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


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sample_planned_frontier.py"
SPEC = importlib.util.spec_from_file_location(
    "sample_planned_frontier",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakePlannedModel:
    def __init__(self):
        self.ops = [
            OP_ROOT,
            OP_GROW,
            OP_CLOSE,
            OP_CLOSE,
            OP_EOS,
        ]
        self.calls = 0

    def encode_plan(self, batch):
        device = batch["plan_presence"].device
        return (
            torch.zeros(1, 2, 8, device=device),
            torch.zeros(
                1,
                2,
                dtype=torch.bool,
                device=device,
            ),
        )

    def decode_program(
        self,
        batch,
        memory,
        memory_padding,
        *,
        input_length,
        last_only=False,
    ):
        device = memory.device
        steps = 1 if last_only else input_length
        op = torch.full(
            (1, steps, 7),
            -20.0,
            device=device,
        )
        next_op = self.ops[
            min(self.calls, len(self.ops) - 1)
        ]
        self.calls += 1
        op[0, -1, next_op] = 20.0
        xy = torch.zeros(
            1,
            steps,
            2,
            device=device,
        )
        if next_op == OP_ROOT:
            xy[0, -1] = torch.tensor(
                [-0.5, 0.0],
                device=device,
            )
        elif next_op == OP_GROW:
            xy[0, -1] = torch.tensor(
                [0.25, 0.0],
                device=device,
            )

        return {
            "op": op,
            "xy_mean": xy,
            "xy_logstd": torch.full(
                (1, steps, 2),
                -5.0,
                device=device,
            ),
            "node_mode": torch.stack(
                [
                    torch.full(
                        (1, steps),
                        20.0,
                        device=device,
                    ),
                    torch.full(
                        (1, steps),
                        -20.0,
                        device=device,
                    ),
                ],
                dim=-1,
            ),
            "node_vertical": torch.zeros(
                1,
                steps,
                4,
                device=device,
            ),
            "node_boundary": torch.zeros(
                1,
                steps,
                device=device,
            ),
            "edge_class": torch.zeros(
                1,
                steps,
                8,
                device=device,
            ),
            "edge_vertical": torch.zeros(
                1,
                steps,
                4,
                device=device,
            ),
            "width_mean": torch.ones(
                1,
                steps,
                1,
                device=device,
            )
            * 0.2,
            "width_logstd": torch.full(
                (1, steps, 1),
                -5.0,
                device=device,
            ),
            "curve_mean": torch.zeros(
                1,
                steps,
                2,
                device=device,
            ),
            "curve_logstd": torch.full(
                (1, steps, 2),
                -5.0,
                device=device,
            ),
        }


def make_sample():
    steps = 16
    return {
        "plan_presence": torch.zeros(16, 8),
        "plan_log_counts": torch.zeros(16, 8),
        "plan_orientation": torch.zeros(16, 4, 2),
        "plan_global": torch.zeros(8),
        "plan_global_raw": torch.tensor(
            [2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ),
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
        "program_progress": torch.zeros(steps, 7),
    }


def test_planned_frontier_rollout_respects_plan_budget():
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
        FakePlannedModel(),
        sample,
        tensor_config,
        program_config,
        torch.device("cpu"),
        0.0,
    )

    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
    assert graph["roots"] == 1
    assert graph["node_budget"] == 2
    assert graph["edge_budget"] == 1
    assert graph["terminated_eos"] is True
