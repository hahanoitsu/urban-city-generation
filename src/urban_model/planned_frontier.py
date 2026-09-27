from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from urban_model.frontier_data import OP_NAMES
from urban_model.planned_frontier_data import PROGRESS_CHANNELS
from urban_model.spatial_world import _decoder


@dataclass(frozen=True)
class PlannedFrontierConfig:
    plan_dimensions: int
    orientation_dimensions: int
    global_dimensions: int
    grid_size: int = 16
    max_steps: int = 1024
    curve_points: int = 8
    model_dimensions: int = 256
    heads: int = 8
    decoder_layers: int = 6
    feedforward_dimensions: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> "PlannedFrontierConfig":
        return cls(**value)


def _grid_coordinates(grid_size: int) -> torch.Tensor:
    values = []
    for row in range(grid_size):
        for column in range(grid_size):
            values.append(
                [
                    (column + 0.5) / grid_size * 2.0 - 1.0,
                    (row + 0.5) / grid_size * 2.0 - 1.0,
                ]
            )
    return torch.tensor(values, dtype=torch.float32)


class PlannedFrontierArchitect(nn.Module):
    def __init__(self, config: PlannedFrontierConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dimensions
        plan_input = (
            config.plan_dimensions * 2
            + config.orientation_dimensions * 2
            + 2
        )
        token_input = (
            2
            + 2
            + 1
            + config.curve_points
            + 1
            + len(PROGRESS_CHANNELS)
        )

        self.register_buffer(
            "grid_coordinates",
            _grid_coordinates(config.grid_size),
            persistent=False,
        )
        self.plan_cell = nn.Sequential(
            nn.Linear(plan_input, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.plan_global = nn.Sequential(
            nn.Linear(config.global_dimensions, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.plan_norm = nn.LayerNorm(d)

        self.op = nn.Embedding(len(OP_NAMES), d)
        self.node_mode = nn.Embedding(2, d)
        self.node_vertical = nn.Embedding(4, d)
        self.edge_class = nn.Embedding(8, d)
        self.edge_vertical = nn.Embedding(4, d)
        self.position = nn.Embedding(config.max_steps, d)
        self.continuous = nn.Sequential(
            nn.Linear(token_input, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.input_norm = nn.LayerNorm(d)
        self.decoder = _decoder(
            d,
            config.heads,
            config.feedforward_dimensions,
            config.dropout,
            config.decoder_layers,
        )
        self.output_norm = nn.LayerNorm(d)

        self.op_head = nn.Linear(d, len(OP_NAMES))
        self.xy_head = nn.Linear(d, 4)
        self.node_mode_head = nn.Linear(d, 2)
        self.node_vertical_head = nn.Linear(d, 4)
        self.node_boundary_head = nn.Linear(d, 1)
        self.edge_class_head = nn.Linear(d, 8)
        self.edge_vertical_head = nn.Linear(d, 4)
        self.width_head = nn.Linear(d, 2)
        self.curve_head = nn.Linear(
            d,
            config.curve_points * 2,
        )

    def encode_plan(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = self.grid_coordinates[None].expand(
            batch["plan_presence"].shape[0],
            -1,
            -1,
        )
        cell_input = torch.cat(
            [
                batch["plan_presence"],
                batch["plan_log_counts"],
                batch["plan_orientation"].flatten(2),
                coordinates,
            ],
            dim=-1,
        )
        cells = self.plan_cell(cell_input)
        global_token = self.plan_global(
            batch["plan_global"]
        )[:, None]
        memory = self.plan_norm(
            torch.cat([cells, global_token], dim=1)
        )
        padding = torch.zeros(
            memory.shape[:2],
            dtype=torch.bool,
            device=memory.device,
        )
        return memory, padding

    def _token_embedding(
        self,
        batch: dict[str, torch.Tensor],
        length: int,
    ) -> torch.Tensor:
        op = batch["program_op"][:, :length]
        continuous = torch.cat(
            [
                batch["program_xy"][:, :length],
                batch["program_active_xy"][:, :length],
                batch["program_edge_width"][:, :length],
                batch["program_curve"][:, :length],
                batch["program_node_boundary"][:, :length, None],
                batch["program_progress"][:, :length],
            ],
            dim=-1,
        )
        positions = torch.arange(length, device=op.device)
        hidden = (
            self.op(op)
            + self.node_mode(
                batch["program_node_mode"][:, :length]
            )
            + self.node_vertical(
                batch["program_node_vertical"][:, :length]
            )
            + self.edge_class(
                batch["program_edge_class"][:, :length]
            )
            + self.edge_vertical(
                batch["program_edge_vertical"][:, :length]
            )
            + self.position(positions)[None]
            + self.continuous(continuous)
        )
        return self.input_norm(hidden)

    def decode_program(
        self,
        batch: dict[str, torch.Tensor],
        memory: torch.Tensor,
        memory_padding: torch.Tensor,
        *,
        input_length: int,
        last_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden = self._token_embedding(
            batch,
            input_length,
        )
        causal = torch.triu(
            torch.ones(
                input_length,
                input_length,
                dtype=torch.bool,
                device=hidden.device,
            ),
            diagonal=1,
        )
        positions = torch.arange(
            input_length,
            device=hidden.device,
        )
        padding = positions[None] >= (
            batch["program_length"] - 1
        )[:, None]
        decoded = self.output_norm(
            self.decoder(
                hidden,
                memory,
                tgt_mask=causal,
                tgt_key_padding_mask=padding,
                memory_key_padding_mask=memory_padding,
            )
        )
        if last_only:
            decoded = decoded[:, -1:]

        xy = self.xy_head(decoded)
        width = self.width_head(decoded)
        curve = self.curve_head(decoded).reshape(
            decoded.shape[0],
            decoded.shape[1],
            self.config.curve_points,
            2,
        )
        return {
            "op": self.op_head(decoded),
            "xy_mean": torch.tanh(xy[..., :2]),
            "xy_logstd": xy[..., 2:].clamp(-5.0, 1.0),
            "node_mode": self.node_mode_head(decoded),
            "node_vertical": self.node_vertical_head(decoded),
            "node_boundary": self.node_boundary_head(decoded).squeeze(-1),
            "edge_class": self.edge_class_head(decoded),
            "edge_vertical": self.edge_vertical_head(decoded),
            "width_mean": width[..., :1],
            "width_logstd": width[..., 1:].clamp(-5.0, 1.0),
            "curve_mean": torch.tanh(curve[..., 0]),
            "curve_logstd": curve[..., 1].clamp(-5.0, 1.0),
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        input_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_length is None:
            input_length = self.config.max_steps - 1
        memory, padding = self.encode_plan(batch)
        return self.decode_program(
            batch,
            memory,
            padding,
            input_length=input_length,
        )
