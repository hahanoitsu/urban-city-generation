from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from urban_model.frontier_data import OP_NAMES
from urban_model.spatial_world import SpatialContextEncoder, _decoder


@dataclass(frozen=True)
class FrontierArchitectConfig:
    context_dimensions: int
    style_dimensions: int
    max_steps: int = 1024
    max_nodes: int = 384
    context_line_points: int = 6
    curve_points: int = 8
    model_dimensions: int = 256
    heads: int = 8
    context_layers: int = 4
    decoder_layers: int = 6
    feedforward_dimensions: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FrontierArchitectConfig":
        return cls(**value)


class FrontierArchitect(nn.Module):
    def __init__(self, config: FrontierArchitectConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dimensions

        self.context = SpatialContextEncoder(config)
        self.op = nn.Embedding(len(OP_NAMES), d)
        self.node_mode = nn.Embedding(2, d)
        self.node_vertical = nn.Embedding(4, d)
        self.edge_class = nn.Embedding(8, d)
        self.edge_vertical = nn.Embedding(4, d)
        self.pointer = nn.Embedding(config.max_nodes + 1, d)
        self.active = nn.Embedding(config.max_nodes + 1, d)
        self.position = nn.Embedding(config.max_steps, d)
        self.continuous = nn.Sequential(
            nn.Linear(2 + 2 + 1 + config.curve_points + 1, d),
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
        self.curve_head = nn.Linear(d, config.curve_points * 2)
        self.pointer_head = nn.Linear(d, config.max_nodes)

    def _token_embedding(
        self,
        batch: dict[str, torch.Tensor],
        length: int,
    ) -> torch.Tensor:
        op = batch["program_op"][:, :length]
        node_mode = batch["program_node_mode"][:, :length]
        node_vertical = batch["program_node_vertical"][:, :length]
        edge_class = batch["program_edge_class"][:, :length]
        edge_vertical = batch["program_edge_vertical"][:, :length]
        pointer = (batch["program_pointer"][:, :length] + 1).clamp(
            0,
            self.config.max_nodes,
        )
        active = (batch["program_active_node"][:, :length] + 1).clamp(
            0,
            self.config.max_nodes,
        )
        continuous = torch.cat(
            [
                batch["program_xy"][:, :length],
                batch["program_active_xy"][:, :length],
                batch["program_edge_width"][:, :length],
                batch["program_curve"][:, :length],
                batch["program_node_boundary"][:, :length, None],
            ],
            dim=-1,
        )
        positions = torch.arange(length, device=op.device)
        hidden = (
            self.op(op)
            + self.node_mode(node_mode)
            + self.node_vertical(node_vertical)
            + self.edge_class(edge_class)
            + self.edge_vertical(edge_vertical)
            + self.pointer(pointer)
            + self.active(active)
            + self.position(positions)[None]
            + self.continuous(continuous)
        )
        return self.input_norm(hidden)

    def encode_context(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, memory_padding, _pool = self.context(batch)
        return memory, memory_padding

    def decode_program(
        self,
        batch: dict[str, torch.Tensor],
        memory: torch.Tensor,
        memory_padding: torch.Tensor,
        *,
        input_length: int,
        last_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden = self._token_embedding(batch, input_length)
        causal = torch.triu(
            torch.ones(
                input_length,
                input_length,
                dtype=torch.bool,
                device=hidden.device,
            ),
            diagonal=1,
        )
        positions = torch.arange(input_length, device=hidden.device)
        padding = positions[None] >= (batch["program_length"] - 1)[:, None]
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
            "pointer": self.pointer_head(decoded),
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        input_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_length is None:
            input_length = self.config.max_steps - 1
        memory, memory_padding = self.encode_context(batch)
        return self.decode_program(
            batch,
            memory,
            memory_padding,
            input_length=input_length,
        )
