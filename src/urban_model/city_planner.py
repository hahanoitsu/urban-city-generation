from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from urban_model.context_encoder_v2 import SpatialContextEncoderV2
from urban_model.spatial_world import _decoder


@dataclass(frozen=True)
class CityPlannerConfig:
    context_dimensions: int
    style_dimensions: int
    plan_dimensions: int
    orientation_dimensions: int
    global_dimensions: int
    grid_size: int = 16
    context_line_points: int = 6
    model_dimensions: int = 256
    heads: int = 8
    context_layers: int = 4
    planner_layers: int = 4
    feedforward_dimensions: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CityPlannerConfig":
        return cls(**value)


def _grid_coordinates(grid_size: int) -> torch.Tensor:
    values = []
    for row in range(grid_size):
        for column in range(grid_size):
            values.append(
                [(column + 0.5) / grid_size * 2.0 - 1.0, (row + 0.5) / grid_size * 2.0 - 1.0]
            )
    return torch.tensor(values, dtype=torch.float32)


class CityPlanner(nn.Module):
    def __init__(self, config: CityPlannerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dimensions
        self.context = SpatialContextEncoderV2(config)
        self.register_buffer(
            "grid_coordinates", _grid_coordinates(config.grid_size), persistent=False
        )
        self.grid_position = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.grid_decoder = _decoder(
            d, config.heads, config.feedforward_dimensions, config.dropout, config.planner_layers
        )
        self.grid_norm = nn.LayerNorm(d)
        self.presence_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.plan_dimensions)
        )
        self.count_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.plan_dimensions)
        )
        self.orientation_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.orientation_dimensions * 2)
        )
        self.global_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, config.global_dimensions)
        )

    def forward(
        self, batch: dict[str, torch.Tensor], encoded_context=None
    ) -> dict[str, torch.Tensor]:
        memory, padding, pool = self.context(batch) if encoded_context is None else encoded_context
        batch_size = memory.shape[0]
        queries = self.grid_position(self.grid_coordinates)[None].expand(batch_size, -1, -1)
        hidden = self.grid_norm(self.grid_decoder(queries, memory, memory_key_padding_mask=padding))
        orientation = self.orientation_head(hidden).reshape(
            hidden.shape[0], hidden.shape[1], self.config.orientation_dimensions, 2
        )
        return {
            "plan_presence": self.presence_head(hidden),
            "plan_log_count": torch.nn.functional.softplus(self.count_head(hidden)),
            "plan_orientation": torch.tanh(orientation),
            "plan_global": self.global_head(pool),
        }
