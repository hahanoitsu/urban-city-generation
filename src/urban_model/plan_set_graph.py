from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from urban_model.spatial_world import _decoder


@dataclass(frozen=True)
class PlanSetGraphConfig:
    plan_dimensions: int
    orientation_dimensions: int
    global_dimensions: int
    plan_grid_size: int = 16
    max_nodes: int = 384
    max_edges: int = 512
    max_degree: int = 8
    edge_shape_points: int = 8
    model_dimensions: int = 256
    edge_dimensions: int = 64
    heads: int = 8
    plan_layers: int = 4
    query_layers: int = 6
    feedforward_dimensions: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> "PlanSetGraphConfig":
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


def _query_coordinates(count: int) -> torch.Tensor:
    columns = int(count**0.5)
    if columns * columns < count:
        columns += 1
    rows = (count + columns - 1) // columns
    values = []
    for row in range(rows):
        for column in range(columns):
            if len(values) == count:
                break
            values.append(
                [
                    (column + 0.5) / columns * 1.9 - 0.95,
                    (row + 0.5) / rows * 1.9 - 0.95,
                ]
            )
    return torch.tensor(values, dtype=torch.float32)


class PlanSetGraphArchitect(nn.Module):
    def __init__(self, config: PlanSetGraphConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dimensions
        e = config.edge_dimensions
        plan_input = (
            config.plan_dimensions * 2
            + config.orientation_dimensions * 2
            + 2
        )

        self.register_buffer(
            "plan_coordinates",
            _grid_coordinates(config.plan_grid_size),
            persistent=False,
        )
        self.register_buffer(
            "query_reference",
            _query_coordinates(config.max_nodes),
            persistent=False,
        )

        self.plan_cell = nn.Sequential(
            nn.Linear(plan_input, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        plan_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dimensions,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.plan_encoder = nn.TransformerEncoder(
            plan_layer,
            num_layers=config.plan_layers,
        )
        self.plan_global = nn.Sequential(
            nn.Linear(config.global_dimensions, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.plan_norm = nn.LayerNorm(d)

        self.query_embedding = nn.Embedding(
            config.max_nodes,
            d,
        )
        self.query_position = nn.Sequential(
            nn.Linear(2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.query_decoder = _decoder(
            d,
            config.heads,
            config.feedforward_dimensions,
            config.dropout,
            config.query_layers,
        )
        self.query_norm = nn.LayerNorm(d)

        self.node_presence = nn.Linear(d, 1)
        self.node_xy = nn.Linear(d, 2)
        self.node_mode = nn.Linear(d, 2)
        self.node_vertical = nn.Linear(d, 4)
        self.node_boundary = nn.Linear(d, 1)
        self.node_degree = nn.Linear(
            d,
            config.max_degree + 1,
        )

        self.edge_node = nn.Linear(d, e)
        self.edge_space = nn.Sequential(
            nn.Linear(7, e),
            nn.GELU(),
            nn.Linear(e, e),
        )
        self.edge_plan = nn.Linear(d, e)
        self.edge_pair = nn.Sequential(
            nn.Linear(e, e),
            nn.GELU(),
            nn.Linear(e, e),
        )
        self.edge_exists = nn.Linear(e, 1)
        self.edge_class = nn.Linear(e, 8)
        self.edge_vertical = nn.Linear(e, 4)
        self.edge_width = nn.Linear(e, 1)
        self.edge_curve = nn.Linear(
            e,
            config.edge_shape_points,
        )

    def encode_plan(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = batch["plan_presence"].shape[0]
        coordinates = self.plan_coordinates[None].expand(
            batch_size,
            -1,
            -1,
        )
        values = torch.cat(
            [
                batch["plan_presence"],
                batch["plan_log_counts"],
                batch["plan_orientation"].flatten(2),
                coordinates,
            ],
            dim=-1,
        )
        cells = self.plan_encoder(
            self.plan_cell(values)
        )
        global_token = self.plan_global(
            batch["plan_global"]
        )[:, None]
        return self.plan_norm(
            torch.cat(
                [cells, global_token],
                dim=1,
            )
        )

    def _midpoint_plan(
        self,
        plan_cells: torch.Tensor,
        midpoint: torch.Tensor,
    ) -> torch.Tensor:
        grid = self.config.plan_grid_size
        unit = (
            (midpoint + 1.0) * 0.5
        ).clamp(
            0.0,
            1.0 - 1e-7,
        )
        column = torch.floor(
            unit[..., 0] * grid
        ).long()
        row = torch.floor(
            unit[..., 1] * grid
        ).long()
        index = row * grid + column
        batch_size = plan_cells.shape[0]
        dimensions = plan_cells.shape[-1]
        flat = index.reshape(
            batch_size,
            -1,
        )
        gathered = torch.gather(
            plan_cells,
            1,
            flat[:, :, None].expand(
                -1,
                -1,
                dimensions,
            ),
        )
        return gathered.reshape(
            *index.shape,
            dimensions,
        )

    def edge_predictions(
        self,
        hidden: torch.Tensor,
        xy: torch.Tensor,
        plan_cells: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node = self.edge_node(hidden)
        left = node[:, :, None, :]
        right = node[:, None, :, :]
        left_xy = xy[:, :, None, :]
        right_xy = xy[:, None, :, :]
        delta = right_xy - left_xy
        midpoint = (
            left_xy + right_xy
        ) * 0.5
        spatial = torch.cat(
            [
                delta,
                delta.abs(),
                torch.linalg.vector_norm(
                    delta,
                    dim=-1,
                    keepdim=True,
                ),
                midpoint.expand(
                    -1,
                    delta.shape[1],
                    -1,
                    -1,
                ),
            ],
            dim=-1,
        )
        plan_pair = self._midpoint_plan(
            plan_cells,
            midpoint.expand(
                -1,
                delta.shape[1],
                -1,
                -1,
            ),
        )
        pair = self.edge_pair(
            left
            + right
            + self.edge_space(spatial)
            + self.edge_plan(plan_pair)
        )
        return {
            "edge_exists": self.edge_exists(
                pair
            ).squeeze(-1),
            "edge_class": self.edge_class(pair),
            "edge_vertical": self.edge_vertical(pair),
            "edge_width": self.edge_width(pair),
            "edge_curve": 0.75
            * torch.tanh(
                self.edge_curve(pair)
            ),
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        memory = self.encode_plan(batch)
        plan_cells = memory[
            :, : self.config.plan_grid_size**2
        ]
        batch_size = memory.shape[0]
        queries = (
            self.query_embedding.weight
            + self.query_position(
                self.query_reference
            )
        )[None].expand(
            batch_size,
            -1,
            -1,
        )
        hidden = self.query_norm(
            self.query_decoder(
                queries,
                memory,
            )
        )
        reference = self.query_reference[None]
        reference_logit = torch.atanh(
            reference.clamp(
                -0.999,
                0.999,
            )
        )
        xy = torch.tanh(
            reference_logit
            + self.node_xy(hidden)
        )
        output = {
            "query_hidden": hidden,
            "node_presence": self.node_presence(
                hidden
            ).squeeze(-1),
            "node_xy": xy,
            "node_mode": self.node_mode(hidden),
            "node_vertical": self.node_vertical(
                hidden
            ),
            "node_boundary": self.node_boundary(
                hidden
            ).squeeze(-1),
            "node_degree": self.node_degree(hidden),
        }
        output.update(
            self.edge_predictions(
                hidden,
                xy,
                plan_cells,
            )
        )
        return output
