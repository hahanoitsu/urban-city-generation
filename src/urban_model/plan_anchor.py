from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from urban_model.spatial_world import _decoder


@dataclass(frozen=True)
class PlanAnchorModelConfig:
    plan_dimensions: int
    orientation_dimensions: int
    global_dimensions: int
    plan_grid_size: int = 16
    anchor_grid_size: int = 32
    slots_per_cell: int = 12
    max_active_nodes: int = 384
    max_edges: int = 512
    max_degree: int = 8
    edge_shape_points: int = 8
    model_dimensions: int = 256
    edge_dimensions: int = 64
    heads: int = 8
    plan_layers: int = 4
    anchor_layers: int = 4
    feedforward_dimensions: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> "PlanAnchorModelConfig":
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


def _subanchor_coordinates(slots: int) -> torch.Tensor:
    columns = 4
    rows = (slots + columns - 1) // columns
    values = []
    for row in range(rows):
        for column in range(columns):
            if len(values) == slots:
                break
            values.append(
                [
                    (column + 0.5) / columns,
                    (row + 0.5) / rows,
                ]
            )
    return torch.tensor(values, dtype=torch.float32)


class PlanAnchorArchitect(nn.Module):
    def __init__(self, config: PlanAnchorModelConfig) -> None:
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
            "anchor_coordinates",
            _grid_coordinates(config.anchor_grid_size),
            persistent=False,
        )
        self.register_buffer(
            "subanchor_coordinates",
            _subanchor_coordinates(config.slots_per_cell),
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

        self.anchor_position = nn.Sequential(
            nn.Linear(2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.anchor_decoder = _decoder(
            d,
            config.heads,
            config.feedforward_dimensions,
            config.dropout,
            config.anchor_layers,
        )
        self.anchor_norm = nn.LayerNorm(d)
        self.subanchor_position = nn.Sequential(
            nn.Linear(2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.node_hidden = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.slot_presence = nn.Linear(d, 1)
        self.node_offset = nn.Linear(d, 2)
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
        cells = self.plan_cell(values)
        cells = self.plan_encoder(cells)
        global_token = self.plan_global(
            batch["plan_global"]
        )[:, None]
        return self.plan_norm(
            torch.cat([cells, global_token], dim=1)
        )

    def _anchor_field(
        self,
        plan_memory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = plan_memory.shape[0]
        queries = self.anchor_position(
            self.anchor_coordinates
        )[None].expand(
            batch_size,
            -1,
            -1,
        )
        hidden = self.anchor_decoder(
            queries,
            plan_memory,
        )
        return self.anchor_norm(hidden)

    def _node_predictions(
        self,
        anchors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sub = self.subanchor_position(
            self.subanchor_coordinates
        )
        hidden = self.node_hidden(
            anchors[:, :, None, :]
            + sub[None, None, :, :]
        )
        return {
            "node_hidden": hidden,
            "slot_presence": self.slot_presence(
                hidden
            ).squeeze(-1),
            "node_offset": 0.55
            * torch.tanh(self.node_offset(hidden)),
            "node_mode": self.node_mode(hidden),
            "node_vertical": self.node_vertical(hidden),
            "node_boundary": self.node_boundary(
                hidden
            ).squeeze(-1),
        }

    def node_positions(
        self,
        node_offset: torch.Tensor,
    ) -> torch.Tensor:
        grid = self.config.anchor_grid_size
        cells = self.anchor_coordinates.reshape(
            grid * grid,
            1,
            2,
        )
        sub = self.subanchor_coordinates.reshape(
            1,
            self.config.slots_per_cell,
            2,
        )
        cell_unit = (cells + 1.0) * 0.5
        column = torch.floor(
            cell_unit[..., 0] * grid
        ) / grid
        row = torch.floor(
            cell_unit[..., 1] * grid
        ) / grid
        origin = torch.stack(
            [column, row],
            dim=-1,
        )
        anchor = origin + sub / grid
        unit = anchor[None] + node_offset / grid
        return unit.clamp(0.0, 1.0) * 2.0 - 1.0

    def _active_nodes(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        active_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, cells, slots, dimensions = hidden.shape
        flat_hidden = hidden.reshape(
            batch,
            cells * slots,
            dimensions,
        )
        flat_positions = positions.reshape(
            batch,
            cells * slots,
            2,
        )
        gather_hidden = active_ids[:, :, None].expand(
            -1,
            -1,
            dimensions,
        )
        gather_position = active_ids[:, :, None].expand(
            -1,
            -1,
            2,
        )
        return (
            torch.gather(
                flat_hidden,
                1,
                gather_hidden,
            ),
            torch.gather(
                flat_positions,
                1,
                gather_position,
            ),
        )

    def _midpoint_plan(
        self,
        plan_cells: torch.Tensor,
        midpoint: torch.Tensor,
    ) -> torch.Tensor:
        grid = self.config.plan_grid_size
        unit = ((midpoint + 1.0) * 0.5).clamp(
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
        batch = plan_cells.shape[0]
        dimensions = plan_cells.shape[-1]
        flat_index = index.reshape(batch, -1)
        gathered = torch.gather(
            plan_cells,
            1,
            flat_index[:, :, None].expand(
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
        active_hidden: torch.Tensor,
        active_positions: torch.Tensor,
        plan_cells: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node = self.edge_node(active_hidden)
        left = node[:, :, None, :]
        right = node[:, None, :, :]
        left_position = active_positions[:, :, None, :]
        right_position = active_positions[:, None, :, :]
        delta = right_position - left_position
        midpoint = (
            left_position + right_position
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
            * torch.tanh(self.edge_curve(pair)),
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        memory = self.encode_plan(batch)
        plan_cells = memory[
            :, : self.config.plan_grid_size ** 2
        ]
        anchors = self._anchor_field(memory)
        output = self._node_predictions(anchors)
        positions = self.node_positions(
            output["node_offset"]
        )
        active_slots = max(
            1,
            int(batch["active_count"].max()),
        )
        active_ids = batch[
            "active_anchor_ids"
        ][:, :active_slots]
        active_hidden, active_positions = self._active_nodes(
            output["node_hidden"],
            positions,
            active_ids,
        )
        output["node_degree"] = self.node_degree(
            active_hidden
        )
        output.update(
            self.edge_predictions(
                active_hidden,
                active_positions,
                plan_cells,
            )
        )
        output["node_positions"] = positions
        output["active_anchor_ids"] = active_ids
        output["active_positions"] = active_positions
        return output

    @torch.inference_mode()
    def generate(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        memory = self.encode_plan(batch)
        plan_cells = memory[
            :, : self.config.plan_grid_size ** 2
        ]
        anchors = self._anchor_field(memory)
        output = self._node_predictions(anchors)
        positions = self.node_positions(
            output["node_offset"]
        )

        batch_size = positions.shape[0]
        maximum = self.config.max_active_nodes
        flat_scores = output[
            "slot_presence"
        ].reshape(batch_size, -1)
        active_ids = torch.zeros(
            batch_size,
            maximum,
            dtype=torch.long,
            device=positions.device,
        )
        active_count = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=positions.device,
        )
        for batch_index in range(batch_size):
            count = int(
                round(
                    float(
                        batch["plan_global_raw"][
                            batch_index,
                            0,
                        ]
                    )
                )
            )
            count = max(
                1,
                min(count, maximum),
            )
            chosen = torch.topk(
                flat_scores[batch_index],
                k=count,
            ).indices
            chosen = torch.sort(chosen).values
            active_ids[
                batch_index,
                :count,
            ] = chosen
            active_count[batch_index] = count

        active_slots = max(
            1,
            int(active_count.max()),
        )
        ids = active_ids[:, :active_slots]
        active_hidden, active_positions = self._active_nodes(
            output["node_hidden"],
            positions,
            ids,
        )
        output["node_degree"] = self.node_degree(
            active_hidden
        )
        output.update(
            self.edge_predictions(
                active_hidden,
                active_positions,
                plan_cells,
            )
        )
        output["node_positions"] = positions
        output["active_anchor_ids"] = ids
        output["active_count"] = active_count
        output["active_positions"] = active_positions
        return output
