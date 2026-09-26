from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from urban_model.spatial_world import SpatialContextEncoder, _decoder


@dataclass(frozen=True)
class SpatialAnchorModelConfig:
    context_dimensions: int
    style_dimensions: int
    grid_size: int = 32
    slots_per_cell: int = 12
    max_active_nodes: int = 384
    edge_shape_points: int = 8
    model_dimensions: int = 256
    latent_dimensions: int = 32
    heads: int = 8
    context_layers: int = 4
    cell_layers: int = 4
    feedforward_dimensions: int = 1024
    edge_dimensions: int = 64
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpatialAnchorModelConfig":
        return cls(**value)


def _cell_coordinates(grid: int) -> torch.Tensor:
    values = []
    for row in range(grid):
        for column in range(grid):
            values.append(
                [
                    (column + 0.5) / grid * 2.0 - 1.0,
                    (row + 0.5) / grid * 2.0 - 1.0,
                ]
            )
    return torch.tensor(values, dtype=torch.float32)


def _subanchor_coordinates(slots: int) -> torch.Tensor:
    columns = 4
    rows = math.ceil(slots / columns)
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


class SpatialAnchorArchitect(nn.Module):
    def __init__(self, config: SpatialAnchorModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dimensions
        z = config.latent_dimensions
        e = config.edge_dimensions

        self.context = SpatialContextEncoder(config)
        self.register_buffer(
            "cell_coordinates",
            _cell_coordinates(config.grid_size),
            persistent=False,
        )
        self.register_buffer(
            "subanchor_coordinates",
            _subanchor_coordinates(config.slots_per_cell),
            persistent=False,
        )

        self.cell_position = nn.Sequential(
            nn.Linear(2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.cell_decoder = _decoder(
            d,
            config.heads,
            config.feedforward_dimensions,
            config.dropout,
            config.cell_layers,
        )
        self.cell_norm = nn.LayerNorm(d)

        self.target_slot = nn.Sequential(
            nn.Linear(4, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.target_mode = nn.Embedding(2, d)
        self.target_vertical = nn.Embedding(4, d)
        self.target_norm = nn.LayerNorm(d)

        self.prior = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, z * 2),
        )
        self.posterior = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, z * 2),
        )
        self.latent = nn.Sequential(
            nn.Linear(z, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

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
        self.cell_count = nn.Linear(d, config.slots_per_cell + 1)
        self.slot_score = nn.Linear(d, 1)
        self.node_offset = nn.Linear(d, 2)
        self.node_mode = nn.Linear(d, 2)
        self.node_vertical = nn.Linear(d, 4)
        self.node_boundary = nn.Linear(d, 1)

        self.edge_node = nn.Linear(d, e)
        self.edge_space = nn.Sequential(
            nn.Linear(3, e),
            nn.GELU(),
            nn.Linear(e, e),
        )
        self.edge_pair = nn.Sequential(
            nn.Linear(e, e),
            nn.GELU(),
            nn.Linear(e, e),
        )
        self.edge_relation = nn.Linear(e, 9)
        self.edge_vertical = nn.Linear(e, 4)
        self.edge_width = nn.Linear(e, 1)
        self.edge_shape = nn.Linear(e, config.edge_shape_points * 2)

    def _cell_field(
        self,
        memory: torch.Tensor,
        padding: torch.Tensor,
    ) -> torch.Tensor:
        batch = memory.shape[0]
        queries = self.cell_position(self.cell_coordinates)[None].expand(
            batch,
            -1,
            -1,
        )
        hidden = self.cell_decoder(
            queries,
            memory,
            memory_key_padding_mask=padding,
        )
        return self.cell_norm(hidden)

    def _target_summary(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        presence = batch["slot_presence"]
        values = torch.cat(
            [
                batch["node_offset"],
                batch["node_boundary"][:, :, :, None],
                presence[:, :, :, None],
            ],
            dim=-1,
        )
        hidden = (
            self.target_slot(values)
            + self.target_mode(batch["node_mode"])
            + self.target_vertical(batch["node_vertical"])
        )
        hidden = hidden * presence[:, :, :, None]
        denominator = presence.sum(dim=2, keepdim=True).clamp_min(1.0)
        return self.target_norm(hidden.sum(dim=2) / denominator)

    def _latent_parameters(
        self,
        cells: torch.Tensor,
        target: torch.Tensor | None,
    ):
        prior = self.prior(cells)
        prior_mu, prior_logvar = prior.chunk(2, dim=-1)
        if target is None:
            return prior_mu, prior_logvar, prior_mu, prior_logvar
        posterior = self.posterior(torch.cat([cells, target], dim=-1))
        posterior_mu, posterior_logvar = posterior.chunk(2, dim=-1)
        return prior_mu, prior_logvar, posterior_mu, posterior_logvar

    def _sample(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        noise = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar.clamp(-10.0, 10.0)) * noise * temperature

    def _node_predictions(
        self,
        cells: torch.Tensor,
        latent: torch.Tensor,
    ):
        batch = cells.shape[0]
        cell_state = cells + self.latent(latent)
        sub = self.subanchor_position(self.subanchor_coordinates)
        hidden = self.node_hidden(
            cell_state[:, :, None, :] + sub[None, None, :, :]
        )
        return {
            "cell_state": cell_state,
            "node_hidden": hidden,
            "cell_count": self.cell_count(cell_state),
            "slot_score": self.slot_score(hidden).squeeze(-1),
            "node_offset": 0.55 * torch.tanh(self.node_offset(hidden)),
            "node_mode": self.node_mode(hidden),
            "node_vertical": self.node_vertical(hidden),
            "node_boundary": self.node_boundary(hidden).squeeze(-1),
        }

    def node_positions(
        self,
        node_offset: torch.Tensor,
    ) -> torch.Tensor:
        grid = self.config.grid_size
        cells = self.cell_coordinates.reshape(grid * grid, 1, 2)
        sub = self.subanchor_coordinates.reshape(1, self.config.slots_per_cell, 2)
        cell_unit = (cells + 1.0) * 0.5
        cell_column = torch.floor(cell_unit[..., 0] * grid) / grid
        cell_row = torch.floor(cell_unit[..., 1] * grid) / grid
        origin = torch.stack([cell_column, cell_row], dim=-1)
        anchor = origin + sub / grid
        unit = anchor[None] + node_offset / grid
        return unit.clamp(0.0, 1.0) * 2.0 - 1.0

    def _active_nodes(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        active_ids: torch.Tensor,
    ):
        batch, cells, slots, dimensions = hidden.shape
        flat_hidden = hidden.reshape(batch, cells * slots, dimensions)
        flat_positions = positions.reshape(batch, cells * slots, 2)
        gather_hidden = active_ids[:, :, None].expand(-1, -1, dimensions)
        gather_position = active_ids[:, :, None].expand(-1, -1, 2)
        active_hidden = torch.gather(flat_hidden, 1, gather_hidden)
        active_positions = torch.gather(flat_positions, 1, gather_position)
        return active_hidden, active_positions

    def edge_predictions(
        self,
        active_hidden: torch.Tensor,
        active_positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node = self.edge_node(active_hidden)
        left = node[:, :, None, :]
        right = node[:, None, :, :]
        delta = active_positions[:, :, None, :] - active_positions[:, None, :, :]
        spatial = torch.cat(
            [
                delta.abs(),
                torch.linalg.vector_norm(delta, dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        pair = self.edge_pair(left + right + self.edge_space(spatial))
        batch = pair.shape[0]
        return {
            "edge_relation": self.edge_relation(pair),
            "edge_vertical": self.edge_vertical(pair),
            "edge_width": self.edge_width(pair),
            "edge_shape": 0.55
            * torch.tanh(self.edge_shape(pair)).reshape(
                batch,
                pair.shape[1],
                pair.shape[2],
                self.config.edge_shape_points,
                2,
            ),
        }

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        memory, padding, _pool = self.context(batch)
        cells = self._cell_field(memory, padding)
        target = self._target_summary(batch) if use_posterior else None
        prior_mu, prior_logvar, posterior_mu, posterior_logvar = (
            self._latent_parameters(cells, target)
        )
        latent = self._sample(posterior_mu, posterior_logvar)
        output = self._node_predictions(cells, latent)
        positions = self.node_positions(output["node_offset"])
        active_hidden, active_positions = self._active_nodes(
            output["node_hidden"],
            positions,
            batch["active_anchor_ids"],
        )
        output.update(self.edge_predictions(active_hidden, active_positions))
        output.update(
            {
                "node_positions": positions,
                "prior_mu": prior_mu,
                "prior_logvar": prior_logvar,
                "posterior_mu": posterior_mu,
                "posterior_logvar": posterior_logvar,
            }
        )
        return output

    @torch.inference_mode()
    def generate_nodes(
        self,
        batch: dict[str, torch.Tensor],
        *,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        memory, padding, _pool = self.context(batch)
        cells = self._cell_field(memory, padding)
        prior = self.prior(cells)
        mu, logvar = prior.chunk(2, dim=-1)
        latent = self._sample(mu, logvar, temperature=temperature)
        output = self._node_predictions(cells, latent)
        output["node_positions"] = self.node_positions(output["node_offset"])
        return output
