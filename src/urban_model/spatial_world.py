from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class SpatialWorldModelConfig:
    context_dimensions: int
    style_dimensions: int
    max_nodes: int = 256
    max_edges: int = 512
    context_line_points: int = 6
    edge_shape_points: int = 8
    model_dimensions: int = 256
    latent_dimensions: int = 128
    heads: int = 8
    context_layers: int = 4
    target_layers: int = 3
    node_layers: int = 4
    edge_layers: int = 4
    feedforward_dimensions: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpatialWorldModelConfig":
        return cls(**value)


def _encoder(dimensions: int, heads: int, feedforward: int, dropout: float, layers: int):
    layer = nn.TransformerEncoderLayer(
        d_model=dimensions,
        nhead=heads,
        dim_feedforward=feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


def _decoder(dimensions: int, heads: int, feedforward: int, dropout: float, layers: int):
    layer = nn.TransformerDecoderLayer(
        d_model=dimensions,
        nhead=heads,
        dim_feedforward=feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=layers)


def _masked_mean(values: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    valid = (~padding).to(values.dtype)
    return (values * valid[:, :, None]).sum(dim=1) / valid.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0)


class SpatialContextEncoder(nn.Module):
    def __init__(self, config: SpatialWorldModelConfig) -> None:
        super().__init__()
        d = config.model_dimensions
        self.cell = nn.Sequential(
            nn.Linear(config.context_dimensions, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.line = nn.Sequential(
            nn.Linear(config.context_line_points * 2 + 2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.port = nn.Sequential(
            nn.Linear(5, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.style = nn.Sequential(
            nn.Linear(config.style_dimensions, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.controls = nn.Sequential(
            nn.Linear(config.style_dimensions, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.mode = nn.Embedding(2, d)
        self.transport_class = nn.Embedding(8, d)
        self.vertical = nn.Embedding(4, d)
        self.kind = nn.Embedding(5, d)
        self.encoder = _encoder(
            d,
            config.heads,
            config.feedforward_dimensions,
            config.dropout,
            config.context_layers,
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, batch: dict[str, torch.Tensor]):
        cells = self.cell(batch["context_cells"]) + self.kind.weight[0][None, None]

        line_continuous = torch.cat(
            [
                batch["context_line_points"].flatten(2),
                batch["context_line_width"],
                batch["context_line_length"],
            ],
            dim=-1,
        )
        lines = (
            self.line(line_continuous)
            + self.mode(batch["context_line_mode"])
            + self.transport_class(batch["context_line_class"])
            + self.vertical(batch["context_line_vertical"])
            + self.kind.weight[1][None, None]
        )

        ports = (
            self.port(batch["ports"])
            + self.mode(batch["port_mode"])
            + self.transport_class(batch["port_class"])
            + self.vertical(batch["port_vertical"])
            + self.kind.weight[2][None, None]
        )

        style = self.style(batch["style"])[:, None] + self.kind.weight[3][None, None]
        controls = self.controls(batch["controls"])[:, None] + self.kind.weight[4][None, None]

        values = torch.cat([cells, lines, ports, style, controls], dim=1)
        padding = torch.cat(
            [
                torch.zeros(
                    cells.shape[:2],
                    dtype=torch.bool,
                    device=cells.device,
                ),
                batch["context_line_padding"],
                batch["port_padding"],
                torch.zeros(
                    (cells.shape[0], 2),
                    dtype=torch.bool,
                    device=cells.device,
                ),
            ],
            dim=1,
        )
        hidden = self.norm(
            self.encoder(
                values,
                src_key_padding_mask=padding,
            )
        )
        return hidden, padding, _masked_mean(hidden, padding)


class TargetGraphEncoder(nn.Module):
    def __init__(self, config: SpatialWorldModelConfig) -> None:
        super().__init__()
        d = config.model_dimensions
        self.config = config
        self.node = nn.Sequential(
            nn.Linear(3, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.edge = nn.Sequential(
            nn.Linear(4 + 1 + config.edge_shape_points * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.mode = nn.Embedding(2, d)
        self.transport_class = nn.Embedding(8, d)
        self.vertical = nn.Embedding(4, d)
        self.kind = nn.Embedding(2, d)
        self.encoder = _encoder(
            d,
            config.heads,
            config.feedforward_dimensions,
            config.dropout,
            config.target_layers,
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["node_xy"].shape[0]
        node_indexes = torch.arange(self.config.max_nodes, device=batch["node_xy"].device)
        node_padding = node_indexes[None] >= batch["node_count"][:, None]
        node_continuous = torch.cat(
            [
                batch["node_xy"],
                batch["node_boundary"][:, :, None],
            ],
            dim=-1,
        )
        nodes = (
            self.node(node_continuous)
            + self.mode(batch["node_mode"])
            + self.vertical(batch["node_vertical"])
            + self.kind.weight[0][None, None]
        )

        safe_from = batch["edge_from"].clamp(0, self.config.max_nodes - 1)
        safe_to = batch["edge_to"].clamp(0, self.config.max_nodes - 1)
        start = torch.gather(
            batch["node_xy"],
            1,
            safe_from[:, :, None].expand(-1, -1, 2),
        )
        end = torch.gather(
            batch["node_xy"],
            1,
            safe_to[:, :, None].expand(-1, -1, 2),
        )
        edge_continuous = torch.cat(
            [
                start,
                end,
                batch["edge_width"],
                batch["edge_shape"].flatten(2),
            ],
            dim=-1,
        )
        edges = (
            self.edge(edge_continuous)
            + self.mode(batch["edge_mode"])
            + self.transport_class(batch["edge_class"])
            + self.vertical(batch["edge_vertical"])
            + self.kind.weight[1][None, None]
        )
        edge_indexes = torch.arange(self.config.max_edges, device=batch["edge_count"].device)
        edge_padding = edge_indexes[None] >= batch["edge_count"][:, None]

        values = torch.cat([nodes, edges], dim=1)
        padding = torch.cat([node_padding, edge_padding], dim=1)
        hidden = self.norm(
            self.encoder(
                values,
                src_key_padding_mask=padding,
            )
        )
        return _masked_mean(hidden, padding)


class SpatialWorldArchitect(nn.Module):
    def __init__(self, config: SpatialWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dimensions
        z = config.latent_dimensions

        self.context = SpatialContextEncoder(config)
        self.target = TargetGraphEncoder(config)

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

        self.node_queries = nn.Embedding(config.max_nodes, d)
        self.node_decoder = _decoder(
            d,
            config.heads,
            config.feedforward_dimensions,
            config.dropout,
            config.node_layers,
        )
        self.node_norm = nn.LayerNorm(d)
        self.node_count = nn.Linear(d + z, config.max_nodes + 1)
        self.node_xy = nn.Linear(d, 2)
        self.node_mode = nn.Linear(d, 2)
        self.node_vertical = nn.Linear(d, 4)
        self.node_boundary = nn.Linear(d, 1)

        self.edge_queries = nn.Embedding(config.max_edges, d)
        self.edge_decoder = _decoder(
            d,
            config.heads,
            config.feedforward_dimensions,
            config.dropout,
            config.edge_layers,
        )
        self.edge_norm = nn.LayerNorm(d)
        self.edge_count = nn.Linear(d + z, config.max_edges + 1)
        self.edge_mode = nn.Linear(d, 2)
        self.edge_class = nn.Linear(d, 8)
        self.edge_vertical = nn.Linear(d, 4)
        self.edge_width = nn.Linear(d, 1)
        self.edge_shape = nn.Linear(d, config.edge_shape_points * 2)
        self.edge_from = nn.Linear(d, d)
        self.edge_to = nn.Linear(d, d)
        self.node_key = nn.Linear(d, d)

    def latent_parameters(
        self,
        context_pool: torch.Tensor,
        target_pool: torch.Tensor | None,
    ):
        prior = self.prior(context_pool)
        prior_mu, prior_logvar = prior.chunk(2, dim=-1)
        if target_pool is None:
            return prior_mu, prior_logvar, prior_mu, prior_logvar
        posterior = self.posterior(torch.cat([context_pool, target_pool], dim=-1))
        posterior_mu, posterior_logvar = posterior.chunk(2, dim=-1)
        return prior_mu, prior_logvar, posterior_mu, posterior_logvar

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        noise = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar.clamp(-10.0, 10.0)) * noise

    def decode(
        self,
        memory: torch.Tensor,
        memory_padding: torch.Tensor,
        context_pool: torch.Tensor,
        latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = memory.shape[0]
        latent_token = self.latent(latent)[:, None]
        decoder_memory = torch.cat([memory, latent_token], dim=1)
        decoder_padding = torch.cat(
            [
                memory_padding,
                torch.zeros((batch, 1), dtype=torch.bool, device=memory.device),
            ],
            dim=1,
        )

        node_queries = self.node_queries.weight[None].expand(batch, -1, -1)
        node_hidden = self.node_norm(
            self.node_decoder(
                node_queries,
                decoder_memory,
                memory_key_padding_mask=decoder_padding,
            )
        )
        node_xy = torch.tanh(self.node_xy(node_hidden))
        count_input = torch.cat([context_pool, latent], dim=-1)

        edge_memory = torch.cat([decoder_memory, node_hidden], dim=1)
        edge_padding = torch.cat(
            [
                decoder_padding,
                torch.zeros(
                    node_hidden.shape[:2],
                    dtype=torch.bool,
                    device=node_hidden.device,
                ),
            ],
            dim=1,
        )
        edge_queries = self.edge_queries.weight[None].expand(batch, -1, -1)
        edge_hidden = self.edge_norm(
            self.edge_decoder(
                edge_queries,
                edge_memory,
                memory_key_padding_mask=edge_padding,
            )
        )

        node_keys = self.node_key(node_hidden)
        scale = math.sqrt(node_keys.shape[-1])
        from_logits = torch.einsum(
            "bed,bnd->ben",
            self.edge_from(edge_hidden),
            node_keys,
        ) / scale
        to_logits = torch.einsum(
            "bed,bnd->ben",
            self.edge_to(edge_hidden),
            node_keys,
        ) / scale

        return {
            "node_count": self.node_count(count_input),
            "node_xy": node_xy,
            "node_mode": self.node_mode(node_hidden),
            "node_vertical": self.node_vertical(node_hidden),
            "node_boundary": self.node_boundary(node_hidden).squeeze(-1),
            "edge_count": self.edge_count(count_input),
            "edge_from": from_logits,
            "edge_to": to_logits,
            "edge_mode": self.edge_mode(edge_hidden),
            "edge_class": self.edge_class(edge_hidden),
            "edge_vertical": self.edge_vertical(edge_hidden),
            "edge_width": self.edge_width(edge_hidden),
            "edge_shape": 0.5
            * torch.tanh(self.edge_shape(edge_hidden)).reshape(
                batch,
                self.config.max_edges,
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
        memory, memory_padding, context_pool = self.context(batch)
        target_pool = self.target(batch) if use_posterior else None
        prior_mu, prior_logvar, posterior_mu, posterior_logvar = self.latent_parameters(
            context_pool,
            target_pool,
        )
        latent = self._sample(posterior_mu, posterior_logvar)
        output = self.decode(memory, memory_padding, context_pool, latent)
        output.update(
            {
                "prior_mu": prior_mu,
                "prior_logvar": prior_logvar,
                "posterior_mu": posterior_mu,
                "posterior_logvar": posterior_logvar,
                "latent": latent,
            }
        )
        return output

    @torch.inference_mode()
    def generate(
        self,
        batch: dict[str, torch.Tensor],
        *,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        memory, memory_padding, context_pool = self.context(batch)
        prior = self.prior(context_pool)
        mu, logvar = prior.chunk(2, dim=-1)
        noise = torch.randn_like(mu)
        latent = mu + torch.exp(0.5 * logvar.clamp(-10.0, 10.0)) * noise * temperature
        return self.decode(memory, memory_padding, context_pool, latent)
