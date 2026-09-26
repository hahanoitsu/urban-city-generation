from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class StructuredCityConfig:
    context_dimensions: int
    relation_count: int
    port_dimensions: int
    node_slots: int = 384
    edge_slots: int = 640
    building_slots: int = 384
    area_slots: int = 96
    edge_shape_points: int = 16
    building_points: int = 24
    area_points: int = 48
    area_classes: int = 6
    model_dimensions: int = 256
    attention_heads: int = 8
    context_layers: int = 4
    transport_layers: int = 6
    building_layers: int = 4
    area_layers: int = 4
    feedforward_dimensions: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StructuredCityConfig":
        return cls(**value)


class TimeEmbedding(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.projection = nn.Sequential(
            nn.Linear(dimensions, dimensions),
            nn.SiLU(),
            nn.Linear(dimensions, dimensions),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimensions // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=time.device, dtype=time.dtype)
        )
        phase = time[:, None] * frequencies[None]
        values = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if values.shape[-1] < self.dimensions:
            values = torch.nn.functional.pad(values, (0, self.dimensions - values.shape[-1]))
        return self.projection(values)


class RelationContextEncoder(nn.Module):
    def __init__(self, config: StructuredCityConfig) -> None:
        super().__init__()
        d = config.model_dimensions
        self.input = nn.Linear(config.context_dimensions, d)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d * (config.relation_count + 1), config.feedforward_dimensions),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.feedforward_dimensions, d),
                )
                for _ in range(config.context_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(config.context_layers)])

    def forward(
        self,
        values: torch.Tensor,
        relations: torch.Tensor,
        padding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input(values)
        hidden = hidden.masked_fill(padding[:, :, None], 0.0)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            neighbours = [
                torch.matmul(relations[:, index], hidden)
                for index in range(relations.shape[1])
            ]
            hidden = norm(hidden + layer(torch.cat([hidden, *neighbours], dim=-1)))
            hidden = hidden.masked_fill(padding[:, :, None], 0.0)
        return hidden


class SlotDecoder(nn.Module):
    def __init__(
        self,
        *,
        slots: int,
        continuous_dimensions: int,
        category_sizes: tuple[int, ...],
        config: StructuredCityConfig,
        layers: int,
    ) -> None:
        super().__init__()
        d = config.model_dimensions
        self.continuous = nn.Linear(continuous_dimensions, d)
        self.categories = nn.ModuleList(
            [nn.Embedding(size + 1, d) for size in category_sizes]
        )
        self.slots = nn.Embedding(slots, d)
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimensions,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        continuous: torch.Tensor,
        categories: tuple[torch.Tensor, ...],
        memory: torch.Tensor,
        memory_padding: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        batch, slots, _dimensions = continuous.shape
        hidden = self.continuous(continuous)
        for embedding, values in zip(self.categories, categories, strict=True):
            hidden = hidden + embedding(values)
        indexes = torch.arange(slots, device=continuous.device)
        hidden = hidden + self.slots(indexes)[None] + time[:, None]
        return self.norm(
            self.decoder(
                hidden,
                memory,
                memory_key_padding_mask=memory_padding,
            )
        )


class StructuredCityDenoiser(nn.Module):
    def __init__(self, config: StructuredCityConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dimensions
        self.time = TimeEmbedding(d)
        self.context = RelationContextEncoder(config)
        self.port = nn.Sequential(
            nn.Linear(config.port_dimensions, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.nodes = SlotDecoder(
            slots=config.node_slots,
            continuous_dimensions=3,
            category_sizes=(2,),
            config=config,
            layers=config.transport_layers,
        )
        self.node_presence = nn.Linear(d, 2)
        self.node_position = nn.Linear(d, 3)

        self.edges = SlotDecoder(
            slots=config.edge_slots,
            continuous_dimensions=1 + config.edge_shape_points * 3,
            category_sizes=(2, 2, 7, 4),
            config=config,
            layers=config.transport_layers,
        )
        self.edge_presence = nn.Linear(d, 2)
        self.edge_mode = nn.Linear(d, 2)
        self.edge_class = nn.Linear(d, 7)
        self.edge_vertical = nn.Linear(d, 4)
        self.edge_width = nn.Linear(d, 1)
        self.edge_shape = nn.Linear(d, config.edge_shape_points * 3)
        self.edge_from = nn.Linear(d, d)
        self.edge_to = nn.Linear(d, d)
        self.node_key = nn.Linear(d, d)

        self.buildings = SlotDecoder(
            slots=config.building_slots,
            continuous_dimensions=config.building_points * 2 + 2,
            category_sizes=(2,),
            config=config,
            layers=config.building_layers,
        )
        self.building_presence = nn.Linear(d, 2)
        self.building_shape = nn.Linear(d, config.building_points * 2)
        self.building_height = nn.Linear(d, 1)
        self.building_base_z = nn.Linear(d, 1)

        self.areas = SlotDecoder(
            slots=config.area_slots,
            continuous_dimensions=config.area_points * 2,
            category_sizes=(2, config.area_classes),
            config=config,
            layers=config.area_layers,
        )
        self.area_presence = nn.Linear(d, 2)
        self.area_kind = nn.Linear(d, config.area_classes)
        self.area_shape = nn.Linear(d, config.area_points * 2)

    def forward(
        self,
        scene: dict[str, torch.Tensor],
        context_values: torch.Tensor,
        relations: torch.Tensor,
        context_padding: torch.Tensor,
        ports: torch.Tensor,
        port_padding: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        time = self.time(diffusion_time)
        context = self.context(context_values, relations, context_padding)
        port_memory = self.port(ports)
        port_memory = port_memory.masked_fill(port_padding[:, :, None], 0.0)
        context_memory = torch.cat([context, port_memory], dim=1)
        context_padding = torch.cat([context_padding, port_padding], dim=1)

        node_hidden = self.nodes(
            scene["node_position"],
            (scene["node_presence"],),
            context_memory,
            context_padding,
            time,
        )
        node_position = self.node_position(node_hidden)

        edge_continuous = torch.cat(
            [
                scene["edge_width"],
                scene["edge_shape"].flatten(2),
            ],
            dim=-1,
        )
        edge_memory = torch.cat([context_memory, node_hidden], dim=1)
        edge_padding = torch.cat(
            [
                context_padding,
                torch.zeros(
                    node_hidden.shape[:2],
                    dtype=torch.bool,
                    device=node_hidden.device,
                ),
            ],
            dim=1,
        )
        edge_hidden = self.edges(
            edge_continuous,
            (
                scene["edge_presence"],
                scene["edge_mode"],
                scene["edge_class"],
                scene["edge_vertical"],
            ),
            edge_memory,
            edge_padding,
            time,
        )

        node_keys = self.node_key(node_hidden)
        edge_from = torch.einsum(
            "bed,bnd->ben",
            self.edge_from(edge_hidden),
            node_keys,
        ) / math.sqrt(node_keys.shape[-1])
        edge_to = torch.einsum(
            "bed,bnd->ben",
            self.edge_to(edge_hidden),
            node_keys,
        ) / math.sqrt(node_keys.shape[-1])

        building_continuous = torch.cat(
            [
                scene["building_shape"].flatten(2),
                scene["building_height"],
                scene["building_base_z"],
            ],
            dim=-1,
        )
        building_memory = torch.cat([context_memory, node_hidden, edge_hidden], dim=1)
        object_padding = torch.cat(
            [
                context_padding,
                torch.zeros(
                    (node_hidden.shape[0], node_hidden.shape[1] + edge_hidden.shape[1]),
                    dtype=torch.bool,
                    device=node_hidden.device,
                ),
            ],
            dim=1,
        )
        building_hidden = self.buildings(
            building_continuous,
            (scene["building_presence"],),
            building_memory,
            object_padding,
            time,
        )

        area_memory = torch.cat([context_memory, node_hidden, edge_hidden], dim=1)
        area_hidden = self.areas(
            scene["area_shape"].flatten(2),
            (scene["area_presence"], scene["area_kind"]),
            area_memory,
            object_padding,
            time,
        )

        return {
            "node_presence": self.node_presence(node_hidden),
            "node_position": node_position,
            "edge_presence": self.edge_presence(edge_hidden),
            "edge_mode": self.edge_mode(edge_hidden),
            "edge_class": self.edge_class(edge_hidden),
            "edge_vertical": self.edge_vertical(edge_hidden),
            "edge_width": self.edge_width(edge_hidden),
            "edge_shape": self.edge_shape(edge_hidden).reshape(
                edge_hidden.shape[0],
                edge_hidden.shape[1],
                self.config.edge_shape_points,
                3,
            ),
            "edge_from": edge_from,
            "edge_to": edge_to,
            "building_presence": self.building_presence(building_hidden),
            "building_shape": self.building_shape(building_hidden).reshape(
                building_hidden.shape[0],
                building_hidden.shape[1],
                self.config.building_points,
                2,
            ),
            "building_height": self.building_height(building_hidden),
            "building_base_z": self.building_base_z(building_hidden),
            "area_presence": self.area_presence(area_hidden),
            "area_kind": self.area_kind(area_hidden),
            "area_shape": self.area_shape(area_hidden).reshape(
                area_hidden.shape[0],
                area_hidden.shape[1],
                self.config.area_points,
                2,
            ),
        }
