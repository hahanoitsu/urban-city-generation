from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from urban_ai.codec import CommandCodecConfig, OP_COUNT


@dataclass(frozen=True)
class ContextGraphModelConfig:
    codec: CommandCodecConfig
    context_dimensions: int
    port_dimensions: int
    relation_count: int
    model_dimensions: int = 256
    attention_heads: int = 8
    context_layers: int = 4
    decoder_layers: int = 6
    feedforward_dimensions: int = 1024
    maximum_sequence_length: int = 768
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["codec"] = self.codec.to_dict()
        return value


class ContextGraphEncoder(nn.Module):
    def __init__(self, config: ContextGraphModelConfig) -> None:
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

    def forward(self, values: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        hidden = self.input(values)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            neighbours = [
                torch.matmul(relations[:, index], hidden)
                for index in range(relations.shape[1])
            ]
            update = layer(torch.cat([hidden, *neighbours], dim=-1))
            hidden = norm(hidden + update)
        return hidden


class ContextGraphProgramModel(nn.Module):
    def __init__(self, config: ContextGraphModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dimensions
        codec = config.codec
        self.context = ContextGraphEncoder(config)
        self.port = nn.Sequential(
            nn.Linear(config.port_dimensions, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.embeddings = nn.ModuleDict(
            {
                "op": nn.Embedding(OP_COUNT, d, padding_idx=0),
                "x": nn.Embedding(codec.program.coordinate_bins + 1, d, padding_idx=0),
                "y": nn.Embedding(codec.program.coordinate_bins + 1, d, padding_idx=0),
                "id1": nn.Embedding(codec.maximum_nodes + 1, d, padding_idx=0),
                "id2": nn.Embedding(codec.maximum_nodes + 1, d, padding_idx=0),
                "mode": nn.Embedding(3, d, padding_idx=0),
                "class": nn.Embedding(8, d, padding_idx=0),
                "width": nn.Embedding(codec.maximum_width_bin + 1, d, padding_idx=0),
                "vertical": nn.Embedding(5, d, padding_idx=0),
                "layer": nn.Embedding(codec.layer_count + 1, d, padding_idx=0),
            }
        )
        self.position = nn.Embedding(config.maximum_sequence_length, d)
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimensions,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.decoder_layers)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict(
            {
                "op": nn.Linear(d, OP_COUNT),
                "x": nn.Linear(d, codec.program.coordinate_bins),
                "y": nn.Linear(d, codec.program.coordinate_bins),
                "id1": nn.Linear(d, codec.maximum_nodes),
                "id2": nn.Linear(d, codec.maximum_nodes),
                "mode": nn.Linear(d, 2),
                "class": nn.Linear(d, 7),
                "width": nn.Linear(d, codec.maximum_width_bin),
                "vertical": nn.Linear(d, 4),
                "layer": nn.Linear(d, codec.layer_count),
            }
        )

    def forward(
        self,
        commands: dict[str, torch.Tensor],
        context_values: torch.Tensor,
        relations: torch.Tensor,
        ports: torch.Tensor,
        port_padding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        op = commands["op"]
        batch, length = op.shape
        context = self.context(context_values, relations)
        port_memory = self.port(ports)
        memory = torch.cat([context, port_memory], dim=1)
        context_padding = torch.zeros(
            (batch, context.shape[1]),
            dtype=torch.bool,
            device=op.device,
        )
        memory_padding = torch.cat([context_padding, port_padding], dim=1)
        hidden = sum(self.embeddings[name](commands[name]) for name in self.embeddings)
        positions = torch.arange(length, device=op.device).unsqueeze(0).expand(batch, length)
        hidden = hidden + self.position(positions)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=op.device),
            diagonal=1,
        )
        hidden = self.decoder(
            hidden,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=op.eq(0),
            memory_key_padding_mask=memory_padding,
        )
        hidden = self.norm(hidden)
        return {name: head(hidden) for name, head in self.heads.items()}
