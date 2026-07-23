from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .codec import CommandCodecConfig, OP_COUNT


@dataclass(frozen=True)
class GraphTransformerConfig:
    codec: CommandCodecConfig
    style_dimensions: int
    maximum_sequence_length: int = 1024
    model_dimensions: int = 384
    attention_heads: int = 6
    layers: int = 8
    feedforward_dimensions: int = 1536
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["codec"] = self.codec.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GraphTransformerConfig":
        copied = dict(value)
        copied["codec"] = CommandCodecConfig.from_dict(copied["codec"])
        return cls(**copied)


class GraphProgramTransformer(nn.Module):
    def __init__(self, config: GraphTransformerConfig) -> None:
        super().__init__()
        self.config = config
        codec = config.codec
        dimensions = config.model_dimensions
        self.embeddings = nn.ModuleDict(
            {
                "op": nn.Embedding(OP_COUNT, dimensions, padding_idx=0),
                "x": nn.Embedding(codec.program.coordinate_bins + 1, dimensions, padding_idx=0),
                "y": nn.Embedding(codec.program.coordinate_bins + 1, dimensions, padding_idx=0),
                "id1": nn.Embedding(codec.maximum_nodes + 1, dimensions, padding_idx=0),
                "id2": nn.Embedding(codec.maximum_nodes + 1, dimensions, padding_idx=0),
                "mode": nn.Embedding(3, dimensions, padding_idx=0),
                "class": nn.Embedding(8, dimensions, padding_idx=0),
                "width": nn.Embedding(codec.maximum_width_bin + 1, dimensions, padding_idx=0),
                "vertical": nn.Embedding(5, dimensions, padding_idx=0),
                "layer": nn.Embedding(codec.layer_count + 1, dimensions, padding_idx=0),
            }
        )
        self.position_embedding = nn.Embedding(config.maximum_sequence_length, dimensions)
        self.style_projection = nn.Sequential(
            nn.Linear(config.style_dimensions, dimensions),
            nn.GELU(),
            nn.Linear(dimensions, dimensions),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dimensions,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimensions,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.normalization = nn.LayerNorm(dimensions)
        self.heads = nn.ModuleDict(
            {
                "op": nn.Linear(dimensions, OP_COUNT),
                "x": nn.Linear(dimensions, codec.program.coordinate_bins),
                "y": nn.Linear(dimensions, codec.program.coordinate_bins),
                "id1": nn.Linear(dimensions, codec.maximum_nodes),
                "id2": nn.Linear(dimensions, codec.maximum_nodes),
                "mode": nn.Linear(dimensions, 2),
                "class": nn.Linear(dimensions, 7),
                "width": nn.Linear(dimensions, codec.maximum_width_bin),
                "vertical": nn.Linear(dimensions, 4),
                "layer": nn.Linear(dimensions, codec.layer_count),
            }
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, commands: dict[str, torch.Tensor], style: torch.Tensor) -> dict[str, torch.Tensor]:
        op = commands["op"]
        batch, length = op.shape
        if length > self.config.maximum_sequence_length:
            raise ValueError(
                f"Sequence length {length} exceeds model maximum "
                f"{self.config.maximum_sequence_length}"
            )
        hidden = sum(self.embeddings[field](commands[field]) for field in self.embeddings)
        positions = torch.arange(length, device=op.device).unsqueeze(0).expand(batch, length)
        hidden = hidden + self.position_embedding(positions)
        hidden = hidden + self.style_projection(style).unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=op.device), diagonal=1
        )
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=op.eq(0),
        )
        hidden = self.normalization(hidden)
        return {name: head(hidden) for name, head in self.heads.items()}
