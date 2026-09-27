from __future__ import annotations

import torch
from torch import nn


def _masked_mean(values: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    valid = (~padding).to(values.dtype)
    return (
        values * valid[:, :, None]
    ).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)


def _decoder(
    dimensions: int,
    heads: int,
    feedforward: int,
    dropout: float,
    layers: int,
):
    layer = nn.TransformerDecoderLayer(
        d_model=dimensions,
        nhead=heads,
        dim_feedforward=feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerDecoder(
        layer,
        num_layers=layers,
    )


class SetResampler(nn.Module):
    def __init__(
        self,
        dimensions: int,
        *,
        queries: int,
        heads: int,
        feedforward: int,
        dropout: float,
        layers: int,
    ) -> None:
        super().__init__()
        self.queries = nn.Embedding(queries, dimensions)
        self.decoder = _decoder(
            dimensions,
            heads,
            feedforward,
            dropout,
            layers,
        )
        self.norm = nn.LayerNorm(dimensions)

    def forward(
        self,
        values: torch.Tensor,
        padding: torch.Tensor,
    ) -> torch.Tensor:
        batch = values.shape[0]
        queries = self.queries.weight[None].expand(
            batch,
            -1,
            -1,
        )
        safe_values = values
        safe_padding = padding
        empty = padding.all(dim=1)
        if bool(empty.any()):
            safe_values = values.clone()
            safe_padding = padding.clone()
            safe_values[empty, 0] = 0.0
            safe_padding[empty, 0] = False
        return self.norm(
            self.decoder(
                queries,
                safe_values,
                memory_key_padding_mask=safe_padding,
            )
        )


class SpatialContextEncoderV2(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        d = config.model_dimensions
        heads = config.heads
        feedforward = config.feedforward_dimensions
        dropout = config.dropout

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

        self.cell_resampler = SetResampler(
            d,
            queries=32,
            heads=heads,
            feedforward=feedforward,
            dropout=dropout,
            layers=2,
        )
        self.line_resampler = SetResampler(
            d,
            queries=48,
            heads=heads,
            feedforward=feedforward,
            dropout=dropout,
            layers=2,
        )
        self.port_resampler = SetResampler(
            d,
            queries=16,
            heads=heads,
            feedforward=feedforward,
            dropout=dropout,
            layers=2,
        )
        self.global_fusion = nn.Sequential(
            nn.Linear(d * 5, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, d),
            nn.LayerNorm(d),
        )
        self.memory_norm = nn.LayerNorm(d)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cells = (
            self.cell(batch["context_cells"])
            + self.kind.weight[0][None, None]
        )
        cell_padding = torch.zeros(
            cells.shape[:2],
            dtype=torch.bool,
            device=cells.device,
        )

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
            + self.transport_class(
                batch["context_line_class"]
            )
            + self.vertical(
                batch["context_line_vertical"]
            )
            + self.kind.weight[1][None, None]
        )
        line_padding = batch["context_line_padding"]

        ports = (
            self.port(batch["ports"])
            + self.mode(batch["port_mode"])
            + self.transport_class(batch["port_class"])
            + self.vertical(batch["port_vertical"])
            + self.kind.weight[2][None, None]
        )
        port_padding = batch["port_padding"]

        style = (
            self.style(batch["style"])
            + self.kind.weight[3][None]
        )
        controls = (
            self.controls(batch["controls"])
            + self.kind.weight[4][None]
        )

        cell_memory = self.cell_resampler(
            cells,
            cell_padding,
        )
        line_memory = self.line_resampler(
            lines,
            line_padding,
        )
        port_memory = self.port_resampler(
            ports,
            port_padding,
        )

        memory = self.memory_norm(
            torch.cat(
                [
                    cell_memory,
                    line_memory,
                    port_memory,
                    style[:, None],
                    controls[:, None],
                ],
                dim=1,
            )
        )
        padding = torch.zeros(
            memory.shape[:2],
            dtype=torch.bool,
            device=memory.device,
        )

        pool = self.global_fusion(
            torch.cat(
                [
                    _masked_mean(cells, cell_padding),
                    _masked_mean(lines, line_padding),
                    _masked_mean(ports, port_padding),
                    style,
                    controls,
                ],
                dim=-1,
            )
        )
        return memory, padding, pool
