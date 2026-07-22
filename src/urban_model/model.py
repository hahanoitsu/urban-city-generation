from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class DownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
        self.block = ResidualBlock(out_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.block = ResidualBlock(out_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(self.reduce(x))


class ReconstructionAutoencoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = [config.base_channels * value for value in config.channel_multipliers]
        self.stem = ResidualBlock(config.input_channels, channels[0], config.dropout)
        self.down = nn.ModuleList(
            DownStage(channels[index], channels[index + 1], config.dropout)
            for index in range(len(channels) - 1)
        )
        self.bottleneck = ResidualBlock(channels[-1], channels[-1], config.dropout)
        self.up = nn.ModuleList(
            UpStage(channels[index], channels[index - 1], config.dropout)
            for index in range(len(channels) - 1, 0, -1)
        )
        output_channels = channels[0]
        self.road_head = nn.Conv2d(output_channels, 4, 1)
        self.landuse_head = nn.Conv2d(output_channels, 6, 1)
        self.binary_head = nn.Conv2d(output_channels, 3, 1)
        self.height_head = nn.Conv2d(output_channels, 1, 1)
        self.centerline_head = nn.Conv2d(output_channels, 3, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.down:
            x = stage(x)
        return self.bottleneck(x)

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        x = latent
        for stage in self.up:
            x = stage(x)
        return {
            "road_logits": self.road_head(x),
            "landuse_logits": self.landuse_head(x),
            "binary_logits": self.binary_head(x),
            "height_logits": self.height_head(x),
            "centerline_logits": self.centerline_head(x),
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(x))
