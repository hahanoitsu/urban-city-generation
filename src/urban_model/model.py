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


class SkipUpStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class OutputHeads(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.road = nn.Conv2d(channels, 4, 1)
        self.landuse = nn.Conv2d(channels, 6, 1)
        self.binary = nn.Conv2d(channels, 3, 1)
        self.height = nn.Conv2d(channels, 1, 1)
        self.centerline = nn.Conv2d(channels, 3, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "road_logits": self.road(x),
            "landuse_logits": self.landuse(x),
            "binary_logits": self.binary(x),
            "height_logits": self.height(x),
            "centerline_logits": self.centerline(x),
        }


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
        self.heads = OutputHeads(channels[0])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.down:
            x = stage(x)
        return self.bottleneck(x)

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        x = latent
        for stage in self.up:
            x = stage(x)
        return self.heads(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(x))


class ExtensionUNet(nn.Module):
    """U-Net used only for map extension.

    The hidden half of the input is zero, so skip connections preserve precise seed and
    boundary-guide information without leaking the target continuation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = [config.base_channels * value for value in config.channel_multipliers]
        self.stem = ResidualBlock(config.input_channels, channels[0], config.dropout)
        self.down = nn.ModuleList(
            DownStage(channels[index], channels[index + 1], config.dropout)
            for index in range(len(channels) - 1)
        )
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1], config.dropout),
            ResidualBlock(channels[-1], channels[-1], config.dropout),
        )
        self.up = nn.ModuleList(
            SkipUpStage(
                channels[index],
                channels[index - 1],
                channels[index - 1],
                config.dropout,
            )
            for index in range(len(channels) - 1, 0, -1)
        )
        self.refine = ResidualBlock(channels[0], channels[0], config.dropout)
        self.heads = OutputHeads(channels[0])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        skips = [self.stem(x)]
        current = skips[0]
        for stage in self.down:
            current = stage(current)
            skips.append(current)
        current = self.bottleneck(current)
        for stage, skip in zip(self.up, reversed(skips[:-1]), strict=True):
            current = stage(current, skip)
        return self.heads(self.refine(current))


def build_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "autoencoder":
        return ReconstructionAutoencoder(config)
    if config.architecture == "extension_unet":
        return ExtensionUNet(config)
    raise ValueError(f"Unknown model architecture: {config.architecture}")
