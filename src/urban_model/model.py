from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from .config import LayeredDiffusionConfig
from .data import MODEL_CHANNELS


def _require_diffusers():
    try:
        from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
    except ImportError as exc:
        raise RuntimeError(
            "Install the project with the 'diffusion' extra to train or sample the layered model"
        ) from exc
    return UNet2DModel, DDPMScheduler, DDIMScheduler


def _block_types(config: LayeredDiffusionConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    down = tuple(
        "AttnDownBlock2D" if attention else "DownBlock2D"
        for attention in config.attention_levels
    )
    up = tuple(
        "AttnUpBlock2D" if attention else "UpBlock2D"
        for attention in reversed(config.attention_levels)
    )
    return down, up


def build_model(config: LayeredDiffusionConfig) -> nn.Module:
    UNet2DModel, _, _ = _require_diffusers()
    down, up = _block_types(config)
    return UNet2DModel(
        sample_size=config.resolution,
        in_channels=MODEL_CHANNELS,
        out_channels=MODEL_CHANNELS,
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=down,
        up_block_types=up,
        norm_num_groups=config.norm_num_groups,
        add_attention=True,
    )


def build_noise_scheduler(config: LayeredDiffusionConfig):
    _, DDPMScheduler, _ = _require_diffusers()
    return DDPMScheduler(
        num_train_timesteps=config.diffusion_steps,
        beta_schedule=config.beta_schedule,
        prediction_type="epsilon",
        clip_sample=True,
    )


def build_inference_scheduler(config: LayeredDiffusionConfig, noise_scheduler):
    _, _, DDIMScheduler = _require_diffusers()
    return DDIMScheduler.from_config(noise_scheduler.config)


def autocast_context(config: LayeredDiffusionConfig, device: torch.device):
    if device.type != "cuda" or config.precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def weighted_diffusion_loss(
    prediction: torch.Tensor,
    target_noise: torch.Tensor,
    valid_mask: torch.Tensor,
    channel_weights: tuple[float, ...],
) -> torch.Tensor:
    weights = prediction.new_tensor(channel_weights).reshape(1, -1, 1, 1)
    mask = valid_mask.to(dtype=prediction.dtype)
    if mask.shape[1] == 1:
        mask = mask.expand(-1, prediction.shape[1], -1, -1)
    if mask.shape != prediction.shape:
        raise ValueError(
            f"Supervision mask shape {tuple(mask.shape)} does not match prediction "
            f"shape {tuple(prediction.shape)}"
        )
    weighted_mask = weights * mask
    squared = (prediction - target_noise).square() * weighted_mask
    return squared.sum() / weighted_mask.sum().clamp_min(1.0)


class ModelEMA:
    """EMA with warm-up so early checkpoints are not mostly random weights."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        warm_decay = (1.0 + self.updates) / (10.0 + self.updates)
        decay = min(self.decay, warm_decay)
        for name, value in model.state_dict().items():
            source = value.detach()
            if source.is_floating_point():
                self.shadow[name].mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                self.shadow[name].copy_(source)

    def state_dict(self, *, cpu: bool = False) -> dict[str, Any]:
        values = self.shadow
        if cpu:
            values = {name: value.detach().cpu() for name, value in values.items()}
        return {"updates": self.updates, "shadow": values}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.updates = int(state.get("updates", 0))
        values = state.get("shadow", state)
        for name, value in values.items():
            self.shadow[name].copy_(value.to(self.shadow[name].device))

    def copy_model(self, config: LayeredDiffusionConfig, device: torch.device) -> nn.Module:
        model = build_model(config).to(device)
        model.load_state_dict(self.shadow)
        model.eval()
        return model


def _initial_noise(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


@torch.inference_mode()
def sample_model(
    model: nn.Module,
    config: LayeredDiffusionConfig,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    noise_scheduler = build_noise_scheduler(config)
    scheduler = build_inference_scheduler(config, noise_scheduler)
    scheduler.set_timesteps(config.inference_steps, device=device)
    sample = _initial_noise(
        (batch_size, MODEL_CHANNELS, *config.resolution),
        seed,
        device,
    )
    model.eval()
    for timestep in scheduler.timesteps:
        with autocast_context(config, device):
            prediction = model(sample, timestep).sample
        sample = scheduler.step(
            prediction.float(),
            timestep,
            sample,
            eta=0.0,
        ).prev_sample
    return sample.clamp(-1.0, 1.0)
