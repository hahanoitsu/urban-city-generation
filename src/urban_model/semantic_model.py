from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .runtime import worker_count
from .semantic_config import (
    SEMANTIC_NAMES,
    SEMANTIC_PALETTE,
    SemanticDiffusionConfig,
    SemanticTask,
)
from .semantic_data import model_space_to_semantic


def _require_diffusers():
    try:
        from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
    except ImportError as exc:
        raise RuntimeError(
            "Install the project with the 'diffusion' extra to use semantic diffusion"
        ) from exc
    return UNet2DModel, DDPMScheduler, DDIMScheduler


def _block_types(config: SemanticDiffusionConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    down = tuple(
        "AttnDownBlock2D" if attention else "DownBlock2D"
        for attention in config.attention_levels
    )
    up = tuple(
        "AttnUpBlock2D" if attention else "UpBlock2D"
        for attention in reversed(config.attention_levels)
    )
    return down, up


def semantic_input_channels(task: SemanticTask) -> int:
    return len(SEMANTIC_NAMES) if task == "block" else len(SEMANTIC_NAMES) + 1 + 3


def build_semantic_model(config: SemanticDiffusionConfig) -> nn.Module:
    UNet2DModel, _, _ = _require_diffusers()
    down_block_types, up_block_types = _block_types(config)
    return UNet2DModel(
        sample_size=config.resolution,
        in_channels=semantic_input_channels(config.task),
        out_channels=len(SEMANTIC_NAMES),
        layers_per_block=config.layers_per_block,
        block_out_channels=config.block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        norm_num_groups=config.norm_num_groups,
        add_attention=True,
    )


def build_noise_scheduler(config: SemanticDiffusionConfig):
    _, DDPMScheduler, _ = _require_diffusers()
    return DDPMScheduler(
        num_train_timesteps=config.diffusion_steps,
        beta_schedule=config.beta_schedule,
        prediction_type="epsilon",
        clip_sample=True,
    )


def build_inference_scheduler(config: SemanticDiffusionConfig, noise_scheduler):
    _, _, DDIMScheduler = _require_diffusers()
    return DDIMScheduler.from_config(noise_scheduler.config)


def diffusion_model_input(
    noisy: torch.Tensor,
    task: SemanticTask,
    *,
    known_mask: torch.Tensor | None = None,
    road_guide: torch.Tensor | None = None,
) -> torch.Tensor:
    if task == "block":
        return noisy
    if known_mask is None or road_guide is None:
        raise ValueError("Outpainting requires a known-region mask and road guide")
    return torch.cat([noisy, known_mask, road_guide], dim=1)


def diffusion_loss(
    prediction: torch.Tensor,
    target_noise: torch.Tensor,
    *,
    known_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    squared = (prediction - target_noise).square()
    if known_mask is None:
        return squared.mean()
    unknown = 1.0 - known_mask
    return (squared * unknown).sum() / (
        unknown.sum() * prediction.shape[1]
    ).clamp_min(1.0)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            source = value.detach()
            if source.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(source)

    def state_dict(self, *, cpu: bool = False) -> dict[str, torch.Tensor]:
        if cpu:
            return {name: value.detach().cpu() for name, value in self.shadow.items()}
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, value in state.items():
            self.shadow[name].copy_(value.to(self.shadow[name].device))


def _copy_ema_model(config: SemanticDiffusionConfig, ema: ModelEMA, device: torch.device):
    model = build_semantic_model(config).to(device)
    model.load_state_dict(ema.state_dict())
    model.eval()
    return model


def _initialise_outpainting_from_block(
    model: nn.Module,
    checkpoint_path: Path,
) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("ema") or checkpoint["model"]
    target = model.state_dict()
    copied = 0
    adapted = 0

    for name, value in source.items():
        if name not in target:
            continue
        if target[name].shape == value.shape:
            target[name].copy_(value)
            copied += 1
            continue
        if name == "conv_in.weight" and value.ndim == 4:
            target[name].zero_()
            channels = min(value.shape[1], target[name].shape[1])
            target[name][:, :channels].copy_(value[:, :channels])
            adapted += 1
    model.load_state_dict(target)
    return {"copied_tensors": copied, "adapted_tensors": adapted}


def _loader(
    dataset: Dataset,
    config: SemanticDiffusionConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    workers = worker_count(config.num_workers)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: AdamW,
    epoch: int,
    best_validation_loss: float,
    config: SemanticDiffusionConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict(cpu=True),
            "optimizer": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "semantic_names": SEMANTIC_NAMES,
            "config": asdict(config),
        },
        path,
    )


def render_semantic(classes: torch.Tensor) -> Image.Image:
    array = classes.detach().cpu().numpy().astype(np.int64)
    return Image.fromarray(SEMANTIC_PALETTE[array])


def _initial_noise(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


@torch.inference_mode()
def sample_semantic(
    model: nn.Module,
    config: SemanticDiffusionConfig,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    known_x0: torch.Tensor | None = None,
    known_mask: torch.Tensor | None = None,
    road_guide: torch.Tensor | None = None,
) -> torch.Tensor:
    noise_scheduler = build_noise_scheduler(config)
    inference_scheduler = build_inference_scheduler(config, noise_scheduler)
    inference_scheduler.set_timesteps(config.inference_steps, device=device)
    shape = (batch_size, len(SEMANTIC_NAMES), *config.resolution)
    sample = _initial_noise(shape, seed, device)

    if config.task == "outpaint":
        if known_x0 is None or known_mask is None or road_guide is None:
            raise ValueError("Outpainting sampling requires seed, mask and road guide")
        known_x0 = known_x0.to(device)
        known_mask = known_mask.to(device)
        road_guide = road_guide.to(device)
        fixed_noise = _initial_noise(shape, seed + 1, device)

    model.eval()
    for timestep in inference_scheduler.timesteps:
        timestep_value = int(timestep.item())
        batch_timestep = torch.full(
            (batch_size,),
            timestep_value,
            device=device,
            dtype=torch.long,
        )
        if config.task == "outpaint":
            known_noisy = noise_scheduler.add_noise(known_x0, fixed_noise, batch_timestep)
            sample = known_noisy * known_mask + sample * (1.0 - known_mask)
            model_input = diffusion_model_input(
                sample,
                config.task,
                known_mask=known_mask,
                road_guide=road_guide,
            )
        else:
            model_input = sample
        prediction = model(model_input, timestep).sample
        sample = inference_scheduler.step(
            prediction,
            timestep,
            sample,
            eta=0.0,
        ).prev_sample

    if config.task == "outpaint":
        sample = known_x0 * known_mask + sample * (1.0 - known_mask)
    return sample


def _save_block_preview(
    path: Path,
    model: nn.Module,
    config: SemanticDiffusionConfig,
    device: torch.device,
    seed: int,
) -> None:
    generated = sample_semantic(
        model,
        config,
        batch_size=4,
        device=device,
        seed=seed,
    )
    classes = model_space_to_semantic(generated)
    height, width = classes.shape[-2:]
    canvas = Image.new("RGB", (width * 2, height * 2), "white")
    for index in range(4):
        canvas.paste(render_semantic(classes[index]), ((index % 2) * width, (index // 2) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def _save_outpaint_preview(
    path: Path,
    model: nn.Module,
    config: SemanticDiffusionConfig,
    batch: dict[str, Any],
    device: torch.device,
    seed: int,
) -> None:
    count = min(4, batch["x0"].shape[0])
    target = batch["x0"][:count].to(device)
    mask = batch["known_mask"][:count].to(device)
    guide = batch["road_guide"][:count].to(device)
    known = target * mask
    generated = sample_semantic(
        model,
        config,
        batch_size=count,
        device=device,
        seed=seed,
        known_x0=known,
        known_mask=mask,
        road_guide=guide,
    )
    target_classes = model_space_to_semantic(target)
    generated_classes = model_space_to_semantic(generated)
    height, width = target_classes.shape[-2:]
    label_height = 22
    canvas = Image.new("RGB", (width * 2, count * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        top = index * (height + label_height)
        canvas.paste(render_semantic(target_classes[index]), (0, top))
        canvas.paste(render_semantic(generated_classes[index]), (width, top))
        draw.text((4, top + height + 4), "real pair", fill="black")
        draw.text((width + 4, top + height + 4), "sampled continuation", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
