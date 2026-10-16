from __future__ import annotations

import math
import random

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F

from urban_layers.data import CHANNELS


def coefficients(time):
    angle = time.clamp(0.0, 1.0) * math.pi / 2.0
    return torch.cos(angle), torch.sin(angle)


def time_features(time, size):
    half = size // 2
    scales = torch.exp(-math.log(10000.0) * torch.arange(half, device=time.device) / max(half - 1, 1))
    angles = time[:, None] * 1000.0 * scales[None]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class Residual(nn.Module):
    def __init__(self, channels_in, channels_out, condition):
        super().__init__()
        self.norm_in = nn.GroupNorm(8, channels_in)
        self.conv_in = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.film = nn.Linear(condition, 2 * channels_out)
        self.norm_out = nn.GroupNorm(8, channels_out)
        self.conv_out = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.skip = nn.Conv2d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()

    def forward(self, x, condition):
        h = self.conv_in(F.silu(self.norm_in(x)))
        scale, shift = self.film(condition)[:, :, None, None].chunk(2, dim=1)
        h = self.norm_out(h) * (1 + scale) + shift
        h = self.conv_out(F.silu(h))
        return self.skip(x) + h


class PlanNet(nn.Module):
    def __init__(self, cities, channels=len(CHANNELS), width=64, condition=256):
        super().__init__()
        self.config = {"cities": cities, "channels": channels, "width": width, "condition": condition}
        self.cities = cities
        self.time = nn.Sequential(nn.Linear(condition, condition), nn.SiLU(), nn.Linear(condition, condition))
        self.city = nn.Linear(cities, condition)
        self.controls = nn.Sequential(nn.Linear(channels + 1, condition), nn.SiLU(), nn.Linear(condition, condition))
        widths = [width, width * 2, width * 4]
        self.stem = nn.Conv2d(channels, width, 3, padding=1)
        self.down = nn.ModuleList()
        current = width
        for level, size in enumerate(widths):
            self.down.append(nn.ModuleList([Residual(current, size, condition), Residual(size, size, condition)]))
            current = size
        self.pool = nn.ModuleList([nn.Conv2d(size, size, 3, stride=2, padding=1) for size in widths[:-1]])
        self.middle = nn.ModuleList([Residual(current, current, condition), Residual(current, current, condition)])
        self.up = nn.ModuleList()
        for size in reversed(widths):
            self.up.append(nn.ModuleList([Residual(current + size, size, condition), Residual(size, size, condition)]))
            current = size
        self.lift = nn.ModuleList([nn.Conv2d(size, size, 3, padding=1) for size in reversed(widths[1:])])
        self.head = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, channels, 3, padding=1))
        self.condition_size = condition

    def forward(self, x, time, city, controls, has_controls):
        condition = self.time(time_features(time, self.condition_size)) + self.city(city)
        condition = condition + self.controls(torch.cat([controls * has_controls[:, None], has_controls[:, None]], dim=1))
        h = self.stem(x)
        skips = []
        for level, blocks in enumerate(self.down):
            for block in blocks:
                h = block(h, condition)
            skips.append(h)
            if level < len(self.pool):
                h = self.pool[level](h)
        for block in self.middle:
            h = block(h, condition)
        for level, blocks in enumerate(self.up):
            if level > 0:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
                h = self.lift[level - 1](h)
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, condition)
        return self.head(h)


class PlanCrops(torch.utils.data.Dataset):
    def __init__(self, plans, mean, std, control_scale, size=64, length=4096, keep=None):
        # plans: list of raw plan arrays (C, H, W); keep: optional boolean masks of usable cells
        self.plans = plans
        self.mean = mean
        self.std = std
        self.control_scale = control_scale
        self.size = size
        self.length = length
        self.keep = keep
        if keep is None:
            areas = np.asarray([plan.shape[1] * plan.shape[2] for plan in plans], dtype=np.float64)
        else:
            areas = np.asarray([mask.sum() for mask in keep], dtype=np.float64)
        self.weights = areas / areas.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        city = int(np.random.choice(len(self.plans), p=self.weights))
        plan = self.plans[city]
        size = self.size
        for _attempt in range(20):
            row = random.randint(0, max(plan.shape[1] - size, 0))
            column = random.randint(0, max(plan.shape[2] - size, 0))
            if self.keep is None or self.keep[city][row : row + size, column : column + size].mean() > 0.5:
                break
        crop = np.zeros((plan.shape[0], size, size), dtype=np.float32)
        piece = plan[:, row : row + size, column : column + size]
        crop[:, : piece.shape[1], : piece.shape[2]] = piece
        crop = np.rot90(crop, random.randint(0, 3), axes=(1, 2))
        if random.random() < 0.5:
            crop = crop[:, :, ::-1]
        crop = np.ascontiguousarray(crop)
        controls = crop.mean(axis=(1, 2)) * self.control_scale
        normalised = (crop - self.mean[:, None, None]) / self.std[:, None, None]
        one_hot = np.zeros(len(self.plans), dtype=np.float32)
        one_hot[city] = 1.0
        return {"plan": normalised.astype(np.float32), "controls": controls.astype(np.float32), "city": one_hot}


def plan_loss(model, batch, drop=0.15):
    x = batch["plan"]
    size = x.shape[0]
    time = torch.rand(size, device=x.device)
    alpha, sigma = coefficients(time)
    noise = torch.randn_like(x)
    noisy = alpha[:, None, None, None] * x + sigma[:, None, None, None] * noise
    target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x
    keep_city = (torch.rand(size, device=x.device) > drop).float()
    has_controls = (torch.rand(size, device=x.device) > drop).float()
    prediction = model(noisy, time, batch["city"] * keep_city[:, None], batch["controls"], has_controls)
    return F.mse_loss(prediction, target)


FRACTIONS = ("coverage", "green", "water", "residential", "commercial", "industrial", "civic")


def match_means(plan, wanted, rounds=4):
    # the model decides where things go, this makes the amounts match what the sliders asked for
    plan = plan.copy()
    for index, channel in enumerate(CHANNELS):
        target = float(wanted[index])
        for _round in range(rounds):
            current = float(plan[index].mean())
            if current < 1e-6 or target <= 0:
                break
            plan[index] *= target / current
            if channel in FRACTIONS:
                plan[index] = np.clip(plan[index], 0.0, 1.0)
    return plan


@torch.no_grad()
def sample_plan(model, height, width, city, controls, mean, std, steps=50, guidance=1.5, generator=None, known=None, known_mask=None, wanted=None):
    device = next(model.parameters()).device
    channels = model.config["channels"]
    city = torch.as_tensor(city, dtype=torch.float32, device=device)[None]
    controls = torch.as_tensor(controls, dtype=torch.float32, device=device)[None]
    on = torch.ones(1, device=device)
    off = torch.zeros(1, device=device)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)[None, :, None, None]
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)[None, :, None, None]
    x = torch.randn(1, channels, height, width, device=device, generator=generator)
    if known is not None:
        known = (torch.as_tensor(known, dtype=torch.float32, device=device)[None] - mean_t) / std_t
        known_mask = torch.as_tensor(known_mask, dtype=torch.float32, device=device)[None, None]
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for index in range(steps):
        time = times[index].reshape(1)
        following = times[index + 1].reshape(1)
        alpha, sigma = coefficients(time)
        next_alpha, next_sigma = coefficients(following)
        conditioned = model(x, time, city, controls, on)
        if guidance != 1.0:
            free = model(x, time, torch.zeros_like(city), controls, off)
            velocity = free + guidance * (conditioned - free)
        else:
            velocity = conditioned
        clean = alpha * x - sigma * velocity
        noise = sigma * x + alpha * velocity
        x = next_alpha * clean + next_sigma * noise
        if known is not None:
            fresh = torch.randn(x.shape, device=device, generator=generator)
            x = known_mask * (next_alpha * known + next_sigma * fresh) + (1 - known_mask) * x
    plan = x[0] * std_t[0] + mean_t[0]
    plan = plan.clamp_min(0.0)
    fractions = [CHANNELS.index(name) for name in FRACTIONS]
    plan[fractions] = plan[fractions].clamp_max(1.0)
    plan = plan.cpu().numpy()
    if wanted is not None:
        plan = match_means(plan, wanted)
    return plan


def urban_cells(plan, reach=2):
    # cells near a road or rail line, so crops and slider defaults skip the sea and empty land
    lines = plan[:4].sum(axis=0) > 0
    if not lines.any():
        return np.ones(lines.shape, dtype=bool)
    return ndimage.binary_dilation(lines, iterations=reach)


def plan_statistics(plans):
    stacked = np.concatenate([plan.reshape(plan.shape[0], -1) for plan in plans], axis=1)
    mean = stacked.mean(axis=1).astype(np.float32)
    std = np.maximum(stacked.std(axis=1), 1e-3).astype(np.float32)
    control_scale = (1.0 / np.maximum(mean, 1e-3)).astype(np.float32)
    return mean, std, control_scale
