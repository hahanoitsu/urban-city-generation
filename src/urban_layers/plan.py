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
    def __init__(self, cities, channels=len(CHANNELS), width=64, condition=256, given=0, controls=None):
        super().__init__()
        controls = channels if controls is None else controls
        self.config = {"cities": cities, "channels": channels, "width": width, "condition": condition, "given": given, "controls": controls}
        self.cities = cities
        self.time = nn.Sequential(nn.Linear(condition, condition), nn.SiLU(), nn.Linear(condition, condition))
        self.city = nn.Linear(cities, condition)
        self.controls = nn.Sequential(nn.Linear(controls + 1, condition), nn.SiLU(), nn.Linear(condition, condition))
        widths = [width, width * 2, width * 4]
        self.stem = nn.Conv2d(channels + given, width, 3, padding=1)
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

    def forward(self, x, time, city, controls, has_controls, given=None):
        condition = self.time(time_features(time, self.condition_size)) + self.city(city)
        condition = condition + self.controls(torch.cat([controls * has_controls[:, None], has_controls[:, None]], dim=1))
        h = self.stem(x if given is None else torch.cat([x, given], dim=1))
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

    def place(self):
        city = int(np.random.choice(len(self.plans), p=self.weights))
        plan = self.plans[city]
        size = self.size
        for _attempt in range(20):
            row = random.randint(0, max(plan.shape[1] - size, 0))
            column = random.randint(0, max(plan.shape[2] - size, 0))
            if self.keep is None or self.keep[city][row : row + size, column : column + size].mean() > 0.5:
                break
        return city, row, column, random.randint(0, 3), random.random() < 0.5

    def cut(self, plan, row, column, turns, mirror):
        size = self.size
        crop = np.zeros((plan.shape[0], size, size), dtype=np.float32)
        piece = plan[:, row : row + size, column : column + size]
        crop[:, : piece.shape[1], : piece.shape[2]] = piece
        crop = np.rot90(crop, turns, axes=(1, 2))
        if mirror:
            crop = crop[:, :, ::-1]
        return np.ascontiguousarray(crop)

    def __getitem__(self, index):
        city, row, column, turns, mirror = self.place()
        crop = self.cut(self.plans[city], row, column, turns, mirror)
        controls = crop.mean(axis=(1, 2)) * self.control_scale
        normalised = (crop - self.mean[:, None, None]) / self.std[:, None, None]
        one_hot = np.zeros(len(self.plans), dtype=np.float32)
        one_hot[city] = 1.0
        return {"plan": normalised.astype(np.float32), "controls": controls.astype(np.float32), "city": one_hot}


def coarse(plan, factor=4):
    channels, height, width = plan.shape
    padded = np.pad(plan, ((0, 0), (0, -height % factor), (0, -width % factor)), mode="edge")
    pooled = padded.reshape(channels, padded.shape[1] // factor, factor, padded.shape[2] // factor, factor).mean(axis=(2, 4))
    return np.repeat(np.repeat(pooled, factor, axis=1), factor, axis=2)[:, :height, :width]


def pattern_controls(pattern, keep=None):
    cells = pattern.reshape(pattern.shape[0], -1) if keep is None else pattern[:, keep]
    known = cells[2] > 0.5
    strength = float(np.hypot(cells[0], cells[1])[known].mean()) if known.any() else 0.0
    return np.asarray([cells[0].mean(), cells[1].mean(), cells[2].mean(), strength], dtype=np.float32)


class PatternCrops(PlanCrops):
    def __init__(self, patterns, plans, mean, std, given_mean, given_std, size=64, length=4096, keep=None):
        super().__init__(patterns, mean, std, np.ones(len(mean), dtype=np.float32), size, length, keep)
        self.given = plans
        self.given_mean = given_mean
        self.given_std = given_std

    def __getitem__(self, index):
        city, row, column, turns, mirror = self.place()
        crop = self.cut(self.plans[city], row, column, turns, mirror)
        if mirror:
            crop[1] = -crop[1]
        given = coarse(self.cut(self.given[city], row, column, turns, mirror))
        one_hot = np.zeros(len(self.plans), dtype=np.float32)
        one_hot[city] = 1.0
        return {
            "plan": ((crop - self.mean[:, None, None]) / self.std[:, None, None]).astype(np.float32),
            "given": ((given - self.given_mean[:, None, None]) / self.given_std[:, None, None]).astype(np.float32),
            "controls": pattern_controls(crop),
            "city": one_hot,
        }


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
    prediction = model(noisy, time, batch["city"] * keep_city[:, None], batch["controls"], has_controls, batch.get("given"))
    return F.mse_loss(prediction, target)


FRACTIONS = ("coverage", "green", "water", "residential", "commercial", "industrial", "civic")


def match_means(plan, wanted, rounds=4):
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
            elif index < 4:
                plan[index] = np.clip(plan[index], 0.0, 3.0)
    return plan


@torch.no_grad()
def denoise(model, height, width, city, controls, mean, std, steps=50, guidance=1.5, generator=None, known=None, known_mask=None, given=None):
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
        conditioned = model(x, time, city, controls, on, given)
        if guidance != 1.0:
            free = model(x, time, torch.zeros_like(city), controls, off, given)
            velocity = free + guidance * (conditioned - free)
        else:
            velocity = conditioned
        clean = alpha * x - sigma * velocity
        noise = sigma * x + alpha * velocity
        x = next_alpha * clean + next_sigma * noise
        if known is not None:
            fresh = torch.randn(x.shape, device=device, generator=generator)
            x = known_mask * (next_alpha * known + next_sigma * fresh) + (1 - known_mask) * x
    return x[0] * std_t[0] + mean_t[0]


def sample_plan(model, height, width, city, controls, mean, std, steps=50, guidance=1.5, generator=None, known=None, known_mask=None, wanted=None):
    plan = denoise(model, height, width, city, controls, mean, std, steps, guidance, generator, known, known_mask).clamp_min(0.0)
    fractions = [CHANNELS.index(name) for name in FRACTIONS]
    plan[fractions] = plan[fractions].clamp_max(1.0)
    plan = plan.cpu().numpy()
    if wanted is not None:
        plan = match_means(plan, wanted)
    return plan


def sample_pattern(model, plan, city, controls, mean, std, given_mean, given_std, steps=50, guidance=1.5, generator=None):
    device = next(model.parameters()).device
    shift = torch.as_tensor(given_mean, dtype=torch.float32, device=device)[:, None, None]
    scale = torch.as_tensor(given_std, dtype=torch.float32, device=device)[:, None, None]
    given = ((torch.as_tensor(coarse(plan), dtype=torch.float32, device=device) - shift) / scale)[None]
    pattern = denoise(model, plan.shape[1], plan.shape[2], city, controls, mean, std, steps, guidance, generator, given=given).cpu().numpy()
    known = pattern[2] > 0.5
    pattern[:2] = pattern[:2] / np.maximum(np.hypot(pattern[0], pattern[1]), 1.0) * known
    pattern[2] = known
    return pattern.astype(np.float32)


def urban_cells(plan, reach=2):
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
