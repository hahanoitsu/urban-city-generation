from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from urban_model.grid_city import category_sizes


CONTINUOUS = (
    "node_pos",
    "edge_width",
    "edge_curve",
    "building_frame",
    "building_shape",
    "building_height",
    "area_frame",
    "area_shape",
)
CATEGORIES = (
    "node_presence",
    "edge_target",
    "edge_mode",
    "edge_class",
    "edge_vertical",
    "building_presence",
    "building_kind",
    "area_presence",
    "area_kind",
)


def coefficients(time):
    angle = time.clamp(0.0, 1.0) * math.pi / 2.0
    return torch.cos(angle), torch.sin(angle)


def _expand(value, like):
    return value.reshape([value.shape[0]] + [1] * (like.ndim - 1))


def corrupt(scene, time, config):
    sizes = category_sizes(config)
    alpha, sigma = coefficients(time)
    result = dict(scene)
    for name in CONTINUOUS:
        value = scene[name]
        result[name] = _expand(alpha, value) * value + _expand(sigma, value) * torch.randn_like(value)
    for name in CATEGORIES:
        value = scene[name]
        masked = torch.rand(value.shape, device=value.device) < _expand(time, value)
        result[name] = torch.where(masked, torch.full_like(value, sizes[name]), value)
    return result


def _mean(values, mask):
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    total = mask.sum()
    return (values * mask).sum() / total.clamp_min(1.0), bool(total > 0)


def _mse(prediction, target, mask, weight=None):
    values = (prediction.float() - target.float()) ** 2
    if weight is not None:
        weight = weight.float()
        while weight.ndim < values.ndim:
            weight = weight.unsqueeze(-1)
        values = values * weight
    return _mean(values, mask)


def _ce(logits, target, mask):
    classes = logits.shape[-1]
    values = F.cross_entropy(logits.float().reshape(-1, classes), target.reshape(-1), reduction="none")
    return _mean(values.reshape(target.shape), mask)


def grid_city_loss(output, target):
    node = target["node_presence"].eq(1)
    edge = node[..., None] & target["edge_target"].gt(0)
    building = target["building_presence"].eq(1)
    area = target["area_presence"].eq(1)
    everything = torch.ones_like(node)
    parts = {
        "node_presence": _ce(output["node_presence"], target["node_presence"], everything),
        "node_pos": _mse(output["node_pos"], target["node_pos"], node),
        "edge_target": _ce(output["edge_target"], target["edge_target"], node[..., None].expand_as(target["edge_target"])),
        "edge_mode": _ce(output["edge_mode"], target["edge_mode"], edge),
        "edge_class": _ce(output["edge_class"], target["edge_class"], edge),
        "edge_vertical": _ce(output["edge_vertical"], target["edge_vertical"], edge),
        "edge_width": _mse(output["edge_width"], target["edge_width"], edge & target["edge_width_valid"], target["edge_width_weight"]),
        "edge_curve": _mse(output["edge_curve"], target["edge_curve"], edge),
        "building_presence": _ce(output["building_presence"], target["building_presence"], torch.ones_like(building)),
        "building_frame": _mse(output["building_frame"], target["building_frame"], building),
        "building_shape": _mse(output["building_shape"], target["building_shape"], building),
        "building_height": _mse(
            output["building_height"],
            target["building_height"],
            building & target["building_height_valid"],
            target["building_height_weight"],
        ),
        "building_kind": _ce(output["building_kind"], target["building_kind"], building),
        "area_presence": _ce(output["area_presence"], target["area_presence"], torch.ones_like(area)),
        "area_frame": _mse(output["area_frame"], target["area_frame"], area),
        "area_shape": _mse(output["area_shape"], target["area_shape"], area),
        "area_kind": _ce(output["area_kind"], target["area_kind"], area),
    }
    # empty slots are zeros in the data, train them too so sampling keeps them looking empty
    empty = [
        _mse(output["node_pos"], target["node_pos"], ~node),
        _ce(output["edge_target"], target["edge_target"], ~node[..., None].expand_as(target["edge_target"])),
        _ce(output["edge_mode"], target["edge_mode"], ~edge),
        _ce(output["edge_class"], target["edge_class"], ~edge),
        _ce(output["edge_vertical"], target["edge_vertical"], ~edge),
        _mse(output["edge_width"], target["edge_width"], ~edge),
        _mse(output["edge_curve"], target["edge_curve"], ~edge),
        _mse(output["building_frame"], target["building_frame"], ~building),
        _mse(output["building_shape"], target["building_shape"], ~building),
        _mse(output["building_height"], target["building_height"], ~building),
        _ce(output["building_kind"], target["building_kind"], ~building),
        _mse(output["area_frame"], target["area_frame"], ~area),
        _mse(output["area_shape"], target["area_shape"], ~area),
        _ce(output["area_kind"], target["area_kind"], ~area),
    ]
    values = [value for value, present in empty if present]
    if values:
        parts["empty"] = (torch.stack(values).mean(), True)
    active = [value for value, present in parts.values() if present]
    total = torch.stack(active).mean()
    return total, {name: float(value.detach()) for name, (value, _present) in parts.items()}


def initial_scene(config, batch, device, generator=None):
    sizes = category_sizes(config)
    nodes = config.node_grid**2 * config.node_slots
    buildings = config.building_grid**2 * config.building_slots
    areas = config.area_grid**2 * config.area_slots
    edges = config.edge_slots

    def noise(*shape):
        return torch.randn(*shape, device=device, generator=generator)

    def masked(name, *shape):
        return torch.full(shape, sizes[name], dtype=torch.long, device=device)

    return {
        "node_pos": noise(batch, nodes, 2),
        "edge_width": noise(batch, nodes, edges, 1),
        "edge_curve": noise(batch, nodes, edges, config.curve_points, 2),
        "building_frame": noise(batch, buildings, 3),
        "building_shape": noise(batch, buildings, config.building_points, 2),
        "building_height": noise(batch, buildings, 1),
        "area_frame": noise(batch, areas, 3),
        "area_shape": noise(batch, areas, config.area_points, 2),
        "node_presence": masked("node_presence", batch, nodes),
        "edge_target": masked("edge_target", batch, nodes, edges),
        "edge_mode": masked("edge_mode", batch, nodes, edges),
        "edge_class": masked("edge_class", batch, nodes, edges),
        "edge_vertical": masked("edge_vertical", batch, nodes, edges),
        "building_presence": masked("building_presence", batch, buildings),
        "building_kind": masked("building_kind", batch, buildings),
        "area_presence": masked("area_presence", batch, areas),
        "area_kind": masked("area_kind", batch, areas),
    }


def _unmask(current, logits, mask_index, keep_fraction, generator, greedy):
    probability = torch.softmax(logits.float(), dim=-1)
    flat = probability.reshape(-1, probability.shape[-1])
    if greedy:
        sample = flat.argmax(dim=-1)
    else:
        sample = torch.multinomial(flat, 1, generator=generator)[:, 0]
    confidence = flat.gather(1, sample[:, None])[:, 0].reshape(current.shape)
    sample = sample.reshape(current.shape)
    masked = current.eq(mask_index)
    if keep_fraction <= 0:
        return torch.where(masked, sample, current)
    flat_masked = masked.reshape(current.shape[0], -1)
    flat_confidence = torch.where(masked, confidence, torch.full_like(confidence, -1.0)).reshape(current.shape[0], -1)
    total = flat_masked.shape[1]
    remain = int(round(keep_fraction * total))
    result = torch.where(masked, sample, current).reshape(current.shape[0], -1)
    if remain > 0:
        # keep the least confident of the still masked tokens masked
        order = torch.argsort(flat_confidence + (~flat_masked) * 10.0, dim=1)
        chosen = order[:, :remain]
        result.scatter_(1, chosen, mask_index)
    return result.reshape(current.shape)


def _hold_absent(scene, sizes, next_time, generator):
    _alpha, sigma = coefficients(next_time)
    level = float(next_time[0])
    scale = float(sigma[0])

    def quiet(value, absent):
        while absent.ndim < value.ndim:
            absent = absent.unsqueeze(-1)
        noise = scale * torch.randn(value.shape, device=value.device, generator=generator)
        return torch.where(absent.expand_as(value), noise, value)

    def blank(value, absent, mask_index):
        masked = torch.rand(value.shape, device=value.device, generator=generator) < level
        fill = torch.where(masked, torch.full_like(value, mask_index), torch.zeros_like(value))
        return torch.where(absent, fill, value)

    node = scene["node_presence"].eq(0)
    no_edge = node[..., None] | scene["edge_target"].eq(0)
    scene["node_pos"] = quiet(scene["node_pos"], node)
    scene["edge_width"] = quiet(scene["edge_width"], no_edge)
    scene["edge_curve"] = quiet(scene["edge_curve"], no_edge)
    scene["edge_target"] = blank(scene["edge_target"], node[..., None].expand_as(scene["edge_target"]), sizes["edge_target"])
    for name in ("edge_mode", "edge_class", "edge_vertical"):
        scene[name] = blank(scene[name], no_edge, sizes[name])
    for kind in ("building", "area"):
        absent = scene[f"{kind}_presence"].eq(0)
        names = ("frame", "shape", "height") if kind == "building" else ("frame", "shape")
        for name in names:
            scene[f"{kind}_{name}"] = quiet(scene[f"{kind}_{name}"], absent)
        scene[f"{kind}_kind"] = blank(scene[f"{kind}_kind"], absent, sizes[f"{kind}_kind"])


@torch.inference_mode()
def generate(model, inputs, steps=40, generator=None, greedy=False):
    config = model.config
    sizes = category_sizes(config)
    device = inputs["context"].device
    batch = inputs["context"].shape[0]
    scene = initial_scene(config, batch, device, generator)
    for index in range(steps, 0, -1):
        time = torch.full((batch,), index / steps, device=device)
        next_time = torch.full((batch,), (index - 1) / steps, device=device)
        output = model(
            scene,
            inputs["context"],
            inputs["relations"],
            inputs["context_padding"],
            inputs["ports"],
            inputs["port_padding"],
            time,
        )
        alpha, sigma = coefficients(time)
        next_alpha, next_sigma = coefficients(next_time)
        for name in CONTINUOUS:
            current = scene[name]
            prediction = output[name].float()
            a = _expand(alpha, current)
            s = _expand(sigma, current).clamp_min(1e-5)
            epsilon = (current - a * prediction) / s
            scene[name] = _expand(next_alpha, current) * prediction + _expand(next_sigma, current) * epsilon
        for name in CATEGORIES:
            scene[name] = _unmask(scene[name], output[name], sizes[name], (index - 1) / steps, generator, greedy)
        _hold_absent(scene, sizes, next_time, generator)
    return scene
