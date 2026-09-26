from __future__ import annotations

import math

import torch
from torch.nn import functional as F


CONTINUOUS_FIELDS = (
    "node_position",
    "edge_width",
    "edge_shape",
    "building_shape",
    "building_height",
    "building_base_z",
    "area_shape",
)

CATEGORY_MASKS = {
    "node_presence": 2,
    "edge_presence": 2,
    "edge_mode": 2,
    "edge_class": 7,
    "edge_vertical": 4,
    "building_presence": 2,
    "building_kind": 8,
    "area_presence": 2,
    "area_kind": 6,
}


def noise_coefficients(time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    angle = time.clamp(0.0, 1.0) * math.pi / 2.0
    return torch.cos(angle), torch.sin(angle)


def corrupt_scene(
    scene: dict[str, torch.Tensor],
    time: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = {}
    alpha, sigma = noise_coefficients(time)
    for name in CONTINUOUS_FIELDS:
        value = scene[name]
        shape = [value.shape[0]] + [1] * (value.ndim - 1)
        a = alpha.reshape(shape)
        s = sigma.reshape(shape)
        result[name] = a * value + s * torch.randn_like(value)

    for name, mask_index in CATEGORY_MASKS.items():
        value = scene[name]
        shape = [value.shape[0]] + [1] * (value.ndim - 1)
        probability = time.reshape(shape)
        mask = torch.rand(value.shape, device=value.device) < probability
        result[name] = torch.where(mask, torch.full_like(value, mask_index), value)

    for name in (
        "node_z_valid",
        "edge_from",
        "edge_to",
        "edge_width_valid",
        "edge_width_weight",
        "edge_z_valid",
        "building_height_valid",
        "building_height_weight",
        "building_base_z_valid",
    ):
        result[name] = scene[name]
    return result


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denominator = mask.expand_as(values).sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def _mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    values = (prediction - target) ** 2
    if weight is None:
        return _masked_mean(values, mask)
    combined = mask.to(values.dtype) * weight.to(values.dtype)
    while combined.ndim < values.ndim:
        combined = combined.unsqueeze(-1)
    denominator = combined.expand_as(values).sum().clamp_min(1.0)
    return (values * combined).sum() / denominator


def _ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    balance_presence: bool = False,
) -> torch.Tensor:
    classes = logits.shape[-1]
    losses = F.cross_entropy(
        logits.reshape(-1, classes),
        target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)
    if mask is not None:
        return _masked_mean(losses, mask)
    if not balance_presence:
        return losses.mean()
    present = target.eq(1)
    count = present.sum().clamp_min(1)
    weight = min(float(target.numel() / count.item()), 12.0)
    weights = torch.where(present, torch.full_like(losses, weight), torch.ones_like(losses))
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def structured_city_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    node = target["node_presence"].eq(1)
    edge = target["edge_presence"].eq(1)
    building = target["building_presence"].eq(1)
    area = target["area_presence"].eq(1)

    node_xy = _mse(output["node_position"][..., :2], target["node_position"][..., :2], node)
    node_z = _mse(
        output["node_position"][..., 2],
        target["node_position"][..., 2],
        node & target["node_z_valid"],
    )

    edge_xy = _mse(output["edge_shape"][..., :2], target["edge_shape"][..., :2], edge)
    edge_z = _mse(
        output["edge_shape"][..., 2],
        target["edge_shape"][..., 2],
        edge[:, :, None] & target["edge_z_valid"],
    )

    losses = {
        "node_presence": _ce(
            output["node_presence"],
            target["node_presence"],
            balance_presence=True,
        ),
        "node_xy": node_xy,
        "node_z": node_z,
        "edge_presence": _ce(
            output["edge_presence"],
            target["edge_presence"],
            balance_presence=True,
        ),
        "edge_mode": _ce(output["edge_mode"], target["edge_mode"], edge),
        "edge_class": _ce(output["edge_class"], target["edge_class"], edge),
        "edge_vertical": _ce(output["edge_vertical"], target["edge_vertical"], edge),
        "edge_from": _ce(output["edge_from"], target["edge_from"], edge),
        "edge_to": _ce(output["edge_to"], target["edge_to"], edge),
        "edge_width": _mse(
            output["edge_width"],
            target["edge_width"],
            edge & target["edge_width_valid"],
            target["edge_width_weight"],
        ),
        "edge_xy": edge_xy,
        "edge_z": edge_z,
        "building_presence": _ce(
            output["building_presence"],
            target["building_presence"],
            balance_presence=True,
        ),
        "building_kind": _ce(
            output["building_kind"],
            target["building_kind"],
            building,
        ),
        "building_shape": _mse(
            output["building_shape"],
            target["building_shape"],
            building,
        ),
        "building_height": _mse(
            output["building_height"],
            target["building_height"],
            building & target["building_height_valid"],
            target["building_height_weight"],
        ),
        "building_base_z": _mse(
            output["building_base_z"],
            target["building_base_z"],
            building & target["building_base_z_valid"],
        ),
        "area_presence": _ce(
            output["area_presence"],
            target["area_presence"],
            balance_presence=True,
        ),
        "area_kind": _ce(output["area_kind"], target["area_kind"], area),
        "area_shape": _mse(output["area_shape"], target["area_shape"], area),
    }
    active = [
        value
        for name, value in losses.items()
        if name not in {"node_z", "edge_z", "building_base_z"}
        or bool(value.detach().abs().item() > 0)
    ]
    total = torch.stack(active).mean()
    return total, {name: float(value.detach()) for name, value in losses.items()}
