from __future__ import annotations

import torch
from torch.nn import functional as F


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(values)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def _channel_iou(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    intersection = (predicted & target).sum(dim=1).to(torch.float32)
    union = (predicted | target).sum(dim=1).clamp_min(1).to(torch.float32)
    return (intersection / union).mean()


def city_plan_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    presence_pos_weight: torch.Tensor,
    global_mean: torch.Tensor,
    global_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    presence = F.binary_cross_entropy_with_logits(
        output["plan_presence"],
        batch["plan_presence"],
        pos_weight=presence_pos_weight[None, None],
    )

    positive = batch["plan_presence"] > 0.5
    count = _masked_mean(
        F.smooth_l1_loss(
            output["plan_log_count"],
            batch["plan_log_counts"],
            reduction="none",
        ),
        positive,
    )

    orientation = _masked_mean(
        F.smooth_l1_loss(
            output["plan_orientation"],
            batch["plan_orientation"],
            reduction="none",
        ),
        batch["plan_orientation_mask"],
    )

    global_loss = F.smooth_l1_loss(
        output["plan_global"],
        batch["plan_global"],
    )

    total = (
        presence * 2.0
        + count
        + orientation
        + global_loss
    ) / 5.0

    probability = torch.sigmoid(output["plan_presence"])
    predicted_presence = probability > 0.5
    target_presence = batch["plan_presence"] > 0.5
    predicted_counts = (
        torch.expm1(output["plan_log_count"]).clamp_min(0.0)
        * probability
    )
    predicted_global = (
        output["plan_global"] * global_std[None]
        + global_mean[None]
    )

    junction_iou = _channel_iou(
        predicted_presence[..., 0],
        target_presence[..., 0],
    )
    corridor_target = target_presence[..., 3:7].any(dim=-1)
    corridor_predicted = predicted_presence[..., 3:7].any(dim=-1)
    corridor_iou = _channel_iou(
        corridor_predicted,
        corridor_target,
    )
    rail_iou = _channel_iou(
        predicted_presence[..., 6],
        target_presence[..., 6],
    )
    count_mae = (
        predicted_counts - batch["plan_counts"]
    ).abs().mean()
    node_mae = (
        predicted_global[:, 0] - batch["plan_global_raw"][:, 0]
    ).abs().mean()
    edge_mae = (
        predicted_global[:, 1] - batch["plan_global_raw"][:, 1]
    ).abs().mean()
    component_mae = (
        predicted_global[:, 2] - batch["plan_global_raw"][:, 2]
    ).abs().mean()

    metrics = {
        "presence": float(presence.detach()),
        "count": float(count.detach()),
        "orientation": float(orientation.detach()),
        "global": float(global_loss.detach()),
        "count_mae": float(count_mae.detach()),
        "node_mae": float(node_mae.detach()),
        "edge_mae": float(edge_mae.detach()),
        "component_mae": float(component_mae.detach()),
        "junction_iou": float(junction_iou.detach()),
        "corridor_iou": float(corridor_iou.detach()),
        "rail_iou": float(rail_iou.detach()),
    }
    return total, metrics
