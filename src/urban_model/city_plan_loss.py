from __future__ import annotations

import torch
from torch.nn import functional as F


def city_plan_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    plan_mean: torch.Tensor,
    plan_std: torch.Tensor,
    global_mean: torch.Tensor,
    global_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    plan_error = F.smooth_l1_loss(
        output["plan_grid"],
        batch["plan_grid"],
        reduction="none",
    )
    positive = batch["plan_grid_raw"] > 0
    plan_weight = torch.where(
        positive,
        torch.full_like(plan_error, 4.0),
        torch.ones_like(plan_error),
    )
    plan = (plan_error * plan_weight).sum() / plan_weight.sum().clamp_min(1.0)

    global_loss = F.smooth_l1_loss(
        output["plan_global"],
        batch["plan_global"],
    )
    total = plan + global_loss

    predicted_plan = (
        output["plan_grid"] * plan_std[None, None]
        + plan_mean[None, None]
    ).clamp_min(0.0)
    predicted_global = (
        output["plan_global"] * global_std[None]
        + global_mean[None]
    )

    raw_plan_mae = (
        predicted_plan - batch["plan_grid_raw"]
    ).abs().mean()
    raw_global_mae = (
        predicted_global - batch["plan_global_raw"]
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

    occupied = batch["plan_grid_raw"][..., 0] > 0
    predicted_occupied = predicted_plan[..., 0] > 0.5
    intersection = (occupied & predicted_occupied).sum(dim=1).to(torch.float32)
    union = (occupied | predicted_occupied).sum(dim=1).clamp_min(1).to(torch.float32)
    occupancy_iou = (intersection / union).mean()

    metrics = {
        "plan": float(plan.detach()),
        "global": float(global_loss.detach()),
        "raw_plan_mae": float(raw_plan_mae.detach()),
        "raw_global_mae": float(raw_global_mae.detach()),
        "node_mae": float(node_mae.detach()),
        "edge_mae": float(edge_mae.detach()),
        "component_mae": float(component_mae.detach()),
        "occupancy_iou": float(occupancy_iou.detach()),
    }
    return total, metrics
