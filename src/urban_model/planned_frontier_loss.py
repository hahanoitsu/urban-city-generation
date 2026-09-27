from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from urban_model.frontier_data import (
    OP_GROW,
    OP_LINK,
    OP_ROOT,
)


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(values)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def _masked_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)
    return _masked_mean(values, mask)


def _gaussian_nll(
    mean: torch.Tensor,
    logstd: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    logstd = logstd.clamp(-5.0, 1.0)
    values = 0.5 * (
        (target - mean).square()
        * torch.exp(-2.0 * logstd)
        + 2.0 * logstd
        + math.log(2.0 * math.pi)
    )
    return _masked_mean(values, mask)


def planned_frontier_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    steps = output["op"].shape[1]
    target_op = batch["program_op"][:, 1 : steps + 1]
    positions = torch.arange(
        steps,
        device=target_op.device,
    )
    valid = positions[None] < (
        batch["program_length"] - 1
    )[:, None]
    node_mask = valid & (
        (target_op == OP_ROOT)
        | (target_op == OP_GROW)
    )
    spatial_mask = valid & (
        (target_op == OP_ROOT)
        | (target_op == OP_GROW)
        | (target_op == OP_LINK)
    )
    edge_mask = valid & (
        (target_op == OP_GROW)
        | (target_op == OP_LINK)
    )

    losses = {
        "op": _masked_ce(
            output["op"],
            target_op,
            valid,
        ),
        "xy": _gaussian_nll(
            output["xy_mean"],
            output["xy_logstd"],
            batch["program_xy"][:, 1 : steps + 1],
            spatial_mask,
        ),
        "node_mode": _masked_ce(
            output["node_mode"],
            batch["program_node_mode"][:, 1 : steps + 1],
            node_mask,
        ),
        "node_vertical": _masked_ce(
            output["node_vertical"],
            batch["program_node_vertical"][:, 1 : steps + 1],
            node_mask,
        ),
        "node_boundary": _masked_mean(
            F.binary_cross_entropy_with_logits(
                output["node_boundary"],
                batch["program_node_boundary"][
                    :, 1 : steps + 1
                ],
                reduction="none",
            ),
            node_mask,
        ),
        "edge_class": _masked_ce(
            output["edge_class"],
            batch["program_edge_class"][:, 1 : steps + 1],
            edge_mask,
        ),
        "edge_vertical": _masked_ce(
            output["edge_vertical"],
            batch["program_edge_vertical"][
                :, 1 : steps + 1
            ],
            edge_mask,
        ),
        "width": _gaussian_nll(
            output["width_mean"],
            output["width_logstd"],
            batch["program_edge_width"][
                :, 1 : steps + 1
            ],
            edge_mask,
        ),
        "curve": _gaussian_nll(
            output["curve_mean"],
            output["curve_logstd"],
            batch["program_curve"][:, 1 : steps + 1],
            edge_mask,
        ),
    }

    weights = {
        "op": 2.0,
        "xy": 2.0,
        "node_mode": 0.5,
        "node_vertical": 0.5,
        "node_boundary": 0.5,
        "edge_class": 1.0,
        "edge_vertical": 0.5,
        "width": 0.5,
        "curve": 1.0,
    }
    total = sum(
        losses[name] * weights[name]
        for name in losses
    ) / sum(weights.values())
    metrics = {
        name: float(value.detach())
        for name, value in losses.items()
    }
    metrics["total"] = float(total.detach())
    return total, metrics
