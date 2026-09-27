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


def _balanced_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    maximum_positive_weight: float = 80.0,
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(
            target,
            dtype=torch.bool,
        )
    valid = target[mask]
    positives = valid.sum()
    negatives = valid.numel() - positives
    positive_weight = (
        negatives / positives.clamp_min(1.0)
    ).clamp(1.0, maximum_positive_weight)
    values = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=positive_weight,
    )
    return _masked_mean(values, mask)


def _edge_targets(
    batch: dict[str, torch.Tensor],
    active_slots: int,
    curve_points: int,
):
    device = batch["edge_pairs"].device
    size = (
        batch["edge_pairs"].shape[0],
        active_slots,
        active_slots,
    )
    edge_class = torch.zeros(
        size,
        dtype=torch.long,
        device=device,
    )
    edge_vertical = torch.zeros(
        size,
        dtype=torch.long,
        device=device,
    )
    edge_width = torch.zeros(
        (*size, 1),
        dtype=torch.float32,
        device=device,
    )
    edge_curve = torch.zeros(
        (*size, curve_points),
        dtype=torch.float32,
        device=device,
    )
    positive = torch.zeros(
        size,
        dtype=torch.bool,
        device=device,
    )

    for batch_index in range(size[0]):
        count = int(batch["edge_count"][batch_index])
        if count == 0:
            continue
        pairs = batch["edge_pairs"][
            batch_index,
            :count,
        ]
        left = pairs[:, 0]
        right = pairs[:, 1]
        valid = (
            (left < active_slots)
            & (right < active_slots)
        )
        left = left[valid]
        right = right[valid]
        source = torch.arange(
            count,
            device=device,
        )[valid]
        edge_class[
            batch_index,
            left,
            right,
        ] = batch["edge_class"][
            batch_index,
            source,
        ]
        edge_vertical[
            batch_index,
            left,
            right,
        ] = batch["edge_vertical"][
            batch_index,
            source,
        ]
        edge_width[
            batch_index,
            left,
            right,
        ] = batch["edge_width"][
            batch_index,
            source,
        ]
        edge_curve[
            batch_index,
            left,
            right,
        ] = batch["edge_curve"][
            batch_index,
            source,
        ]
        positive[
            batch_index,
            left,
            right,
        ] = True

    return (
        edge_class,
        edge_vertical,
        edge_width,
        edge_curve,
        positive,
    )


def plan_anchor_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    target_size_m: float,
    anchor_grid_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    present = batch["slot_presence"].bool()
    slot_presence = _balanced_bce(
        output["slot_presence"],
        batch["slot_presence"],
        maximum_positive_weight=60.0,
    )
    node_offset = _masked_mean(
        F.smooth_l1_loss(
            output["node_offset"],
            batch["node_offset"],
            reduction="none",
        ),
        present,
    )
    node_mode = _masked_ce(
        output["node_mode"],
        batch["node_mode"],
        present,
    )
    node_vertical = _masked_ce(
        output["node_vertical"],
        batch["node_vertical"],
        present,
    )
    node_boundary = _balanced_bce(
        output["node_boundary"],
        batch["node_boundary"],
        present,
        maximum_positive_weight=20.0,
    )

    active_slots = output["node_degree"].shape[1]
    indexes = torch.arange(
        active_slots,
        device=output["node_degree"].device,
    )
    active = indexes[None] < batch[
        "active_count"
    ][:, None]
    node_degree = _masked_ce(
        output["node_degree"],
        batch["node_degree"][
            :, :active_slots
        ].clamp_max(
            output["node_degree"].shape[-1] - 1
        ),
        active,
    )

    (
        edge_class_target,
        edge_vertical_target,
        edge_width_target,
        edge_curve_target,
        positive,
    ) = _edge_targets(
        batch,
        active_slots,
        output["edge_curve"].shape[-1],
    )
    pair_mask = (
        active[:, :, None]
        & active[:, None, :]
        & (
            indexes[None, :, None]
            < indexes[None, None, :]
        )
    )
    edge_exists = _balanced_bce(
        output["edge_exists"],
        positive.to(
            output["edge_exists"].dtype
        ),
        pair_mask,
        maximum_positive_weight=80.0,
    )
    edge_class = _masked_ce(
        output["edge_class"],
        edge_class_target,
        positive,
    )
    edge_vertical = _masked_ce(
        output["edge_vertical"],
        edge_vertical_target,
        positive,
    )
    edge_width = _masked_mean(
        F.smooth_l1_loss(
            output["edge_width"],
            edge_width_target,
            reduction="none",
        ),
        positive,
    )
    edge_curve = _masked_mean(
        F.smooth_l1_loss(
            output["edge_curve"],
            edge_curve_target,
            reduction="none",
        ),
        positive,
    )
    curve_second = (
        output["edge_curve"][..., 2:]
        - 2.0 * output["edge_curve"][..., 1:-1]
        + output["edge_curve"][..., :-2]
    )
    curve_smooth = _masked_mean(
        curve_second.square(),
        positive,
    )

    losses = {
        "slot_presence": slot_presence,
        "node_offset": node_offset,
        "node_mode": node_mode,
        "node_vertical": node_vertical,
        "node_boundary": node_boundary,
        "node_degree": node_degree,
        "edge_exists": edge_exists,
        "edge_class": edge_class,
        "edge_vertical": edge_vertical,
        "edge_width": edge_width,
        "edge_curve": edge_curve,
        "curve_smooth": curve_smooth,
    }
    weights = {
        "slot_presence": 2.0,
        "node_offset": 2.0,
        "node_mode": 0.5,
        "node_vertical": 0.5,
        "node_boundary": 0.5,
        "node_degree": 0.75,
        "edge_exists": 2.0,
        "edge_class": 1.0,
        "edge_vertical": 0.5,
        "edge_width": 0.5,
        "edge_curve": 1.5,
        "curve_smooth": 0.05,
    }
    total = sum(
        losses[name] * weights[name]
        for name in losses
    ) / sum(weights.values())

    batch_size = output["slot_presence"].shape[0]
    scores = output["slot_presence"].reshape(
        batch_size,
        -1,
    )
    target_presence = batch[
        "slot_presence"
    ].reshape(batch_size, -1)
    anchor_recalls = []
    for batch_index in range(batch_size):
        count = int(batch["active_count"][batch_index])
        if count <= 0:
            continue
        selected = torch.topk(
            scores[batch_index],
            k=count,
        ).indices
        anchor_recalls.append(
            target_presence[
                batch_index,
                selected,
            ].mean()
        )

    edge_recalls = []
    for batch_index in range(batch_size):
        node_count = int(
            min(
                batch["active_count"][batch_index],
                active_slots,
            )
        )
        edge_count = int(batch["edge_count"][batch_index])
        if node_count < 2 or edge_count <= 0:
            continue
        tri = torch.triu_indices(
            node_count,
            node_count,
            offset=1,
            device=output["edge_exists"].device,
        )
        pair_scores = output["edge_exists"][
            batch_index,
            tri[0],
            tri[1],
        ]
        requested = min(
            edge_count,
            pair_scores.numel(),
        )
        chosen = torch.topk(
            pair_scores,
            k=requested,
        ).indices
        chosen_left = tri[0, chosen]
        chosen_right = tri[1, chosen]
        hits = positive[
            batch_index,
            chosen_left,
            chosen_right,
        ].sum()
        edge_recalls.append(
            hits.to(torch.float32)
            / max(edge_count, 1)
        )

    offset_distance = torch.linalg.vector_norm(
        output["node_offset"]
        - batch["node_offset"],
        dim=-1,
    )
    node_position_mae_m = _masked_mean(
        offset_distance,
        present,
    ) * (target_size_m / anchor_grid_size)

    metrics = {
        name: float(value.detach())
        for name, value in losses.items()
    }
    metrics["total"] = float(total.detach())
    metrics["anchor_recall"] = float(
        torch.stack(anchor_recalls).mean().detach()
        if anchor_recalls
        else torch.tensor(
            1.0,
            device=total.device,
        )
    )
    metrics["edge_recall"] = float(
        torch.stack(edge_recalls).mean().detach()
        if edge_recalls
        else torch.tensor(
            1.0,
            device=total.device,
        )
    )
    metrics["node_position_mae_m"] = float(
        node_position_mae_m.detach()
    )
    return total, metrics
