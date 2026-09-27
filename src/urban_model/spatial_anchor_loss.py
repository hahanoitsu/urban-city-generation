from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    expanded = weight.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def _masked_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    classes = logits.shape[-1]
    values = F.cross_entropy(
        logits.reshape(-1, classes),
        target.reshape(-1),
        reduction="none",
    ).reshape(target.shape)
    return _masked_mean(values, mask)


def _balanced_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    maximum_positive_weight: float = 40.0,
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    valid_target = target[mask]
    positives = valid_target.sum()
    negatives = valid_target.numel() - positives
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


def _count_fraction(count: torch.Tensor, maximum: int) -> torch.Tensor:
    return torch.log1p(count.to(torch.float32)) / math.log1p(maximum)


def _kl(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    posterior_logvar = posterior_logvar.clamp(-10.0, 10.0)
    prior_logvar = prior_logvar.clamp(-10.0, 10.0)
    values = 0.5 * (
        prior_logvar
        - posterior_logvar
        + torch.exp(posterior_logvar - prior_logvar)
        + (posterior_mu - prior_mu).square() / torch.exp(prior_logvar)
        - 1.0
    )
    return values.mean()


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
    edge_class = torch.zeros(size, dtype=torch.long, device=device)
    vertical = torch.zeros(size, dtype=torch.long, device=device)
    width = torch.zeros((*size, 1), dtype=torch.float32, device=device)
    curve = torch.zeros(
        (*size, curve_points),
        dtype=torch.float32,
        device=device,
    )
    positive = torch.zeros(size, dtype=torch.bool, device=device)

    for batch_index in range(size[0]):
        count = int(batch["edge_count"][batch_index])
        if count == 0:
            continue
        pairs = batch["edge_pairs"][batch_index, :count]
        left = pairs[:, 0]
        right = pairs[:, 1]
        edge_class[batch_index, left, right] = batch["edge_class"][
            batch_index, :count
        ]
        vertical[batch_index, left, right] = batch["edge_vertical"][
            batch_index, :count
        ]
        width[batch_index, left, right] = batch["edge_width"][
            batch_index, :count
        ]
        curve[batch_index, left, right] = batch["edge_curve"][
            batch_index, :count
        ]
        positive[batch_index, left, right] = True

    return edge_class, vertical, width, curve, positive


def spatial_anchor_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    kl_weight: float,
    max_active_nodes: int,
    max_edges: int,
):
    slots = batch["slot_presence"].shape[2]
    occupied_cells = batch["cell_count"].gt(0)
    present = batch["slot_presence"].bool()

    node_count = F.smooth_l1_loss(
        torch.sigmoid(output["global_node_count"]),
        _count_fraction(batch["active_count"], max_active_nodes),
    )
    edge_count = F.smooth_l1_loss(
        torch.sigmoid(output["global_edge_count"]),
        _count_fraction(batch["edge_count"], max_edges),
    )
    cell_occupancy = _balanced_bce(
        output["cell_occupancy"],
        occupied_cells.to(output["cell_occupancy"].dtype),
    )
    cell_count = _masked_ce(
        output["cell_count"],
        (batch["cell_count"] - 1).clamp_min(0),
        occupied_cells,
    )
    slot_score = _balanced_bce(
        output["slot_score"],
        batch["slot_presence"],
        occupied_cells[:, :, None].expand(-1, -1, slots),
        maximum_positive_weight=12.0,
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

    active_indexes = torch.arange(
        output["node_degree"].shape[1],
        device=output["node_degree"].device,
    )
    active_node_mask = active_indexes[None] < batch["active_count"][:, None]
    node_degree = _masked_ce(
        output["node_degree"],
        batch["node_degree"].clamp_max(output["node_degree"].shape[-1] - 1),
        active_node_mask,
    )

    edge_class_target, vertical, width, curve, positive = _edge_targets(
        batch,
        output["edge_exists"].shape[1],
        output["edge_curve"].shape[-1],
    )
    indexes = torch.arange(
        output["edge_exists"].shape[1],
        device=output["edge_exists"].device,
    )
    active = indexes[None] < batch["active_count"][:, None]
    pair_mask = (
        active[:, :, None]
        & active[:, None, :]
        & (indexes[None, :, None] < indexes[None, None, :])
    )

    edge_exists = _balanced_bce(
        output["edge_exists"],
        positive.to(output["edge_exists"].dtype),
        pair_mask,
        maximum_positive_weight=60.0,
    )
    edge_class = _masked_ce(
        output["edge_class"],
        edge_class_target,
        positive,
    )
    edge_vertical = _masked_ce(
        output["edge_vertical"],
        vertical,
        positive,
    )
    edge_width = _masked_mean(
        F.smooth_l1_loss(
            output["edge_width"],
            width,
            reduction="none",
        ),
        positive,
    )
    edge_curve = _masked_mean(
        F.smooth_l1_loss(
            output["edge_curve"],
            curve,
            reduction="none",
        ),
        positive,
    )

    kl = _kl(
        output["posterior_mu"],
        output["posterior_logvar"],
        output["prior_mu"],
        output["prior_logvar"],
    )

    losses = {
        "node_count": node_count,
        "edge_count": edge_count,
        "cell_occupancy": cell_occupancy,
        "cell_count": cell_count,
        "slot_score": slot_score,
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
    }
    reconstruction = torch.stack(list(losses.values())).mean()
    total = reconstruction + kl * kl_weight
    metrics = {name: float(value.detach()) for name, value in losses.items()}
    predicted_nodes = torch.expm1(
        torch.sigmoid(output["global_node_count"])
        * math.log1p(max_active_nodes)
    )
    predicted_edges = torch.expm1(
        torch.sigmoid(output["global_edge_count"])
        * math.log1p(max_edges)
    )
    metrics["node_count_mae"] = float(
        (predicted_nodes - batch["active_count"]).abs().mean().detach()
    )
    metrics["edge_count_mae"] = float(
        (predicted_edges - batch["edge_count"]).abs().mean().detach()
    )
    scores = (
        output["cell_occupancy"][:, :, None] + output["slot_score"]
    ).reshape(output["slot_score"].shape[0], -1)
    target_presence = batch["slot_presence"].reshape(
        batch["slot_presence"].shape[0],
        -1,
    )
    recalls = []
    for batch_index in range(scores.shape[0]):
        count = int(batch["active_count"][batch_index])
        if count <= 0:
            continue
        chosen = torch.topk(scores[batch_index], k=count).indices
        recalls.append(target_presence[batch_index, chosen].mean())
    metrics["anchor_recall_at_target_count"] = float(
        torch.stack(recalls).mean().detach()
        if recalls
        else torch.tensor(1.0, device=scores.device)
    )
    metrics["kl"] = float(kl.detach())
    metrics["reconstruction"] = float(reconstruction.detach())
    return total, metrics
