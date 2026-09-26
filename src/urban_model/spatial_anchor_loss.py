from __future__ import annotations

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
    shape_points: int,
):
    device = batch["edge_pairs"].device
    size = (
        batch["edge_pairs"].shape[0],
        active_slots,
        active_slots,
    )
    relation = torch.zeros(size, dtype=torch.long, device=device)
    vertical = torch.zeros(size, dtype=torch.long, device=device)
    width = torch.zeros((*size, 1), dtype=torch.float32, device=device)
    shape = torch.zeros(
        (*size, shape_points, 2),
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
        relation[batch_index, left, right] = (
            batch["edge_class"][batch_index, :count] + 1
        )
        vertical[batch_index, left, right] = batch["edge_vertical"][
            batch_index, :count
        ]
        width[batch_index, left, right] = batch["edge_width"][
            batch_index, :count
        ]
        shape[batch_index, left, right] = batch["edge_shape"][
            batch_index, :count
        ]
        positive[batch_index, left, right] = True

    return relation, vertical, width, shape, positive


def spatial_anchor_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    kl_weight: float,
):
    cells = batch["cell_count"].shape[1]
    slots = batch["slot_presence"].shape[2]
    active_slots = output["edge_relation"].shape[1]

    count_weights = torch.ones(
        slots + 1,
        dtype=output["cell_count"].dtype,
        device=output["cell_count"].device,
    )
    count_weights[0] = 0.1
    cell_count = F.cross_entropy(
        output["cell_count"].reshape(-1, slots + 1),
        batch["cell_count"].reshape(-1),
        weight=count_weights,
    )

    occupied_cells = batch["cell_count"].gt(0)
    slot_loss = F.binary_cross_entropy_with_logits(
        output["slot_score"],
        batch["slot_presence"],
        reduction="none",
    )
    slot_score = _masked_mean(
        slot_loss,
        occupied_cells[:, :, None].expand(-1, -1, slots),
    )

    present = batch["slot_presence"].bool()
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
    node_boundary = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output["node_boundary"],
            batch["node_boundary"],
            reduction="none",
        ),
        present,
    )

    relation, vertical, width, shape, positive = _edge_targets(
        batch,
        active_slots,
        output["edge_shape"].shape[-2],
    )
    indexes = torch.arange(active_slots, device=relation.device)
    active = indexes[None] < batch["active_count"][:, None]
    pair_mask = (
        active[:, :, None]
        & active[:, None, :]
        & (indexes[None, :, None] < indexes[None, None, :])
    )

    relation_weights = torch.ones(
        9,
        dtype=output["edge_relation"].dtype,
        device=output["edge_relation"].device,
    )
    relation_weights[0] = 0.01
    relation_values = F.cross_entropy(
        output["edge_relation"].reshape(-1, 9),
        relation.reshape(-1),
        reduction="none",
        weight=relation_weights,
    ).reshape(relation.shape)
    edge_relation = _masked_mean(relation_values, pair_mask)
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
    edge_shape = _masked_mean(
        F.smooth_l1_loss(
            output["edge_shape"],
            shape,
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
        "cell_count": cell_count,
        "slot_score": slot_score,
        "node_offset": node_offset,
        "node_mode": node_mode,
        "node_vertical": node_vertical,
        "node_boundary": node_boundary,
        "edge_relation": edge_relation,
        "edge_vertical": edge_vertical,
        "edge_width": edge_width,
        "edge_shape": edge_shape,
    }
    reconstruction = torch.stack(list(losses.values())).mean()
    total = reconstruction + kl * kl_weight
    metrics = {name: float(value.detach()) for name, value in losses.items()}
    metrics["kl"] = float(kl.detach())
    metrics["reconstruction"] = float(reconstruction.detach())
    return total, metrics
