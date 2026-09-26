from __future__ import annotations

import torch
from torch.nn import functional as F


def _mask(count: torch.Tensor, slots: int) -> torch.Tensor:
    indexes = torch.arange(slots, device=count.device)
    return indexes[None] < count[:, None]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    denominator = weight.expand_as(values).sum().clamp_min(1.0)
    return (values * weight).sum() / denominator


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
    variance_ratio = torch.exp(posterior_logvar - prior_logvar)
    mean_distance = (posterior_mu - prior_mu).square() / torch.exp(prior_logvar)
    values = 0.5 * (
        prior_logvar
        - posterior_logvar
        + variance_ratio
        + mean_distance
        - 1.0
    )
    return values.mean()


def spatial_world_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    max_nodes: int,
    max_edges: int,
    kl_weight: float,
):
    node_mask = _mask(target["node_count"], max_nodes)
    edge_mask = _mask(target["edge_count"], max_edges)

    losses = {
        "node_count": F.cross_entropy(output["node_count"], target["node_count"]),
        "node_xy": _masked_mean(
            F.smooth_l1_loss(
                output["node_xy"],
                target["node_xy"],
                reduction="none",
            ),
            node_mask,
        ),
        "node_mode": _masked_ce(output["node_mode"], target["node_mode"], node_mask),
        "node_vertical": _masked_ce(
            output["node_vertical"],
            target["node_vertical"],
            node_mask,
        ),
        "node_boundary": _masked_mean(
            F.binary_cross_entropy_with_logits(
                output["node_boundary"],
                target["node_boundary"],
                reduction="none",
            ),
            node_mask,
        ),
        "edge_count": F.cross_entropy(output["edge_count"], target["edge_count"]),
        "edge_from": _masked_ce(output["edge_from"], target["edge_from"], edge_mask),
        "edge_to": _masked_ce(output["edge_to"], target["edge_to"], edge_mask),
        "edge_mode": _masked_ce(output["edge_mode"], target["edge_mode"], edge_mask),
        "edge_class": _masked_ce(output["edge_class"], target["edge_class"], edge_mask),
        "edge_vertical": _masked_ce(
            output["edge_vertical"],
            target["edge_vertical"],
            edge_mask,
        ),
        "edge_width": _masked_mean(
            F.smooth_l1_loss(
                output["edge_width"],
                target["edge_width"],
                reduction="none",
            ),
            edge_mask,
        ),
        "edge_shape": _masked_mean(
            F.smooth_l1_loss(
                output["edge_shape"],
                target["edge_shape"],
                reduction="none",
            ),
            edge_mask,
        ),
    }
    kl = _kl(
        output["posterior_mu"],
        output["posterior_logvar"],
        output["prior_mu"],
        output["prior_logvar"],
    )
    reconstruction = torch.stack(list(losses.values())).mean()
    total = reconstruction + kl * kl_weight
    metrics = {
        name: float(value.detach())
        for name, value in losses.items()
    }
    metrics["kl"] = float(kl.detach())
    metrics["reconstruction"] = float(reconstruction.detach())
    return total, metrics
