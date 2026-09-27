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
    return (
        values * weight
    ).sum() / weight.sum().clamp_min(1.0)


def _balanced_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    maximum_positive_weight: float = 100.0,
) -> torch.Tensor:
    positives = target.sum()
    negatives = target.numel() - positives
    positive_weight = (
        negatives / positives.clamp_min(1.0)
    ).clamp(1.0, maximum_positive_weight)
    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=positive_weight,
    )


def _match_queries(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    batch_index: int,
    *,
    candidates: int = 24,
) -> torch.Tensor:
    count = int(batch["node_count"][batch_index])
    predicted_xy = output["node_xy"][batch_index]
    target_xy = batch["node_xy"][
        batch_index,
        :count,
    ]
    cost = torch.cdist(
        predicted_xy,
        target_xy,
        p=1,
    )
    mode_log = F.log_softmax(
        output["node_mode"][batch_index],
        dim=-1,
    )
    target_mode = batch["node_mode"][
        batch_index,
        :count,
    ]
    cost = (
        cost * 4.0
        - mode_log[:, target_mode] * 0.35
        - torch.sigmoid(
            output["node_presence"][batch_index]
        )[:, None]
        * 0.1
    )

    keep = min(
        candidates,
        cost.shape[0],
    )
    values, indexes = torch.topk(
        cost,
        k=keep,
        dim=0,
        largest=False,
    )
    values_cpu = values.detach().cpu()
    indexes_cpu = indexes.detach().cpu()
    proposals = []
    for target in range(count):
        for rank in range(keep):
            proposals.append(
                (
                    float(
                        values_cpu[
                            rank,
                            target,
                        ]
                    ),
                    target,
                    int(
                        indexes_cpu[
                            rank,
                            target,
                        ]
                    ),
                )
            )
    proposals.sort(
        key=lambda value: value[0]
    )

    assigned = [-1] * count
    used = set()
    remaining = count
    for _cost, target, query in proposals:
        if (
            assigned[target] >= 0
            or query in used
        ):
            continue
        assigned[target] = query
        used.add(query)
        remaining -= 1
        if remaining == 0:
            break

    if remaining:
        available = torch.ones(
            cost.shape[0],
            dtype=torch.bool,
            device=cost.device,
        )
        if used:
            available[
                torch.tensor(
                    sorted(used),
                    device=cost.device,
                )
            ] = False
        for target in range(count):
            if assigned[target] >= 0:
                continue
            values = cost[:, target].masked_fill(
                ~available,
                float("inf"),
            )
            query = int(
                values.argmin().item()
            )
            assigned[target] = query
            available[query] = False

    return torch.tensor(
        assigned,
        dtype=torch.long,
        device=cost.device,
    )


def _target_degree(
    batch: dict[str, torch.Tensor],
    batch_index: int,
    node_count: int,
) -> torch.Tensor:
    degree = torch.zeros(
        node_count,
        dtype=torch.long,
        device=batch["edge_from"].device,
    )
    edge_count = int(
        batch["edge_count"][batch_index]
    )
    for edge_index in range(edge_count):
        left = int(
            batch["edge_from"][
                batch_index,
                edge_index,
            ]
        )
        right = int(
            batch["edge_to"][
                batch_index,
                edge_index,
            ]
        )
        if (
            left == right
            or left >= node_count
            or right >= node_count
        ):
            continue
        degree[left] += 1
        degree[right] += 1
    return degree


def _edge_targets(
    batch: dict[str, torch.Tensor],
    batch_index: int,
    node_count: int,
):
    device = batch["edge_from"].device
    exists = torch.zeros(
        node_count,
        node_count,
        dtype=torch.bool,
        device=device,
    )
    edge_class = torch.zeros(
        node_count,
        node_count,
        dtype=torch.long,
        device=device,
    )
    edge_vertical = torch.zeros(
        node_count,
        node_count,
        dtype=torch.long,
        device=device,
    )
    edge_width = torch.zeros(
        node_count,
        node_count,
        1,
        dtype=torch.float32,
        device=device,
    )
    curve_points = batch[
        "edge_shape"
    ].shape[-2]
    edge_curve = torch.zeros(
        node_count,
        node_count,
        curve_points,
        dtype=torch.float32,
        device=device,
    )

    edge_count = int(
        batch["edge_count"][batch_index]
    )
    xy = batch["node_xy"][
        batch_index,
        :node_count,
    ]
    for edge_index in range(edge_count):
        left = int(
            batch["edge_from"][
                batch_index,
                edge_index,
            ]
        )
        right = int(
            batch["edge_to"][
                batch_index,
                edge_index,
            ]
        )
        if (
            left == right
            or left >= node_count
            or right >= node_count
        ):
            continue

        shape = batch["edge_shape"][
            batch_index,
            edge_index,
        ]
        start = xy[left]
        end = xy[right]
        reverse = left > right
        if reverse:
            left, right = right, left
            start, end = end, start
            shape = torch.flip(
                shape,
                dims=[0],
            )

        chord = end - start
        length = torch.linalg.vector_norm(
            chord
        ).clamp_min(1e-4)
        normal = torch.stack(
            [-chord[1], chord[0]]
        ) / length
        curve = (
            shape * normal[None]
        ).sum(dim=-1) / length
        exists[left, right] = True
        edge_class[left, right] = batch[
            "edge_class"
        ][batch_index, edge_index]
        edge_vertical[left, right] = batch[
            "edge_vertical"
        ][batch_index, edge_index]
        edge_width[left, right] = batch[
            "edge_width"
        ][batch_index, edge_index]
        edge_curve[left, right] = curve

    return (
        exists,
        edge_class,
        edge_vertical,
        edge_width,
        edge_curve,
    )


def plan_set_graph_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    target_size_m: float,
) -> tuple[
    torch.Tensor,
    dict[str, float],
    list[torch.Tensor],
]:
    device = output["node_xy"].device
    batch_size = output["node_xy"].shape[0]
    query_count = output[
        "node_xy"
    ].shape[1]

    presence_target = torch.zeros(
        batch_size,
        query_count,
        dtype=torch.float32,
        device=device,
    )
    node_xy_losses = []
    node_mode_losses = []
    node_vertical_losses = []
    node_boundary_losses = []
    node_degree_losses = []
    edge_exists_losses = []
    edge_class_losses = []
    edge_vertical_losses = []
    edge_width_losses = []
    edge_curve_losses = []
    curve_smooth_losses = []
    node_errors_m = []
    query_recalls = []
    set_chamfers_m = []
    edge_recalls = []
    assignments = []

    for batch_index in range(batch_size):
        node_count = int(
            batch["node_count"][batch_index]
        )
        query_ids = _match_queries(
            output,
            batch,
            batch_index,
        )
        assignments.append(query_ids)
        presence_target[
            batch_index,
            query_ids,
        ] = 1.0

        predicted_xy = output[
            "node_xy"
        ][batch_index, query_ids]
        target_xy = batch["node_xy"][
            batch_index,
            :node_count,
        ]
        xy_error = F.smooth_l1_loss(
            predicted_xy,
            target_xy,
            reduction="none",
        )
        node_xy_losses.append(
            xy_error.mean()
        )
        node_errors_m.append(
            torch.linalg.vector_norm(
                predicted_xy - target_xy,
                dim=-1,
            ).mean()
            * (
                target_size_m
                / 2.0
            )
        )

        node_mode_losses.append(
            F.cross_entropy(
                output["node_mode"][
                    batch_index,
                    query_ids,
                ],
                batch["node_mode"][
                    batch_index,
                    :node_count,
                ],
            )
        )
        node_vertical_losses.append(
            F.cross_entropy(
                output["node_vertical"][
                    batch_index,
                    query_ids,
                ],
                batch["node_vertical"][
                    batch_index,
                    :node_count,
                ],
            )
        )
        node_boundary_losses.append(
            F.binary_cross_entropy_with_logits(
                output["node_boundary"][
                    batch_index,
                    query_ids,
                ],
                batch["node_boundary"][
                    batch_index,
                    :node_count,
                ],
            )
        )
        degree = _target_degree(
            batch,
            batch_index,
            node_count,
        ).clamp_max(
            output["node_degree"].shape[-1]
            - 1
        )
        node_degree_losses.append(
            F.cross_entropy(
                output["node_degree"][
                    batch_index,
                    query_ids,
                ],
                degree,
            )
        )

        (
            target_exists,
            target_class,
            target_vertical,
            target_width,
            target_curve,
        ) = _edge_targets(
            batch,
            batch_index,
            node_count,
        )
        predicted_exists = output[
            "edge_exists"
        ][batch_index][
            query_ids[:, None],
            query_ids[None, :],
        ]
        predicted_class = output[
            "edge_class"
        ][batch_index][
            query_ids[:, None],
            query_ids[None, :],
        ]
        predicted_vertical = output[
            "edge_vertical"
        ][batch_index][
            query_ids[:, None],
            query_ids[None, :],
        ]
        predicted_width = output[
            "edge_width"
        ][batch_index][
            query_ids[:, None],
            query_ids[None, :],
        ]
        predicted_curve = output[
            "edge_curve"
        ][batch_index][
            query_ids[:, None],
            query_ids[None, :],
        ]

        tri = torch.triu(
            torch.ones(
                node_count,
                node_count,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )
        tri_target = target_exists[tri].to(
            predicted_exists.dtype
        )
        edge_exists_losses.append(
            _balanced_bce(
                predicted_exists[tri],
                tri_target,
                maximum_positive_weight=80.0,
            )
        )

        positive = target_exists
        if bool(positive.any()):
            edge_class_losses.append(
                F.cross_entropy(
                    predicted_class[positive],
                    target_class[positive],
                )
            )
            edge_vertical_losses.append(
                F.cross_entropy(
                    predicted_vertical[positive],
                    target_vertical[positive],
                )
            )
            edge_width_losses.append(
                F.smooth_l1_loss(
                    predicted_width[positive],
                    target_width[positive],
                )
            )
            edge_curve_losses.append(
                F.smooth_l1_loss(
                    predicted_curve[positive],
                    target_curve[positive],
                )
            )
            second = (
                predicted_curve[
                    positive
                ][..., 2:]
                - 2.0
                * predicted_curve[
                    positive
                ][..., 1:-1]
                + predicted_curve[
                    positive
                ][..., :-2]
            )
            curve_smooth_losses.append(
                second.square().mean()
            )
        else:
            zero = predicted_exists.sum() * 0.0
            edge_class_losses.append(zero)
            edge_vertical_losses.append(zero)
            edge_width_losses.append(zero)
            edge_curve_losses.append(zero)
            curve_smooth_losses.append(zero)

        selected = torch.topk(
            output["node_presence"][
                batch_index
            ],
            k=node_count,
        ).indices
        matched_set = torch.zeros(
            query_count,
            dtype=torch.bool,
            device=device,
        )
        matched_set[query_ids] = True
        query_recalls.append(
            matched_set[selected]
            .to(torch.float32)
            .mean()
        )

        selected_xy = output[
            "node_xy"
        ][batch_index, selected]
        distances = torch.cdist(
            selected_xy,
            target_xy,
        )
        chamfer = (
            distances.min(dim=1).values.mean()
            + distances.min(dim=0).values.mean()
        ) * 0.5
        set_chamfers_m.append(
            chamfer
            * (
                target_size_m
                / 2.0
            )
        )

        pair_scores = predicted_exists[tri]
        target_edges = int(
            target_exists.sum()
        )
        if target_edges > 0:
            requested = min(
                target_edges,
                int(pair_scores.numel()),
            )
            chosen = torch.topk(
                pair_scores,
                k=requested,
            ).indices
            target_flat = target_exists[tri]
            edge_recalls.append(
                target_flat[chosen]
                .to(torch.float32)
                .sum()
                / target_edges
            )

    losses = {
        "presence": _balanced_bce(
            output["node_presence"],
            presence_target,
            maximum_positive_weight=100.0,
        ),
        "node_xy": torch.stack(
            node_xy_losses
        ).mean(),
        "node_mode": torch.stack(
            node_mode_losses
        ).mean(),
        "node_vertical": torch.stack(
            node_vertical_losses
        ).mean(),
        "node_boundary": torch.stack(
            node_boundary_losses
        ).mean(),
        "node_degree": torch.stack(
            node_degree_losses
        ).mean(),
        "edge_exists": torch.stack(
            edge_exists_losses
        ).mean(),
        "edge_class": torch.stack(
            edge_class_losses
        ).mean(),
        "edge_vertical": torch.stack(
            edge_vertical_losses
        ).mean(),
        "edge_width": torch.stack(
            edge_width_losses
        ).mean(),
        "edge_curve": torch.stack(
            edge_curve_losses
        ).mean(),
        "curve_smooth": torch.stack(
            curve_smooth_losses
        ).mean(),
    }
    weights = {
        "presence": 1.5,
        "node_xy": 4.0,
        "node_mode": 0.5,
        "node_vertical": 0.35,
        "node_boundary": 0.35,
        "node_degree": 0.5,
        "edge_exists": 2.0,
        "edge_class": 0.8,
        "edge_vertical": 0.35,
        "edge_width": 0.35,
        "edge_curve": 1.0,
        "curve_smooth": 0.05,
    }
    total = sum(
        losses[name] * weights[name]
        for name in losses
    ) / sum(weights.values())

    metrics = {
        name: float(value.detach())
        for name, value in losses.items()
    }
    metrics["total"] = float(
        total.detach()
    )
    metrics["node_position_mae_m"] = float(
        torch.stack(
            node_errors_m
        ).mean().detach()
    )
    metrics["query_recall"] = float(
        torch.stack(
            query_recalls
        ).mean().detach()
    )
    metrics["set_chamfer_m"] = float(
        torch.stack(
            set_chamfers_m
        ).mean().detach()
    )
    metrics["edge_recall"] = float(
        torch.stack(
            edge_recalls
        ).mean().detach()
        if edge_recalls
        else torch.tensor(
            1.0,
            device=device,
        )
    )
    return (
        total,
        metrics,
        assignments,
    )
