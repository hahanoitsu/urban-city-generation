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


def _spread_bits(value: torch.Tensor) -> torch.Tensor:
    value = value.to(torch.int64) & 0xFFFF
    value = (value | (value << 8)) & 0x00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F
    value = (value | (value << 2)) & 0x33333333
    value = (value | (value << 1)) & 0x55555555
    return value


def _canonical_order(
    xy: torch.Tensor,
) -> torch.Tensor:
    unit = (
        (xy + 1.0) * 0.5
    ).clamp(
        0.0,
        1.0 - 1e-7,
    )
    quantized = torch.floor(
        unit * 65535.0
    ).to(torch.int64)
    x = _spread_bits(
        quantized[:, 0]
    )
    y = _spread_bits(
        quantized[:, 1]
    )
    morton = x | (y << 1)
    return torch.argsort(
        morton,
        stable=True,
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
    order: torch.Tensor,
):
    device = batch["edge_from"].device
    inverse = torch.empty(
        node_count,
        dtype=torch.long,
        device=device,
    )
    inverse[order] = torch.arange(
        node_count,
        device=device,
    )

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
        source_left = int(
            batch["edge_from"][
                batch_index,
                edge_index,
            ]
        )
        source_right = int(
            batch["edge_to"][
                batch_index,
                edge_index,
            ]
        )
        if (
            source_left == source_right
            or source_left >= node_count
            or source_right >= node_count
        ):
            continue

        left = int(
            inverse[source_left]
        )
        right = int(
            inverse[source_right]
        )
        shape = batch["edge_shape"][
            batch_index,
            edge_index,
        ]
        start = xy[source_left]
        end = xy[source_right]
        if left > right:
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
        inverse,
        exists,
        edge_class,
        edge_vertical,
        edge_width,
        edge_curve,
    )


def _cell_iou(
    predicted_xy: torch.Tensor,
    target_xy: torch.Tensor,
    grid_size: int = 16,
) -> torch.Tensor:
    predicted_unit = (
        (predicted_xy + 1.0) * 0.5
    ).clamp(
        0.0,
        1.0 - 1e-7,
    )
    target_unit = (
        (target_xy + 1.0) * 0.5
    ).clamp(
        0.0,
        1.0 - 1e-7,
    )
    predicted_index = (
        torch.floor(
            predicted_unit[:, 1]
            * grid_size
        ).long()
        * grid_size
        + torch.floor(
            predicted_unit[:, 0]
            * grid_size
        ).long()
    )
    target_index = (
        torch.floor(
            target_unit[:, 1]
            * grid_size
        ).long()
        * grid_size
        + torch.floor(
            target_unit[:, 0]
            * grid_size
        ).long()
    )
    predicted_mask = torch.zeros(
        grid_size * grid_size,
        dtype=torch.bool,
        device=predicted_xy.device,
    )
    target_mask = torch.zeros_like(
        predicted_mask
    )
    predicted_mask[
        torch.unique(predicted_index)
    ] = True
    target_mask[
        torch.unique(target_index)
    ] = True
    intersection = (
        predicted_mask & target_mask
    ).sum()
    union = (
        predicted_mask | target_mask
    ).sum().clamp_min(1)
    return (
        intersection.to(torch.float32)
        / union.to(torch.float32)
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
    batch_size = output[
        "node_xy"
    ].shape[0]

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
    cell_ious = []
    set_chamfers_m = []
    edge_recalls = []
    orders = []

    for batch_index in range(batch_size):
        node_count = int(
            batch["node_count"][
                batch_index
            ]
        )
        target_xy_unsorted = batch[
            "node_xy"
        ][
            batch_index,
            :node_count,
        ]
        order = _canonical_order(
            target_xy_unsorted
        )
        orders.append(order)
        target_xy = target_xy_unsorted[
            order
        ]
        predicted_xy = output[
            "node_xy"
        ][
            batch_index,
            :node_count,
        ]

        node_xy_losses.append(
            F.smooth_l1_loss(
                predicted_xy,
                target_xy,
            )
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
        cell_ious.append(
            _cell_iou(
                predicted_xy,
                target_xy,
            )
        )
        distances = torch.cdist(
            predicted_xy,
            target_xy,
        )
        set_chamfers_m.append(
            (
                distances.min(
                    dim=1
                ).values.mean()
                + distances.min(
                    dim=0
                ).values.mean()
            )
            * 0.25
            * target_size_m
        )

        node_mode_losses.append(
            F.cross_entropy(
                output["node_mode"][
                    batch_index,
                    :node_count,
                ],
                batch["node_mode"][
                    batch_index,
                    :node_count,
                ][order],
            )
        )
        node_vertical_losses.append(
            F.cross_entropy(
                output["node_vertical"][
                    batch_index,
                    :node_count,
                ],
                batch["node_vertical"][
                    batch_index,
                    :node_count,
                ][order],
            )
        )
        node_boundary_losses.append(
            F.binary_cross_entropy_with_logits(
                output["node_boundary"][
                    batch_index,
                    :node_count,
                ],
                batch["node_boundary"][
                    batch_index,
                    :node_count,
                ][order],
            )
        )

        source_degree = _target_degree(
            batch,
            batch_index,
            node_count,
        )
        target_degree = source_degree[
            order
        ].clamp_max(
            output["node_degree"].shape[-1]
            - 1
        )
        node_degree_losses.append(
            F.cross_entropy(
                output["node_degree"][
                    batch_index,
                    :node_count,
                ],
                target_degree,
            )
        )

        (
            _inverse,
            target_exists,
            target_class,
            target_vertical,
            target_width,
            target_curve,
        ) = _edge_targets(
            batch,
            batch_index,
            node_count,
            order,
        )
        predicted_exists = output[
            "edge_exists"
        ][
            batch_index,
            :node_count,
            :node_count,
        ]
        predicted_class = output[
            "edge_class"
        ][
            batch_index,
            :node_count,
            :node_count,
        ]
        predicted_vertical = output[
            "edge_vertical"
        ][
            batch_index,
            :node_count,
            :node_count,
        ]
        predicted_width = output[
            "edge_width"
        ][
            batch_index,
            :node_count,
            :node_count,
        ]
        predicted_curve = output[
            "edge_curve"
        ][
            batch_index,
            :node_count,
            :node_count,
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
        edge_exists_losses.append(
            _balanced_bce(
                predicted_exists[tri],
                target_exists[tri].to(
                    predicted_exists.dtype
                ),
                maximum_positive_weight=80.0,
            )
        )

        positive = target_exists
        if bool(
            positive.any()
        ):
            edge_class_losses.append(
                F.cross_entropy(
                    predicted_class[
                        positive
                    ],
                    target_class[
                        positive
                    ],
                )
            )
            edge_vertical_losses.append(
                F.cross_entropy(
                    predicted_vertical[
                        positive
                    ],
                    target_vertical[
                        positive
                    ],
                )
            )
            edge_width_losses.append(
                F.smooth_l1_loss(
                    predicted_width[
                        positive
                    ],
                    target_width[
                        positive
                    ],
                )
            )
            edge_curve_losses.append(
                F.smooth_l1_loss(
                    predicted_curve[
                        positive
                    ],
                    target_curve[
                        positive
                    ],
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
            zero = (
                predicted_exists.sum()
                * 0.0
            )
            edge_class_losses.append(
                zero
            )
            edge_vertical_losses.append(
                zero
            )
            edge_width_losses.append(
                zero
            )
            edge_curve_losses.append(
                zero
            )
            curve_smooth_losses.append(
                zero
            )

        target_edges = int(
            target_exists.sum()
        )
        if target_edges > 0:
            pair_scores = predicted_exists[
                tri
            ]
            requested = min(
                target_edges,
                int(
                    pair_scores.numel()
                ),
            )
            chosen = torch.topk(
                pair_scores,
                k=requested,
            ).indices
            flat_target = target_exists[
                tri
            ]
            edge_recalls.append(
                flat_target[
                    chosen
                ]
                .to(
                    torch.float32
                )
                .sum()
                / target_edges
            )

    losses = {
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
        "node_xy": 5.0,
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
        losses[name]
        * weights[name]
        for name in losses
    ) / sum(
        weights.values()
    )

    metrics = {
        name: float(
            value.detach()
        )
        for name, value
        in losses.items()
    }
    metrics["total"] = float(
        total.detach()
    )
    metrics[
        "node_position_mae_m"
    ] = float(
        torch.stack(
            node_errors_m
        ).mean().detach()
    )
    metrics[
        "node_cell_iou"
    ] = float(
        torch.stack(
            cell_ious
        ).mean().detach()
    )
    metrics[
        "set_chamfer_m"
    ] = float(
        torch.stack(
            set_chamfers_m
        ).mean().detach()
    )
    metrics[
        "edge_recall"
    ] = float(
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
        orders,
    )
