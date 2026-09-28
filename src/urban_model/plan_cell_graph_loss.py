from __future__ import annotations

import torch
from torch.nn import functional as F


def _balanced_bce(
    logits: torch.Tensor, target: torch.Tensor, *, maximum_positive_weight: float = 80.0
) -> torch.Tensor:
    positives = target.sum()
    negatives = target.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, maximum_positive_weight)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(values)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def _spread_bits(value: torch.Tensor) -> torch.Tensor:
    value = value.to(torch.int64) & 0xFFFF
    value = (value | (value << 8)) & 0x00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F
    value = (value | (value << 2)) & 0x33333333
    value = (value | (value << 1)) & 0x55555555
    return value


def canonical_cell_order(xy: torch.Tensor, grid_size: int) -> torch.Tensor:
    unit = ((xy + 1.0) * 0.5).clamp(0.0, 1.0 - 1e-7)
    column = torch.floor(unit[:, 0] * grid_size).long().clamp(0, grid_size - 1)
    row = torch.floor(unit[:, 1] * grid_size).long().clamp(0, grid_size - 1)
    cell = row * grid_size + column
    local_x = (unit[:, 0] * grid_size - column).clamp(0.0, 1.0 - 1e-7)
    local_y = (unit[:, 1] * grid_size - row).clamp(0.0, 1.0 - 1e-7)
    quantized_x = torch.floor(local_x * 65535.0).to(torch.int64)
    quantized_y = torch.floor(local_y * 65535.0).to(torch.int64)
    morton = _spread_bits(quantized_x) | (_spread_bits(quantized_y) << 1)
    key = (cell.to(torch.int64) << 33) | morton
    return torch.argsort(key, stable=True)


def _target_degree(
    batch: dict[str, torch.Tensor], batch_index: int, node_count: int
) -> torch.Tensor:
    degree = torch.zeros(node_count, dtype=torch.long, device=batch["edge_from"].device)
    edge_count = int(batch["edge_count"][batch_index])
    for edge_index in range(edge_count):
        left = int(batch["edge_from"][batch_index, edge_index])
        right = int(batch["edge_to"][batch_index, edge_index])
        if left == right or left >= node_count or right >= node_count:
            continue
        degree[left] += 1
        degree[right] += 1
    return degree


def _edge_targets(
    batch: dict[str, torch.Tensor],
    batch_index: int,
    node_count: int,
    order: torch.Tensor,
    curve_dimensions: int = 2,
):
    device = batch["edge_from"].device
    inverse = torch.empty(node_count, dtype=torch.long, device=device)
    inverse[order] = torch.arange(node_count, device=device)
    exists = torch.zeros(node_count, node_count, dtype=torch.bool, device=device)
    edge_class = torch.zeros(node_count, node_count, dtype=torch.long, device=device)
    edge_vertical = torch.zeros(node_count, node_count, dtype=torch.long, device=device)
    edge_width = torch.zeros(node_count, node_count, 1, dtype=torch.float32, device=device)
    curve_points = batch["edge_shape"].shape[-2]
    shape = (node_count, node_count, curve_points)
    if curve_dimensions == 2:
        shape = (*shape, 2)
    edge_curve = torch.zeros(shape, dtype=torch.float32, device=device)

    edge_count = int(batch["edge_count"][batch_index])
    xy = batch["node_xy"][batch_index, :node_count]
    for edge_index in range(edge_count):
        source_left = int(batch["edge_from"][batch_index, edge_index])
        source_right = int(batch["edge_to"][batch_index, edge_index])
        if source_left == source_right or source_left >= node_count or source_right >= node_count:
            raise ValueError(
                "Plan graph targets require valid, distinct endpoints; rebuild with simple_graph=True"
            )

        left = int(inverse[source_left])
        right = int(inverse[source_right])
        shape = batch["edge_shape"][batch_index, edge_index]
        start = xy[source_left]
        end = xy[source_right]
        if left > right:
            left, right = (right, left)
            start, end = (end, start)
            shape = torch.flip(shape, dims=[0])

        if exists[left, right]:
            raise ValueError("Parallel target edges must be subdivided before training")
        curve = shape
        if curve_dimensions == 1:
            chord = end - start
            length = torch.linalg.vector_norm(chord).clamp_min(1e-4)
            normal = torch.stack([-chord[1], chord[0]]) / length
            curve = (shape * normal[None]).sum(dim=-1) / length
        exists[left, right] = True
        edge_class[left, right] = batch["edge_class"][batch_index, edge_index]
        edge_vertical[left, right] = batch["edge_vertical"][batch_index, edge_index]
        edge_width[left, right] = batch["edge_width"][batch_index, edge_index]
        edge_curve[left, right] = curve

    return (exists, edge_class, edge_vertical, edge_width, edge_curve)


def plan_cell_graph_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    target_size_m: float,
    grid_size: int,
    geometry_scale_m: float | None = None,
):
    if geometry_scale_m is not None and geometry_scale_m <= 0:
        raise ValueError("geometry_scale_m must be positive")
    batch_size = output["node_xy"].shape[0]
    device = output["node_xy"].device
    losses = {
        "node_local": [],
        "node_mode": [],
        "node_vertical": [],
        "node_boundary": [],
        "node_degree": [],
        "edge_exists": [],
        "edge_class": [],
        "edge_vertical": [],
        "edge_width": [],
        "edge_curve": [],
        "curve_smooth": [],
    }
    metrics = {
        "node_position_mae_m": [],
        "set_chamfer_m": [],
        "edge_recall": [],
        "curve_mae_m": [],
        "edge_shape_mae_m": [],
    }
    orders = []

    for batch_index in range(batch_size):
        node_count = int(batch["node_count"][batch_index])
        if node_count < 2:
            raise ValueError("Plan graph training requires at least two nodes")
        target_xy_unsorted = batch["node_xy"][batch_index, :node_count]
        order = canonical_cell_order(target_xy_unsorted, grid_size)
        orders.append(order)
        target_xy = target_xy_unsorted[order]
        predicted_xy = output["node_xy"][batch_index, :node_count]

        target_unit = (target_xy + 1.0) * 0.5
        target_column = torch.floor(target_unit[:, 0] * grid_size).long().clamp(0, grid_size - 1)
        target_row = torch.floor(target_unit[:, 1] * grid_size).long().clamp(0, grid_size - 1)
        target_local = torch.stack(
            [
                target_unit[:, 0] * grid_size - target_column,
                target_unit[:, 1] * grid_size - target_row,
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        if geometry_scale_m is None:
            position_loss = F.smooth_l1_loss(
                output["node_local"][batch_index, :node_count], target_local
            )
        else:
            position_loss = F.smooth_l1_loss(
                predicted_xy.float() * (target_size_m / 2 / geometry_scale_m),
                target_xy.float() * (target_size_m / 2 / geometry_scale_m),
                beta=1.0 / geometry_scale_m,
            )
        losses["node_local"].append(position_loss)
        metrics["node_position_mae_m"].append(
            torch.linalg.vector_norm(predicted_xy - target_xy, dim=-1).mean()
            * (target_size_m / 2.0)
        )
        distances = torch.cdist(predicted_xy, target_xy)
        metrics["set_chamfer_m"].append(
            (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean())
            * 0.25
            * target_size_m
        )

        losses["node_mode"].append(
            F.cross_entropy(
                output["node_mode"][batch_index, :node_count],
                batch["node_mode"][batch_index, :node_count][order],
            )
        )
        losses["node_vertical"].append(
            F.cross_entropy(
                output["node_vertical"][batch_index, :node_count],
                batch["node_vertical"][batch_index, :node_count][order],
            )
        )
        losses["node_boundary"].append(
            F.binary_cross_entropy_with_logits(
                output["node_boundary"][batch_index, :node_count],
                batch["node_boundary"][batch_index, :node_count][order],
            )
        )
        degree = _target_degree(batch, batch_index, node_count)[order].clamp_max(
            output["node_degree"].shape[-1] - 1
        )
        losses["node_degree"].append(
            F.cross_entropy(output["node_degree"][batch_index, :node_count], degree)
        )

        (target_exists, target_class, target_vertical, target_width, target_curve) = _edge_targets(
            batch,
            batch_index,
            node_count,
            order,
            curve_dimensions=2 if output["edge_curve"].ndim == 5 else 1,
        )
        predicted_exists = output["edge_exists"][batch_index, :node_count, :node_count]
        predicted_class = output["edge_class"][batch_index, :node_count, :node_count]
        predicted_vertical = output["edge_vertical"][batch_index, :node_count, :node_count]
        predicted_width = output["edge_width"][batch_index, :node_count, :node_count]
        predicted_curve = output["edge_curve"][batch_index, :node_count, :node_count]
        tri = torch.triu(
            torch.ones(node_count, node_count, dtype=torch.bool, device=device), diagonal=1
        )
        losses["edge_exists"].append(
            _balanced_bce(predicted_exists[tri], target_exists[tri].to(predicted_exists.dtype))
        )
        positive = target_exists
        if bool(positive.any()):
            losses["edge_class"].append(
                F.cross_entropy(predicted_class[positive], target_class[positive])
            )
            losses["edge_vertical"].append(
                F.cross_entropy(predicted_vertical[positive], target_vertical[positive])
            )
            losses["edge_width"].append(
                F.smooth_l1_loss(predicted_width[positive], target_width[positive])
            )
            curve = predicted_curve[positive].float()
            target = target_curve[positive].float()
            scale = target_size_m / 2 / geometry_scale_m if geometry_scale_m is not None else 1.0
            beta = 1.0 / geometry_scale_m if geometry_scale_m is not None else 1.0
            if geometry_scale_m is not None and curve.ndim != 3:
                raise ValueError("Metric geometry loss requires x/y curve residuals")
            losses["edge_curve"].append(F.smooth_l1_loss(curve * scale, target * scale, beta=beta))
            if curve.ndim == 3:
                metrics["curve_mae_m"].append(
                    torch.linalg.vector_norm(curve - target, dim=-1).mean() * target_size_m / 2
                )
                left, right = positive.nonzero(as_tuple=True)
                fraction = torch.arange(1, curve.shape[1] + 1, device=device)[None, :, None] / (
                    curve.shape[1] + 1
                )
                points = (
                    predicted_xy[left, None] * (1 - fraction)
                    + predicted_xy[right, None] * fraction
                    + curve
                )
                target_points = (
                    target_xy[left, None] * (1 - fraction)
                    + target_xy[right, None] * fraction
                    + target
                )
                metrics["edge_shape_mae_m"].append(
                    torch.linalg.vector_norm(points - target_points, dim=-1).mean()
                    * target_size_m
                    / 2
                )
            axis = -2 if curve.ndim == 3 else -1
            if curve.shape[axis] >= 3:
                # Match real bends instead of penalising every bend towards a straight line.
                second = torch.diff(curve, n=2, dim=axis)
                target_second = torch.diff(target, n=2, dim=axis)
                losses["curve_smooth"].append(
                    F.smooth_l1_loss(second * scale, target_second * scale, beta=beta)
                )
            else:
                losses["curve_smooth"].append(curve.sum() * 0)
        else:
            zero = predicted_exists.sum() * 0.0
            for name in ("edge_class", "edge_vertical", "edge_width", "edge_curve", "curve_smooth"):
                losses[name].append(zero)

        target_edge_count = int(target_exists.sum())
        if target_edge_count > 0:
            pair_scores = predicted_exists[tri]
            requested = min(target_edge_count, int(pair_scores.numel()))
            chosen = torch.topk(pair_scores, k=requested).indices
            metrics["edge_recall"].append(
                target_exists[tri][chosen].to(torch.float32).sum() / target_edge_count
            )

    reduced = {name: torch.stack(values).mean() for name, values in losses.items()}
    weights = {
        "node_local": 5.0,
        "node_mode": 0.5,
        "node_vertical": 0.35,
        "node_boundary": 0.35,
        "node_degree": 0.75,
        "edge_exists": 2.0,
        "edge_class": 0.8,
        "edge_vertical": 0.35,
        "edge_width": 0.35,
        "edge_curve": 1.0,
        "curve_smooth": 0.05,
    }
    total = sum(reduced[name] * weights[name] for name in reduced) / sum(weights.values())
    result = {name: float(value.detach()) for name, value in reduced.items()}
    result["total"] = float(total.detach())
    for name, values in metrics.items():
        if values:
            result[name] = float(torch.stack(values).mean().detach())
        else:
            result[name] = 1.0
    return (total, result, orders)
