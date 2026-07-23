from __future__ import annotations

import torch
from torch.nn import functional as F

from .codec import OP_ADD, OP_CONNECT, OP_PAD, OP_ROOT


def _masked_loss(logits: torch.Tensor, encoded_targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    targets = encoded_targets[mask] - 1
    return F.cross_entropy(logits[mask], targets)


def graph_program_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    op_targets = targets["op"]
    op_loss = F.cross_entropy(
        logits["op"].reshape(-1, logits["op"].shape[-1]),
        op_targets.reshape(-1),
        ignore_index=OP_PAD,
    )
    root = op_targets.eq(OP_ROOT)
    add = op_targets.eq(OP_ADD)
    connect = op_targets.eq(OP_CONNECT)
    coordinate = root | add
    edge = add | connect
    losses = {
        "op": op_loss,
        "x": _masked_loss(logits["x"], targets["x"], coordinate),
        "y": _masked_loss(logits["y"], targets["y"], coordinate),
        "id1": _masked_loss(logits["id1"], targets["id1"], edge),
        "id2": _masked_loss(logits["id2"], targets["id2"], connect),
        "mode": _masked_loss(logits["mode"], targets["mode"], root | edge),
        "class": _masked_loss(logits["class"], targets["class"], edge),
        "width": _masked_loss(logits["width"], targets["width"], edge),
        "vertical": _masked_loss(logits["vertical"], targets["vertical"], root | edge),
        "layer": _masked_loss(logits["layer"], targets["layer"], root | edge),
    }
    total = torch.stack(list(losses.values())).mean()
    return total, {name: float(value.detach()) for name, value in losses.items()}
