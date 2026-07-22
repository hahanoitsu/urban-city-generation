from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import LossConfig


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _multiclass_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)[:, 1:]
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)[:, 1:]
    one_hot = one_hot.to(dtype=probability.dtype)
    mask = mask.to(dtype=probability.dtype).unsqueeze(1)
    probability = probability * mask
    one_hot = one_hot * mask
    dimensions = (0, 2, 3)
    intersection = (probability * one_hot).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + one_hot.sum(dim=dimensions)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    mask = mask.to(dtype=probability.dtype).unsqueeze(1)
    probability = probability * mask
    target = target * mask
    dimensions = (0, 2, 3)
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    expanded_mask = mask.to(dtype=probability.dtype).unsqueeze(1)
    probability = probability * expanded_mask
    target = target * expanded_mask
    dimensions = (0, 2, 3)
    true_positive = (probability * target).sum(dim=dimensions)
    false_positive = (probability * (1.0 - target)).sum(dim=dimensions)
    false_negative = ((1.0 - probability) * target).sum(dim=dimensions)
    score = (true_positive + 1.0) / (
        true_positive + alpha * false_positive + beta * false_negative + 1.0
    )
    return (1.0 - score).mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("road_class_weights", torch.tensor(config.road_class_weights))
        self.register_buffer("landuse_class_weights", torch.tensor(config.landuse_class_weights))
        self.register_buffer(
            "binary_positive_weights",
            torch.tensor(config.binary_positive_weights).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "height_confidence_weights", torch.tensor(config.height_confidence_weights)
        )
        self.register_buffer(
            "centerline_positive_weight", torch.tensor(config.centerline_positive_weight)
        )

    def forward(
        self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        valid = batch["valid_mask"] > 0.5

        road_values = F.cross_entropy(
            outputs["road_logits"],
            batch["road_target"],
            weight=self.road_class_weights,
            reduction="none",
        )
        road_loss = _masked_mean(road_values, valid)
        road_loss = road_loss + self.config.dice * _multiclass_dice_loss(
            outputs["road_logits"], batch["road_target"], valid
        )

        landuse_values = F.cross_entropy(
            outputs["landuse_logits"],
            batch["landuse_target"],
            weight=self.landuse_class_weights,
            reduction="none",
        )
        landuse_mask = valid & (batch["landuse_known_mask"] > 0.5)
        landuse_loss = _masked_mean(landuse_values, landuse_mask)
        landuse_loss = landuse_loss + self.config.dice * _multiclass_dice_loss(
            outputs["landuse_logits"], batch["landuse_target"], landuse_mask
        )

        binary_bce = F.binary_cross_entropy_with_logits(
            outputs["binary_logits"],
            batch["binary_target"],
            pos_weight=self.binary_positive_weights,
            reduction="none",
        )
        binary_loss = _masked_mean(binary_bce, valid)
        binary_loss = binary_loss + self.config.dice * _dice_loss(
            outputs["binary_logits"], batch["binary_target"], valid
        )

        height_prediction = torch.sigmoid(outputs["height_logits"])
        height_values = F.smooth_l1_loss(
            height_prediction,
            batch["height_target"],
            reduction="none",
            beta=0.02,
        )
        confidence = batch["height_confidence"].clamp(0, 3).long()
        confidence_weight = self.height_confidence_weights[confidence]
        building = batch["binary_target"][:, 1] > 0.5
        height_mask = valid & building
        height_loss = _masked_mean(height_values * confidence_weight.unsqueeze(1), height_mask)

        centerline_bce = F.binary_cross_entropy_with_logits(
            outputs["centerline_logits"],
            batch["centerline_target"],
            pos_weight=self.centerline_positive_weight,
            reduction="none",
        )
        centerline_loss = _masked_mean(centerline_bce, valid)
        centerline_loss = centerline_loss + self.config.dice * _dice_loss(
            outputs["centerline_logits"], batch["centerline_target"], valid
        )
        if self.config.centerline_tversky > 0:
            centerline_loss = centerline_loss + self.config.centerline_tversky * _tversky_loss(
                outputs["centerline_logits"],
                batch["centerline_target"],
                valid,
                alpha=self.config.tversky_alpha,
                beta=self.config.tversky_beta,
            )

        boundary_loss = outputs["centerline_logits"].sum() * 0.0
        if self.config.boundary_centerline > 0 and "boundary_guide" in batch:
            guide = batch["boundary_guide"] > 0.5
            guide_mask = guide.any(dim=1)
            if guide_mask.any():
                guide_values = F.binary_cross_entropy_with_logits(
                    outputs["centerline_logits"],
                    guide.to(dtype=outputs["centerline_logits"].dtype),
                    reduction="none",
                )
                boundary_loss = _masked_mean(guide_values, guide_mask)

        total = (
            self.config.road * road_loss
            + self.config.landuse * landuse_loss
            + self.config.binary * binary_loss
            + self.config.height * height_loss
            + self.config.centerline * centerline_loss
            + self.config.boundary_centerline * boundary_loss
        )
        return {
            "total": total,
            "road": road_loss,
            "landuse": landuse_loss,
            "binary": binary_loss,
            "height": height_loss,
            "centerline": centerline_loss,
            "boundary_centerline": boundary_loss,
        }
