from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

ROAD_NAMES = ("major", "secondary", "local")
LANDUSE_NAMES = ("residential", "commercial_mixed", "industrial", "green", "civic")
BINARY_NAMES = ("water", "building", "rail")


@dataclass
class MetricAccumulator:
    height_scale_m: float = 180.0
    loss_sums: dict[str, float] = field(default_factory=dict)
    batches: int = 0
    road_intersection: torch.Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )
    road_union: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))
    landuse_intersection: torch.Tensor = field(
        default_factory=lambda: torch.zeros(5, dtype=torch.float64)
    )
    landuse_union: torch.Tensor = field(
        default_factory=lambda: torch.zeros(5, dtype=torch.float64)
    )
    binary_intersection: torch.Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )
    binary_union: torch.Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=torch.float64)
    )
    height_absolute_error: float = 0.0
    height_pixels: int = 0
    boundary_required: int = 0
    boundary_matched: int = 0

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        losses: dict[str, torch.Tensor],
    ) -> None:
        for name, value in losses.items():
            self.loss_sums[name] = self.loss_sums.get(name, 0.0) + float(value.detach().cpu())
        self.batches += 1

        valid = batch["valid_mask"] > 0.5
        road_prediction = outputs["road_logits"].argmax(dim=1)
        for class_index in range(1, 4):
            predicted = (road_prediction == class_index) & valid
            target = (batch["road_target"] == class_index) & valid
            self.road_intersection[class_index - 1] += (predicted & target).sum().cpu()
            self.road_union[class_index - 1] += (predicted | target).sum().cpu()

        landuse_prediction = outputs["landuse_logits"].argmax(dim=1)
        landuse_mask = valid & (batch["landuse_known_mask"] > 0.5)
        for class_index in range(1, 6):
            predicted = (landuse_prediction == class_index) & landuse_mask
            target = (batch["landuse_target"] == class_index) & landuse_mask
            self.landuse_intersection[class_index - 1] += (predicted & target).sum().cpu()
            self.landuse_union[class_index - 1] += (predicted | target).sum().cpu()

        binary_prediction = torch.sigmoid(outputs["binary_logits"]) > 0.5
        binary_target = batch["binary_target"] > 0.5
        for channel in range(3):
            predicted = binary_prediction[:, channel] & valid
            target = binary_target[:, channel] & valid
            self.binary_intersection[channel] += (predicted & target).sum().cpu()
            self.binary_union[channel] += (predicted | target).sum().cpu()

        height_prediction = torch.sigmoid(outputs["height_logits"])[:, 0]
        height_target = batch["height_target"][:, 0]
        height_mask = valid & (batch["height_confidence"] >= 2) & binary_target[:, 1]
        self.height_absolute_error += float(
            torch.abs(height_prediction - height_target)[height_mask].sum().detach().cpu()
        )
        self.height_pixels += int(height_mask.sum().detach().cpu())

        if "boundary_guide" in batch:
            guide = batch["boundary_guide"] > 0.5
            midpoint = guide.shape[-1] // 2
            required = guide[:, :, :, midpoint]
            prediction = torch.sigmoid(outputs["centerline_logits"]) > 0.5
            guide_length = batch.get("guide_length", 12)
            if torch.is_tensor(guide_length):
                guide_length = int(guide_length.max().item())
            elif isinstance(guide_length, (list, tuple)):
                guide_length = max(int(value) for value in guide_length)
            guide_length = max(1, int(guide_length))
            unknown_band = prediction[:, :, :, midpoint : midpoint + guide_length]
            predicted_rows = unknown_band.any(dim=-1).to(torch.float32)
            batch_size, classes, rows = predicted_rows.shape
            dilated = F.max_pool1d(
                predicted_rows.reshape(batch_size * classes, 1, rows),
                kernel_size=3,
                stride=1,
                padding=1,
            ).reshape(batch_size, classes, rows) > 0.5
            self.boundary_required += int(required.sum().detach().cpu())
            self.boundary_matched += int((required & dilated).sum().detach().cpu())

    @staticmethod
    def _iou(
        names: tuple[str, ...], intersection: torch.Tensor, union: torch.Tensor
    ) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for name, numerator, denominator in zip(names, intersection, union, strict=True):
            result[name] = float(numerator / denominator) if denominator > 0 else None
        return result

    def compute(self) -> dict[str, object]:
        divisor = max(self.batches, 1)
        height_mae = self.height_absolute_error / self.height_pixels if self.height_pixels else None
        return {
            "loss": {name: value / divisor for name, value in self.loss_sums.items()},
            "road_iou": self._iou(ROAD_NAMES, self.road_intersection, self.road_union),
            "landuse_iou": self._iou(
                LANDUSE_NAMES, self.landuse_intersection, self.landuse_union
            ),
            "binary_iou": self._iou(BINARY_NAMES, self.binary_intersection, self.binary_union),
            "height_mae_normalized_observed": height_mae,
            "height_mae_metres_observed": (
                height_mae * self.height_scale_m if height_mae is not None else None
            ),
            "boundary_road_recall": (
                self.boundary_matched / self.boundary_required
                if self.boundary_required
                else None
            ),
            "boundary_road_crossings": self.boundary_required,
        }
