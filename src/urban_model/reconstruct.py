from __future__ import annotations

import torch

ROAD_CHANNELS = (1, 2, 3)
LANDUSE_CHANNELS = (4, 5, 6, 7, 10)


def reconstruct_layers(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    road_class = outputs["road_logits"].argmax(dim=1)
    landuse_class = outputs["landuse_logits"].argmax(dim=1)
    binary = (torch.sigmoid(outputs["binary_logits"]) >= 0.5).to(
        outputs["binary_logits"].dtype
    )
    height = torch.sigmoid(outputs["height_logits"])

    batch, _, height_pixels, width_pixels = outputs["road_logits"].shape
    layers = torch.zeros(
        (batch, 12, height_pixels, width_pixels),
        device=outputs["road_logits"].device,
        dtype=outputs["road_logits"].dtype,
    )
    layers[:, 0] = binary[:, 0]
    layers[:, 8] = binary[:, 1]
    layers[:, 9] = height[:, 0] * binary[:, 1]
    layers[:, 11] = binary[:, 2]

    for class_index, channel_index in enumerate(ROAD_CHANNELS, start=1):
        layers[:, channel_index] = (road_class == class_index).to(layers.dtype)
    for class_index, channel_index in enumerate(LANDUSE_CHANNELS, start=1):
        layers[:, channel_index] = (landuse_class == class_index).to(layers.dtype)
    return layers
