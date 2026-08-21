from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .config import LayeredDiffusionConfig
from .data import LayeredBlockDataset, check_layered_data

VERTICAL_BACKGROUND_MIN = 0.01
VERTICAL_BACKGROUND_MAX = 0.5


def balance_vertical_supervision(
    values: torch.Tensor,
    supervision: torch.Tensor,
) -> torch.Tensor:
    result = supervision.clone()
    valid = result[:1].clamp(0.0, 1.0)
    active = values[8:12] > 0.0
    valid_pixels = valid > 0.0

    positive = (active & valid_pixels).sum(dim=(-2, -1)).float()
    negative = ((~active) & valid_pixels).sum(dim=(-2, -1)).float()
    background_weights = (positive / negative.clamp_min(1.0)).clamp(
        VERTICAL_BACKGROUND_MIN,
        VERTICAL_BACKGROUND_MAX,
    )
    background_weights = torch.where(
        positive > 0,
        background_weights,
        torch.ones_like(background_weights),
    )
    background = valid * background_weights[:, None, None]
    result[8:12] = torch.where(active, valid, background)
    return result


class CityConditionedDataset(LayeredBlockDataset):
    def __init__(
        self,
        config: LayeredDiffusionConfig,
        manifest: str | Path,
        *,
        augment: bool,
    ) -> None:
        super().__init__(config, manifest, augment=augment)
        lookup = {name: index for index, name in enumerate(config.city_names)}
        self.city_indices: list[int] = []
        for tile_index, _, _ in self.crops:
            city_id = str(self.tiles.rows[tile_index].get("city_id", "")).strip()
            if not city_id and len(config.city_names) == 1:
                city_id = config.city_names[0]
            if city_id not in lookup:
                raise ValueError(
                    f"Manifest city {city_id!r} is not listed in data.cities"
                )
            self.city_indices.append(lookup[city_id])
        counts = Counter(self.city_indices)
        self.city_counts = {
            name: int(counts.get(index, 0))
            for index, name in enumerate(config.city_names)
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample["valid_mask"] = balance_vertical_supervision(
            sample["x0"], sample["valid_mask"]
        )
        city = torch.zeros(len(self.config.city_names), dtype=torch.float32)
        city[self.city_indices[index]] = 1.0
        sample["city"] = city
        return sample


def parse_city_mix(
    city_names: tuple[str, ...],
    value: str | None,
) -> dict[str, float]:
    if value is None or not value.strip():
        if len(city_names) == 1:
            return {city_names[0]: 1.0}
        weight = 1.0 / len(city_names)
        return {name: weight for name in city_names}

    raw: dict[str, float] = {}
    for part in value.split(","):
        name, separator, number = part.strip().partition("=")
        if not separator or name not in city_names:
            raise ValueError(f"Invalid city mixture item: {part!r}")
        weight = float(number)
        if weight < 0:
            raise ValueError("City mixture weights cannot be negative")
        raw[name] = raw.get(name, 0.0) + weight

    total = sum(raw.values())
    if total <= 0:
        raise ValueError("City mixture needs at least one positive weight")
    return {name: raw.get(name, 0.0) / total for name in city_names}


def city_mix_tensor(
    city_names: tuple[str, ...],
    mixture: dict[str, float],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = torch.tensor(
        [mixture.get(name, 0.0) for name in city_names],
        device=device,
        dtype=dtype,
    )
    return values.unsqueeze(0).expand(batch_size, -1)


def preview_city_mix(
    config: LayeredDiffusionConfig,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.zeros(
        (count, len(config.city_names)),
        device=device,
        dtype=torch.float32,
    )
    for index in range(count):
        rows[index, index % len(config.city_names)] = 1.0
    return rows


def build_model_input(
    noisy: torch.Tensor,
    city: torch.Tensor,
    config: LayeredDiffusionConfig,
) -> torch.Tensor:
    if city.ndim == 1:
        city = city.unsqueeze(0)
    if city.shape != (noisy.shape[0], len(config.city_names)):
        raise ValueError(
            f"Expected city condition {(noisy.shape[0], len(config.city_names))}, "
            f"found {tuple(city.shape)}"
        )
    height, width = noisy.shape[-2:]
    city_maps = city.to(dtype=noisy.dtype).unsqueeze(-1).unsqueeze(-1)
    city_maps = city_maps.expand(-1, -1, height, width)
    parts = [noisy, city_maps]
    if config.coordinate_channels:
        y = torch.linspace(-1.0, 1.0, height, device=noisy.device, dtype=noisy.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=noisy.device, dtype=noisy.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((grid_x, grid_y), dim=0)
        parts.append(coordinates.unsqueeze(0).expand(noisy.shape[0], -1, -1, -1))
    return torch.cat(parts, dim=1)


def diffusion_target(
    scheduler,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "sample":
        return clean
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(clean, noise, timesteps)
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def check_conditioned_data(
    config: LayeredDiffusionConfig,
    *,
    samples_per_split: int = 4,
) -> dict[str, Any]:
    result = check_layered_data(config, samples_per_split=samples_per_split)
    result["cities"] = list(config.city_names)
    for split, manifest in (
        ("train", config.train_manifest),
        ("validation", config.validation_manifest),
    ):
        dataset = CityConditionedDataset(config, manifest, augment=False)
        result[split]["city_samples"] = dataset.city_counts
    return result
