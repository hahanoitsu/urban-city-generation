from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import QualityConfig
from .extract import CityLayers
from .raster import RasterResult


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float | int]


def validate_tile(
    city: CityLayers,
    raster: RasterResult,
    quality: QualityConfig,
) -> ValidationResult:
    total_pixels = raster.layers.shape[1] * raster.layers.shape[2]
    valid = raster.valid_data_mask > 0
    valid_pixels = int(valid.sum())
    valid_fraction = valid_pixels / total_pixels if total_pixels else 0.0
    denominator = max(valid_pixels, 1)

    building_count = int(len(city.buildings))
    road_length_m = float(city.roads.geometry.length.sum()) if not city.roads.empty else 0.0
    nonempty = np.maximum.reduce(
        [raster.layers[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]]
    )
    nonempty_fraction = float(((nonempty > 0) & valid).sum() / denominator)
    water_fraction = float(((raster.layers[0] > 0) & valid).sum() / denominator)
    building = (raster.layers[8] > 0) & valid
    building_pixels = int(building.sum())
    building_fraction = float(building_pixels / denominator)
    roads = np.maximum.reduce(raster.layers[1:4])
    road = (roads > 0) & valid
    road_fraction = float(road.sum() / denominator)
    green_fraction = float(((raster.layers[7] > 0) & valid).sum() / denominator)
    rail_fraction = float(((raster.layers[11] > 0) & valid).sum() / denominator)
    landuse_known_fraction = float(((raster.landuse_known_mask > 0) & valid).sum() / denominator)
    building_road_overlap = int((building & road).sum())
    building_road_overlap_fraction = (
        float(building_road_overlap / building_pixels) if building_pixels else 0.0
    )
    classified_landuse = np.maximum.reduce(raster.layers[[4, 5, 6, 7, 10]]) > 0
    building_without_classified_landuse_fraction = (
        float((building & ~classified_landuse).sum() / building_pixels)
        if building_pixels
        else 0.0
    )
    building_without_osm_landuse_coverage_fraction = (
        float((building & (raster.landuse_known_mask == 0)).sum() / building_pixels)
        if building_pixels
        else 0.0
    )
    observed = raster.observed_building_heights
    observed_fraction = float(observed / building_count) if building_count else 0.0
    contextual = raster.height_confidence_counts.get(1, 0)
    explicit = raster.height_confidence_counts.get(3, 0)
    levels = raster.height_confidence_counts.get(2, 0)
    defaults = raster.height_confidence_counts.get(0, 0)

    reasons: list[str] = []
    if valid_fraction < quality.minimum_valid_fraction:
        reasons.append(f"valid_fraction<{quality.minimum_valid_fraction}")
    if building_count < quality.minimum_buildings:
        reasons.append(f"building_count<{quality.minimum_buildings}")
    if road_length_m < quality.minimum_road_length_m:
        reasons.append(f"road_length_m<{quality.minimum_road_length_m}")
    if nonempty_fraction < quality.minimum_nonempty_fraction:
        reasons.append(f"nonempty_fraction<{quality.minimum_nonempty_fraction}")
    if water_fraction > quality.reject_water_fraction_above:
        reasons.append(f"water_fraction>{quality.reject_water_fraction_above}")
    if not np.isfinite(raster.layers).all():
        reasons.append("non_finite_raster_value")

    return ValidationResult(
        accepted=not reasons,
        reasons=tuple(reasons),
        metrics={
            "building_count": building_count,
            "road_length_m": round(road_length_m, 3),
            "valid_fraction": round(valid_fraction, 6),
            "nonempty_fraction": round(nonempty_fraction, 6),
            "water_fraction": round(water_fraction, 6),
            "building_coverage": round(building_fraction, 6),
            "road_coverage": round(road_fraction, 6),
            "green_fraction": round(green_fraction, 6),
            "rail_fraction": round(rail_fraction, 6),
            "landuse_known_fraction": round(landuse_known_fraction, 6),
            "building_road_overlap_fraction": round(building_road_overlap_fraction, 6),
            "building_without_classified_landuse_fraction": round(
                building_without_classified_landuse_fraction, 6
            ),
            "building_without_osm_landuse_coverage_fraction": round(
                building_without_osm_landuse_coverage_fraction, 6
            ),
            "height_explicit_buildings": explicit,
            "height_levels_buildings": levels,
            "height_contextual_buildings": contextual,
            "height_default_buildings": defaults,
            "height_observed_buildings": observed,
            "height_observed_fraction": round(observed_fraction, 6),
            "mean_building_height_m": round(
                float(np.mean(raster.building_heights_m)) if raster.building_heights_m else 0.0,
                3,
            ),
        },
    )
