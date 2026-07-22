from __future__ import annotations

from dataclasses import replace

import geopandas as gpd
import numpy as np

from .classify import clean_tag, estimate_road_width_metres
from .config import BuildConfig
from .extract import CityLayers
from .heights import HeightEstimate, estimate_building_height


def enrich_roads(roads: gpd.GeoDataFrame, config: BuildConfig) -> gpd.GeoDataFrame:
    if roads.empty:
        return roads.copy()
    enriched = roads.copy()
    enriched["estimated_width_m"] = [
        estimate_road_width_metres(
            row,
            class_fallbacks=config.roads.widths_m,
            lane_width_m=config.roads.lane_width_m,
            edge_margin_m=config.roads.edge_margin_m,
            minimum_width_m=config.roads.minimum_width_m,
            maximum_width_m=config.roads.maximum_width_m,
        )
        for _, row in enriched.iterrows()
    ]
    return enriched


def enrich_buildings(buildings: gpd.GeoDataFrame, config: BuildConfig) -> gpd.GeoDataFrame:
    if buildings.empty:
        return buildings.copy()
    enriched = buildings.copy()
    estimates: list[HeightEstimate] = []
    types: list[str] = []
    for _, row in enriched.iterrows():
        estimates.append(
            estimate_building_height(
                row,
                floor_height_m=config.heights.floor_height_m,
                default_height_m=config.heights.default_height_m,
                default_by_building=config.heights.default_by_building,
            )
        )
        types.append(clean_tag(row.get("building")))

    observed_by_type: dict[str, list[float]] = {}
    for estimate, building_type in zip(estimates, types, strict=True):
        if building_type and estimate.confidence >= 2:
            observed_by_type.setdefault(building_type, []).append(estimate.metres)
    type_medians = {
        building_type: float(np.median(values))
        for building_type, values in observed_by_type.items()
        if len(values) >= 3
    }

    final: list[HeightEstimate] = []
    for estimate, building_type in zip(estimates, types, strict=True):
        if estimate.confidence == 0 and building_type in type_medians:
            final.append(HeightEstimate(type_medians[building_type], 1, "city_type_median"))
        else:
            final.append(estimate)

    enriched["estimated_height_m"] = [estimate.metres for estimate in final]
    enriched["height_confidence"] = [estimate.confidence for estimate in final]
    enriched["height_source"] = [estimate.source for estimate in final]
    return enriched


def enrich_layers(layers: CityLayers, config: BuildConfig) -> CityLayers:
    return replace(
        layers,
        roads=enrich_roads(layers.roads, config),
        buildings=enrich_buildings(layers.buildings, config),
    )
