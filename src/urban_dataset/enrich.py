from __future__ import annotations

from dataclasses import replace

import geopandas as gpd
import pandas as pd

from .classify import DEFAULT_WIDTH_BY_HIGHWAY_M
from .config import BuildConfig
from .extract import CityLayers

_NUMBER_PATTERN = r"([-+]?\d+(?:[.,]\d+)?)"


def _clean_tags(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.split(";", n=1)
        .str[0]
    )


def _numbers(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(_NUMBER_PATTERN, expand=False)
    return pd.to_numeric(extracted.str.replace(",", ".", regex=False), errors="coerce")


def enrich_roads(roads: gpd.GeoDataFrame, config: BuildConfig) -> gpd.GeoDataFrame:
    if roads.empty:
        return roads.copy()

    result = roads.copy()
    highway = _clean_tags(result.get("highway", pd.Series(index=result.index)))
    road_class = _clean_tags(result.get("road_class", pd.Series(index=result.index)))

    fallback = highway.map(DEFAULT_WIDTH_BY_HIGHWAY_M)
    fallback = fallback.fillna(road_class.map(config.roads.widths_m)).fillna(5.0)

    lanes = _numbers(result.get("lanes", pd.Series(index=result.index)))
    lane_width = lanes * config.roads.lane_width_m + config.roads.edge_margin_m
    lane_width = lane_width.where(lanes.between(0, 20, inclusive="neither"))

    width_text = result.get("width", pd.Series(index=result.index)).astype("string")
    explicit = _numbers(width_text)
    feet = width_text.str.lower().str.contains(r"\b(?:ft|feet|foot)\b", regex=True, na=False)
    explicit = explicit.where(~feet, explicit * 0.3048)
    explicit = explicit.where(explicit.between(1.5, 60.0))

    estimated = fallback.where(lane_width.isna(), lane_width)
    estimated = estimated.where(explicit.isna(), explicit)
    result["estimated_width_m"] = estimated.clip(
        lower=config.roads.minimum_width_m,
        upper=config.roads.maximum_width_m,
    ).astype(float)
    return result


def enrich_buildings(buildings: gpd.GeoDataFrame, config: BuildConfig) -> gpd.GeoDataFrame:
    if buildings.empty:
        return buildings.copy()

    result = buildings.copy()
    building_type = _clean_tags(result.get("building", pd.Series(index=result.index)))

    fallback = building_type.map(config.heights.default_by_building)
    estimated = fallback.fillna(config.heights.default_height_m).astype(float)
    confidence = pd.Series(0, index=result.index, dtype="int8")
    source = pd.Series("default", index=result.index, dtype="string")

    levels = _numbers(result.get("building:levels", pd.Series(index=result.index)))
    valid_levels = levels.between(0, 300, inclusive="neither")
    estimated.loc[valid_levels] = levels.loc[valid_levels] * config.heights.floor_height_m
    confidence.loc[valid_levels] = 2
    source.loc[valid_levels] = "building:levels"

    height_text = result.get("height", pd.Series(index=result.index)).astype("string")
    explicit = _numbers(height_text)
    feet = height_text.str.lower().str.contains(r"\b(?:ft|feet|foot)\b", regex=True, na=False)
    explicit = explicit.where(~feet, explicit * 0.3048)
    valid_explicit = explicit.between(1.5, 1000.0)
    estimated.loc[valid_explicit] = explicit.loc[valid_explicit]
    confidence.loc[valid_explicit] = 3
    source.loc[valid_explicit] = "height"

    observed = pd.DataFrame(
        {"building_type": building_type, "height": estimated, "confidence": confidence},
        index=result.index,
    )
    observed = observed[(observed["building_type"] != "") & (observed["confidence"] >= 2)]
    grouped = observed.groupby("building_type")["height"].agg(["median", "count"])
    medians = grouped.loc[grouped["count"] >= 3, "median"]

    contextual = (confidence == 0) & building_type.isin(medians.index)
    estimated.loc[contextual] = building_type.loc[contextual].map(medians)
    confidence.loc[contextual] = 1
    source.loc[contextual] = "city_type_median"

    result["estimated_height_m"] = estimated.astype(float)
    result["height_confidence"] = confidence.astype("int8")
    result["height_source"] = source.astype(object)
    return result


def enrich_layers(layers: CityLayers, config: BuildConfig) -> CityLayers:
    return replace(
        layers,
        roads=enrich_roads(layers.roads, config),
        buildings=enrich_buildings(layers.buildings, config),
    )
