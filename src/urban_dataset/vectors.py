from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping
from shapely.ops import transform

from .city_state import (
    build_transport_graph,
    building_solids,
    city_state_header,
)
from .extract import CityLayers
from .tile import TileSpec


def _local_geometry(geometry, tile: TileSpec):
    return transform(lambda x, y, z=None: (x - tile.minx, y - tile.miny), geometry)


def _clean_properties(row: Any, geometry_column: str = "geometry") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key == geometry_column or value is None:
            continue
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            continue
        if isinstance(value, (str, int, float, bool)):
            result[str(key)] = value
    return result


def frame_features(frame: gpd.GeoDataFrame, tile: TileSpec) -> list[dict[str, Any]]:
    features = []
    for _, row in frame.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        features.append(
            {
                "geometry": mapping(_local_geometry(geometry, tile)),
                "properties": _clean_properties(row),
            }
        )
    return features


def tile_vector_payload(
    layers: CityLayers,
    tile: TileSpec,
    crs: str,
    *,
    area_id: str | None = None,
) -> dict[str, Any]:
    payload = city_state_header(tile, crs, area_id=area_id)
    payload.update(
        {
            # The original vector collections remain available for data inspection and
            # backward-compatible consumers. New 3D/PCG code should use the attributed
            # graph and building solids below.
            "roads": frame_features(layers.roads, tile),
            "buildings": frame_features(layers.buildings, tile),
            "landuse": frame_features(layers.landuse, tile),
            "landuse_known": frame_features(layers.landuse_known, tile),
            "water": frame_features(layers.water, tile),
            "green": frame_features(layers.green, tile),
            "rail": frame_features(layers.rail, tile),
            "transport_graph": build_transport_graph(layers.roads, layers.rail, tile),
            "building_solids": building_solids(layers.buildings, tile),
        }
    )
    return payload
