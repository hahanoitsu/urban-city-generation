from __future__ import annotations

from dataclasses import replace

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import box

from .extract import CityLayers


def choose_metric_crs(
    bbox_wgs84: tuple[float, float, float, float], requested: str | int
) -> CRS:
    if str(requested).lower() != "auto":
        return CRS.from_user_input(requested)
    boundary = gpd.GeoDataFrame(
        {"geometry": [box(*bbox_wgs84)]}, geometry="geometry", crs="EPSG:4326"
    )
    estimated = boundary.estimate_utm_crs()
    if estimated is None:
        raise ValueError("Could not estimate a suitable UTM CRS; set input.metric_crs explicitly")
    return CRS.from_user_input(estimated)


def project_and_clip_layers(
    layers: CityLayers,
    bbox_wgs84: tuple[float, float, float, float],
    metric_crs: CRS,
) -> tuple[CityLayers, gpd.GeoDataFrame]:
    boundary_wgs84 = gpd.GeoDataFrame(
        {"geometry": [box(*bbox_wgs84)]}, geometry="geometry", crs="EPSG:4326"
    )
    boundary = boundary_wgs84.to_crs(metric_crs)
    roi = boundary.geometry.iloc[0]

    projected: dict[str, gpd.GeoDataFrame] = {}
    for name, frame in layers.items():
        if frame.empty:
            projected[name] = gpd.GeoDataFrame({"geometry": []}, crs=metric_crs)
            continue
        candidate = frame.to_crs(metric_crs)
        candidate = candidate[candidate.geometry.intersects(roi)].copy()
        candidate["geometry"] = candidate.geometry.intersection(roi)
        candidate = candidate[candidate.geometry.notna() & ~candidate.geometry.is_empty].copy()
        projected[name] = candidate

    return replace(layers, **projected), boundary
