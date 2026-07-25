from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_bounds
from shapely.geometry import LineString

from urban_dataset.config import (
    BuildConfig,
    HeightConfig,
    InputConfig,
    OutputConfig,
    ProjectConfig,
    QualityConfig,
    RasterConfig,
    RoadConfig,
    VerticalProfileConfig,
)
from urban_dataset.tile import TileSpec
from urban_dataset.vertical_profile import build_vertical_profiles


def _config() -> BuildConfig:
    return BuildConfig(
        project=ProjectConfig("test"),
        input=InputConfig(Path("dummy.pbf"), (0.0, 0.0, 1.0, 1.0)),
        output=OutputConfig(Path("out")),
        raster=RasterConfig(tile_size_m=100, pixels=100, stride_m=100),
        roads=RoadConfig(),
        heights=HeightConfig(),
        vertical_profiles=VerticalProfileConfig(
            road_default_elevated_m=8.0,
            road_default_tunnel_depth_m=10.0,
            sample_step_m=1.0,
        ),
        quality=QualityConfig(),
    )


def test_bridge_profile_ramps_from_surface_to_deck() -> None:
    frame = gpd.GeoDataFrame(
        {
            "bridge": ["yes"],
            "tunnel": [None],
            "layer": ["1"],
            "road_class": ["secondary"],
            "estimated_width_m": [6.0],
            "geometry": [LineString([(10.0, 50.0), (90.0, 50.0)])],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    tile = TileSpec("test", 0, 0, 0.0, 0.0, 100.0, 100.0)
    result = build_vertical_profiles(
        frame,
        transport_mode="road",
        tile=tile,
        config=_config(),
        pixels=100,
        transform=from_bounds(0.0, 0.0, 100.0, 100.0, 100, 100),
    )

    elevated = result.offsets_m[2]
    assert elevated.max() == pytest.approx(8.0, abs=0.2)
    assert elevated[50, 10] < elevated[50, 50]
    assert elevated[50, 90] < elevated[50, 50]
    assert result.confidence[2].max() == 1


def test_explicit_min_height_and_incline_are_used() -> None:
    frame = gpd.GeoDataFrame(
        {
            "bridge": ["yes"],
            "min_height": ["11.5"],
            "incline": ["2%"],
            "road_class": ["major"],
            "estimated_width_m": [8.0],
            "geometry": [LineString([(0.0, 50.0), (100.0, 50.0)])],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    tile = TileSpec("test", 0, 0, 0.0, 0.0, 100.0, 100.0)
    result = build_vertical_profiles(
        frame,
        transport_mode="road",
        tile=tile,
        config=_config(),
        pixels=100,
        transform=from_bounds(0.0, 0.0, 100.0, 100.0, 100, 100),
    )

    values = result.offsets_m[2]
    assert np.max(values) > 12.0
    assert result.confidence[2].max() == 2
    assert result.evidence_counts["min_height"] == 1
