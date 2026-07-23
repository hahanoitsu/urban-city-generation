from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, box

from urban_dataset.config import (
    BuildConfig,
    HeightConfig,
    InputConfig,
    OutputConfig,
    ProjectConfig,
    QualityConfig,
    RasterConfig,
    RoadConfig,
)
from urban_dataset.extract import CityLayers
from urban_dataset.raster import rasterize_tile
from urban_dataset.tile import TileSpec
from urban_dataset.vertical import VerticalMode, classify_vertical_mode


def _empty(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _config(tmp_path: Path) -> BuildConfig:
    return BuildConfig(
        project=ProjectConfig(city_id="vertical-test"),
        input=InputConfig(
            pbf_path=Path("unused.osm.pbf"),
            bbox_wgs84=(0.0, 0.0, 1.0, 1.0),
            metric_crs="EPSG:3857",
        ),
        output=OutputConfig(root=tmp_path),
        raster=RasterConfig(tile_size_m=64, pixels=64, stride_m=64),
        roads=RoadConfig(),
        heights=HeightConfig(),
        quality=QualityConfig(),
    )


def _city(
    *,
    roads: gpd.GeoDataFrame,
    rail: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> CityLayers:
    crs = str(roads.crs or rail.crs or buildings.crs)
    empty = _empty(crs)
    return CityLayers(
        roads=roads,
        buildings=buildings,
        landuse=empty.copy(),
        landuse_known=empty.copy(),
        water=empty.copy(),
        green=empty.copy(),
        rail=rail,
    )


def _building(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "building": ["commercial"],
            "estimated_height_m": [24.0],
            "height_confidence": [3],
            "geometry": [box(20, 20, 44, 44)],
        },
        crs=crs,
    )


def test_vertical_mode_requires_explicit_structure_tags():
    assert classify_vertical_mode({}) == VerticalMode.SURFACE
    assert classify_vertical_mode({"railway": "subway"}) == VerticalMode.SURFACE
    assert classify_vertical_mode({"tunnel": "yes", "layer": "-1"}) == VerticalMode.UNDERGROUND
    assert classify_vertical_mode({"bridge": "viaduct", "layer": "1"}) == VerticalMode.ELEVATED
    assert classify_vertical_mode({"layer": "-1"}) == VerticalMode.UNKNOWN
    assert classify_vertical_mode({"bridge": "yes", "tunnel": "yes"}) == VerticalMode.UNKNOWN


def test_underground_rail_does_not_erase_surface_building(tmp_path):
    crs = "EPSG:3857"
    roads = _empty(crs)
    rail = gpd.GeoDataFrame(
        {
            "railway": ["subway"],
            "tunnel": ["yes"],
            "layer": ["-1"],
            "geometry": [LineString([(0, 32), (64, 32)])],
        },
        crs=crs,
    )
    result = rasterize_tile(
        TileSpec("vertical-test", 0, 0, 0, 0, 64, 64),
        _city(roads=roads, rail=rail, buildings=_building(crs)),
        _config(tmp_path),
    )

    assert result.layers[8, 32, 32] == 1
    assert result.layers[11, 32, 32] == 0
    assert result.rail_vertical_masks[int(VerticalMode.UNDERGROUND), 32, 32] == 1
    assert result.surface_transport_reservation[32, 32] == 0
    assert result.buildability_known_mask[32, 32] == 1


def test_elevated_rail_can_overlap_surface_road(tmp_path):
    crs = "EPSG:3857"
    roads = gpd.GeoDataFrame(
        {
            "highway": ["residential"],
            "road_class": ["local"],
            "estimated_width_m": [6.0],
            "geometry": [LineString([(0, 32), (64, 32)])],
        },
        crs=crs,
    )
    rail = gpd.GeoDataFrame(
        {
            "railway": ["light_rail"],
            "bridge": ["viaduct"],
            "layer": ["1"],
            "geometry": [LineString([(32, 0), (32, 64)])],
        },
        crs=crs,
    )
    result = rasterize_tile(
        TileSpec("vertical-test", 0, 0, 0, 0, 64, 64),
        _city(roads=roads, rail=rail, buildings=_building(crs)),
        _config(tmp_path),
    )

    assert result.layers[3, 32, 32] == 1
    assert result.layers[11, 32, 32] == 0
    assert result.rail_vertical_masks[int(VerticalMode.ELEVATED), 32, 32] == 1
    assert result.surface_transport_reservation[32, 32] == 1
    assert result.buildable_surface_mask[32, 32] == 0


def test_ambiguous_layer_masks_buildability_supervision(tmp_path):
    crs = "EPSG:3857"
    roads = _empty(crs)
    rail = gpd.GeoDataFrame(
        {
            "railway": ["rail"],
            "layer": ["-1"],
            "geometry": [LineString([(0, 32), (64, 32)])],
        },
        crs=crs,
    )
    result = rasterize_tile(
        TileSpec("vertical-test", 0, 0, 0, 0, 64, 64),
        _city(roads=roads, rail=rail, buildings=_building(crs)),
        _config(tmp_path),
    )

    assert result.rail_vertical_masks[int(VerticalMode.UNKNOWN), 32, 32] == 1
    assert result.buildability_known_mask[32, 32] == 0
    assert result.buildable_surface_mask[32, 32] == 0
