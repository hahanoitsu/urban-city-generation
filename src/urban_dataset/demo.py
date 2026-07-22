from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, box

from .config import (
    BuildConfig,
    HeightConfig,
    InputConfig,
    OutputConfig,
    ProjectConfig,
    QualityConfig,
    RasterConfig,
    RoadConfig,
)
from .extract import CityLayers
from .pipeline import run_build


def create_demo_layers() -> CityLayers:
    crs = "EPSG:4326"
    # A tiny synthetic neighbourhood around Singapore coordinates. It tests the
    # complete projection/raster/output path without downloading OSM data.
    roads = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "highway": ["primary", "secondary", "residential", "residential", "service"],
            "road_class": ["major", "secondary", "local", "local", "local"],
            "geometry": [
                LineString([(103.8500, 1.2950), (103.8600, 1.2950)]),
                LineString([(103.8550, 1.2900), (103.8550, 1.3000)]),
                LineString([(103.8505, 1.2920), (103.8595, 1.2920)]),
                LineString([(103.8505, 1.2980), (103.8595, 1.2980)]),
                LineString([(103.8520, 1.2905), (103.8520, 1.2995)]),
            ],
        },
        crs=crs,
    )
    building_geometries = []
    types = []
    heights = []
    levels = []
    for x in [103.8510, 103.8530, 103.8560, 103.8580]:
        for y in [1.2910, 1.2932, 1.2962, 1.2982]:
            building_geometries.append(box(x, y, x + 0.00075, y + 0.00055))
            types.append("apartments" if x > 103.855 else "residential")
            heights.append("24 m" if x > 103.857 else None)
            levels.append("6" if x > 103.855 and x <= 103.857 else None)
    buildings = gpd.GeoDataFrame(
        {
            "id": range(len(building_geometries)),
            "building": types,
            "height": heights,
            "building:levels": levels,
            "geometry": building_geometries,
        },
        crs=crs,
    )
    landuse = gpd.GeoDataFrame(
        {
            "landuse": ["residential", "commercial", "industrial", None],
            "leisure": [None, None, None, "park"],
            "landuse_class": ["residential", "commercial_mixed", "industrial", "green"],
            "geometry": [
                box(103.8502, 1.2902, 103.8550, 1.2998),
                box(103.8550, 1.2948, 103.8600, 1.2998),
                box(103.8550, 1.2902, 103.8600, 1.2948),
                box(103.8530, 1.2935, 103.8547, 1.2947),
            ],
        },
        crs=crs,
    )
    water = gpd.GeoDataFrame(
        {"natural": ["water"], "geometry": [box(103.8592, 1.2900, 103.8600, 1.3000)]},
        crs=crs,
    )
    green = landuse[landuse["landuse_class"] == "green"].copy()
    rail = gpd.GeoDataFrame(
        {"railway": ["rail"], "geometry": [LineString([(103.8500, 1.3000), (103.8600, 1.2900)])]},
        crs=crs,
    )
    return CityLayers(roads, buildings, landuse, landuse.copy(), water, green, rail)


def run_demo(output: str | Path, overwrite: bool = False) -> dict:
    output_path = Path(output).expanduser().resolve()
    config = BuildConfig(
        project=ProjectConfig(city_id="demo_singapore"),
        input=InputConfig(
            pbf_path=Path("unused-demo.osm.pbf"),
            bbox_wgs84=(103.8460, 1.2860, 103.8640, 1.3040),
            metric_crs="auto",
        ),
        output=OutputConfig(
            root=output_path,
            overwrite=overwrite,
            save_extracted_gpkg=True,
            save_tile_vectors=True,
        ),
        raster=RasterConfig(tile_size_m=1024, pixels=256, stride_m=1024),
        roads=RoadConfig(),
        heights=HeightConfig(default_by_building={"apartments": 24.8, "residential": 9.3}),
        quality=QualityConfig(minimum_buildings=1, minimum_road_length_m=1),
    )
    return run_build(config, extracted_layers=create_demo_layers())
