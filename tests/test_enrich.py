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
from urban_dataset.enrich import enrich_buildings, enrich_roads


def _config(tmp_path):
    return BuildConfig(
        project=ProjectConfig(city_id="test"),
        input=InputConfig(
            pbf_path=tmp_path / "source.osm.pbf",
            bbox_wgs84=(103.8, 1.2, 103.9, 1.3),
        ),
        output=OutputConfig(root=tmp_path / "output"),
        raster=RasterConfig(),
        roads=RoadConfig(),
        heights=HeightConfig(
            floor_height_m=3.0,
            default_height_m=9.0,
            default_by_building={"apartments": 21.0},
        ),
        quality=QualityConfig(),
    )


def test_road_width_priority(tmp_path):
    roads = gpd.GeoDataFrame(
        {
            "highway": ["primary", "residential", "service"],
            "road_class": ["major", "local", "local"],
            "width": ["10 m", None, None],
            "lanes": ["8", "2", None],
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 1), (1, 1)]),
                LineString([(0, 2), (1, 2)]),
            ],
        },
        crs="EPSG:32648",
    )
    result = enrich_roads(roads, _config(tmp_path))
    assert result["estimated_width_m"].tolist() == [10.0, 7.3, 3.5]


def test_building_height_uses_type_median(tmp_path):
    buildings = gpd.GeoDataFrame(
        {
            "building": ["apartments"] * 4 + ["house"],
            "height": ["30 m", None, None, None, None],
            "building:levels": [None, "8", "10", None, None],
            "geometry": [box(index, 0, index + 0.5, 0.5) for index in range(5)],
        },
        crs="EPSG:32648",
    )
    result = enrich_buildings(buildings, _config(tmp_path))
    assert result.loc[0, "height_confidence"] == 3
    assert result.loc[1, "estimated_height_m"] == 24.0
    assert result.loc[2, "estimated_height_m"] == 30.0
    assert result.loc[3, "estimated_height_m"] == 30.0
    assert result.loc[3, "height_confidence"] == 1
    assert result.loc[4, "estimated_height_m"] == 9.0
