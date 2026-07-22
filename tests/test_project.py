import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from urban_dataset.extract import CityLayers
from urban_dataset.project import project_and_clip_layers


def _empty():
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")


def test_invalid_polygon_is_repaired_before_clipping():
    bow_tie = Polygon(
        [
            (103.84, 1.29),
            (103.85, 1.30),
            (103.84, 1.30),
            (103.85, 1.29),
            (103.84, 1.29),
        ]
    )
    roads = gpd.GeoDataFrame(
        {
            "road_class": ["local"],
            "geometry": [LineString([(103.84, 1.29), (103.85, 1.30)])],
        },
        crs="EPSG:4326",
    )
    buildings = gpd.GeoDataFrame(
        {"building": ["yes"], "geometry": [bow_tie]}, crs="EPSG:4326"
    )
    layers = CityLayers(
        roads=roads,
        buildings=buildings,
        landuse=_empty(),
        landuse_known=_empty(),
        water=_empty(),
        green=_empty(),
        rail=_empty(),
    )

    projected, _ = project_and_clip_layers(
        layers,
        (103.83, 1.28, 103.86, 1.31),
        CRS.from_epsg(32648),
    )
    assert not projected.buildings.empty
    assert projected.buildings.geometry.is_valid.all()
