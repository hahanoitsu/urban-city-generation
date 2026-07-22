import sys
import types

import geopandas as gpd
from shapely.geometry import LineString, box

from urban_dataset.extract import extract_from_pbf


class FakeOSM:
    def __init__(self, filepath, **kwargs):
        self.filepath = filepath
        self.kwargs = kwargs

    def get_network(self, **kwargs):
        return gpd.GeoDataFrame(
            {
                "id": [1, 2],
                "highway": ["primary", "footway"],
                "geometry": [
                    LineString([(103.84, 1.29), (103.85, 1.30)]),
                    LineString([(103.84, 1.30), (103.85, 1.31)]),
                ],
            },
            crs="EPSG:4326",
        )

    def get_buildings(self, **kwargs):
        return gpd.GeoDataFrame(
            {
                "id": [3],
                "building": ["apartments"],
                "height": ["30 m"],
                "geometry": [box(103.845, 1.295, 103.846, 1.296)],
            },
            crs="EPSG:4326",
        )

    def get_landuse(self, **kwargs):
        return gpd.GeoDataFrame(
            {
                "id": [4],
                "landuse": ["residential"],
                "geometry": [box(103.84, 1.29, 103.85, 1.30)],
            },
            crs="EPSG:4326",
        )

    def get_natural(self, **kwargs):
        return gpd.GeoDataFrame(
            {
                "id": [5],
                "natural": ["water"],
                "geometry": [box(103.849, 1.29, 103.85, 1.30)],
            },
            crs="EPSG:4326",
        )

    def get_data_by_custom_criteria(self, custom_filter=None, **kwargs):
        key = next(iter(custom_filter))
        if key == "railway":
            return gpd.GeoDataFrame(
                {
                    "id": [6],
                    "railway": ["rail"],
                    "geometry": [LineString([(103.84, 1.291), (103.85, 1.301)])],
                },
                crs="EPSG:4326",
            )
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")


def test_fake_pyrosm_extraction(monkeypatch, tmp_path):
    fake_module = types.ModuleType("pyrosm")
    fake_module.OSM = FakeOSM
    monkeypatch.setitem(sys.modules, "pyrosm", fake_module)
    pbf = tmp_path / "test.osm.pbf"
    pbf.write_bytes(b"fake")
    layers = extract_from_pbf(
        pbf,
        (103.84, 1.29, 103.85, 1.31),
        engine="out_of_core",
        workers="auto",
    )
    assert len(layers.roads) == 1
    assert layers.roads.iloc[0]["road_class"] == "major"
    assert len(layers.buildings) == 1
    assert len(layers.landuse) >= 1
    assert len(layers.water) >= 1
    assert len(layers.rail) == 1
