import geopandas as gpd
import numpy as np
import pyogrio
import pytest
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from urban_dataset.classify import clean_tag
from urban_dataset.extract import _expand_other_tags, _extract_with_gdal, _parse_other_tags, station_kind


def test_parse_other_tags_restores_duplicate_suffix_alias():
    parsed = _parse_other_tags('"service_1"=>"parking_aisle","lanes"=>"2"')
    assert parsed["service_1"] == "parking_aisle"
    assert parsed["service"] == "parking_aisle"
    assert parsed["lanes"] == "2"


def test_expand_other_tags_uses_way_ids_for_multipolygons():
    frame = gpd.GeoDataFrame(
        {
            "osm_id": [None, "99"],
            "osm_way_id": ["42", None],
            "other_tags": ['"building"=>"yes"', '"landuse"=>"residential"'],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
        },
        crs="EPSG:4326",
    )
    result = _expand_other_tags(frame, ["building", "landuse"])
    assert result["id"].tolist() == ["42", "99"]
    assert result["osm_type"].tolist() == ["way", "relation"]


def test_expand_other_tags_treats_gdal_lines_as_ways():
    frame = gpd.GeoDataFrame(
        {
            "osm_id": ["123"],
            "other_tags": ['"highway"=>"primary","lanes"=>"3"'],
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        crs="EPSG:4326",
    )
    result = _expand_other_tags(frame, ["highway", "lanes"])
    assert result.loc[0, "id"] == "123"
    assert result.loc[0, "osm_type"] == "way"
    assert result.loc[0, "highway"] == "primary"


def test_clean_tag_treats_nan_as_missing():
    assert clean_tag(np.nan) == ""
    assert clean_tag(None) == ""


def test_station_kind_reads_the_station_tags():
    assert station_kind({"railway": "station", "station": "subway"}) == "subway"
    assert station_kind({"railway": "station", "light_rail": "yes"}) == "light_rail"
    assert station_kind({"railway": "tram_stop"}) == "tram"
    assert station_kind({"railway": "halt"}) == "rail"


def write_osm(path, nodes, ways):
    tags = lambda values: "".join(f'<tag k="{key}" v="{value}"/>' for key, value in values.items())
    rows = [f'<node id="{number}" visible="true" version="1" lat="{lat}" lon="{lon}">{tags(values)}</node>' for number, lon, lat, values in nodes]
    rows += [f'<way id="{number}" visible="true" version="1">{"".join(f"<nd ref=\"{ref}\"/>" for ref in refs)}{tags(values)}</way>' for number, refs, values in ways]
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="test">\n' + "\n".join(rows) + "\n</osm>\n")
    return path


def test_stations_and_monorail_come_out_of_an_osm_file(tmp_path):
    if "OSM" not in pyogrio.list_drivers():
        pytest.skip("no osm driver")
    nodes = [
        (1, 103.800, 1.300, {}),
        (2, 103.820, 1.300, {}),
        (3, 103.800, 1.305, {}),
        (4, 103.820, 1.305, {}),
        (5, 103.800, 1.310, {}),
        (6, 103.820, 1.310, {}),
        (10, 103.805, 1.305, {"railway": "station", "station": "subway", "name": "first"}),
        (11, 103.810, 1.310, {"railway": "tram_stop"}),
        (12, 103.812, 1.302, {"railway": "level_crossing"}),
        (13, 103.811, 1.301, {"amenity": "cafe"}),
        (20, 103.8080, 1.3020, {}),
        (21, 103.8084, 1.3020, {}),
        (22, 103.8084, 1.3024, {}),
        (23, 103.8080, 1.3024, {}),
    ]
    ways = [
        (100, [1, 2], {"highway": "primary"}),
        (101, [3, 4], {"railway": "subway", "tunnel": "yes"}),
        (102, [5, 6], {"railway": "monorail", "bridge": "yes"}),
        (103, [20, 21, 22, 23, 20], {"building": "train_station", "railway": "station"}),
    ]
    layers = _extract_with_gdal(write_osm(tmp_path / "tiny.osm", nodes, ways), (103.79, 1.29, 103.83, 1.32))
    assert sorted(layers.stations["station_kind"]) == ["rail", "subway", "tram"]
    assert set(layers.stations.geometry.geom_type) == {"Point"}
    assert sorted(layers.rail["railway"]) == ["monorail", "subway"]
    assert len(layers.roads) == 1 and len(layers.buildings) == 1
    assert layers.water.empty


def coast(backwards):
    nodes = [(1 + index, lon, lat, {}) for index, (lon, lat) in enumerate([(103.780, 1.300), (103.800, 1.298), (103.810, 1.302), (103.820, 1.299), (103.840, 1.300)])]
    nodes += [(20 + index, lon, lat, {}) for index, (lon, lat) in enumerate([(103.805, 1.293), (103.808, 1.293), (103.808, 1.295), (103.805, 1.295)])]
    ways = [(100, [5, 4, 3, 2, 1] if backwards else [1, 2, 3, 4, 5], {"natural": "coastline"}), (101, [20, 21, 22, 23, 20], {"natural": "coastline"})]
    for row in range(8):
        nodes += [(40 + 2 * row, 103.792, 1.303 + row * 0.002, {}), (41 + 2 * row, 103.828, 1.303 + row * 0.002, {})]
        ways.append((200 + row, [40 + 2 * row, 41 + 2 * row], {"highway": "residential"}))
    return nodes, ways


def test_the_sea_is_the_right_hand_side_of_the_coastline(tmp_path):
    if "OSM" not in pyogrio.list_drivers():
        pytest.skip("no osm driver")
    bbox = (103.79, 1.29, 103.83, 1.32)
    layers = _extract_with_gdal(write_osm(tmp_path / "coast.osm", *coast(False)), bbox)
    sea = unary_union(list(layers.water.geometry))
    assert set(layers.water["natural"]) == {"sea"}
    assert sea.contains(Point(103.815, 1.292))
    assert not sea.contains(Point(103.815, 1.310))
    assert not sea.contains(Point(103.8065, 1.294))
    assert 0.25 < sea.area / box(*bbox).area < 0.35


def test_a_built_up_side_is_never_called_sea(tmp_path):
    if "OSM" not in pyogrio.list_drivers():
        pytest.skip("no osm driver")
    layers = _extract_with_gdal(write_osm(tmp_path / "coast.osm", *coast(True)), (103.79, 1.29, 103.83, 1.32))
    assert layers.water.empty
