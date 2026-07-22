import geopandas as gpd
from shapely.geometry import LineString, Polygon

from urban_dataset.extract import _expand_other_tags, _parse_other_tags


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
