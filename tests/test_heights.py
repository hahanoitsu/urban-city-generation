from urban_dataset.heights import estimate_building_height, parse_height_metres


def test_parse_height_units():
    assert parse_height_metres("24 m") == 24
    assert round(parse_height_metres("30 ft"), 3) == 9.144
    assert parse_height_metres("unknown") is None


def test_height_priority():
    row = {"height": "18", "building:levels": "20", "building": "apartments"}
    result = estimate_building_height(
        row,
        floor_height_m=3.1,
        default_height_m=9.3,
        default_by_building={"apartments": 24.8},
    )
    assert result.metres == 18
    assert result.observed
    assert result.source == "height"


def test_level_and_default_height():
    level_result = estimate_building_height(
        {"building:levels": "4", "building": "residential"},
        floor_height_m=3.1,
        default_height_m=9.3,
        default_by_building={},
    )
    assert level_result.metres == 12.4
    assert level_result.observed

    fallback = estimate_building_height(
        {"building": "apartments"},
        floor_height_m=3.1,
        default_height_m=9.3,
        default_by_building={"apartments": 24.8},
    )
    assert fallback.metres == 24.8
    assert not fallback.observed
