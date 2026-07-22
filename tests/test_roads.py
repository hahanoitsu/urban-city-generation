from urban_dataset.classify import estimate_road_width_metres, road_is_usable


def test_minor_service_roads_are_filtered():
    assert not road_is_usable({"highway": "service", "service": "parking_aisle"})
    assert not road_is_usable({"highway": "service", "service": "driveway"})
    assert not road_is_usable({"highway": "service", "access": "private"})
    assert road_is_usable({"highway": "service", "service": "access"})
    assert road_is_usable({"highway": "residential"})


def test_road_width_uses_explicit_then_lanes_then_highway_default():
    defaults = {"major": 8.0, "secondary": 6.5, "local": 5.0}
    assert estimate_road_width_metres({"highway": "primary", "width": "10 m"}, class_fallbacks=defaults) == 10
    assert estimate_road_width_metres({"highway": "primary", "lanes": "2"}, class_fallbacks=defaults) == 7.3
    assert estimate_road_width_metres({"highway": "motorway"}, class_fallbacks=defaults) == 8.5
    assert estimate_road_width_metres({"highway": "service"}, class_fallbacks=defaults) == 3.5
