from urban_dataset.vertical import VerticalMode, classify_vertical_mode


def test_building_passage_stays_on_surface() -> None:
    assert classify_vertical_mode({"tunnel": "building_passage"}) == VerticalMode.SURFACE
    assert classify_vertical_mode({"tunnel": "covered"}) == VerticalMode.SURFACE


def test_conventional_tunnel_is_underground() -> None:
    assert classify_vertical_mode({"tunnel": "yes"}) == VerticalMode.UNDERGROUND
    assert classify_vertical_mode({"location": "underground"}) == VerticalMode.UNDERGROUND


def test_explicit_bridge_remains_elevated() -> None:
    assert classify_vertical_mode({"bridge": "yes"}) == VerticalMode.ELEVATED
