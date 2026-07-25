import json

import numpy as np

from urban_dataset.demo import run_demo
from urban_dataset.schema import CHANNEL_NAMES, VERTICAL_MODE_NAMES


def test_demo_pipeline(tmp_path):
    output = tmp_path / "demo"
    summary = run_demo(output)
    assert summary["accepted_tiles"] >= 1
    index = output / "index.csv"
    assert index.exists()
    tile_dirs = list((output / "tiles").iterdir())
    assert tile_dirs
    tile = tile_dirs[0]
    with np.load(tile / "layers.npz", allow_pickle=False) as archive:
        assert archive["layers"].shape == (len(CHANNEL_NAMES), 256, 256)
        assert archive["height_known_mask"].shape == (256, 256)
        assert archive["height_confidence"].shape == (256, 256)
        assert archive["landuse_known_mask"].shape == (256, 256)
        assert archive["road_centerlines"].shape == (3, 256, 256)
        assert archive["road_vertical_masks"].shape == (len(VERTICAL_MODE_NAMES), 256, 256)
        assert archive["rail_vertical_masks"].shape == (len(VERTICAL_MODE_NAMES), 256, 256)
        assert archive["surface_transport_reservation"].shape == (256, 256)
        assert archive["buildable_surface_mask"].shape == (256, 256)
        assert archive["buildability_known_mask"].shape == (256, 256)
        assert np.isfinite(archive["layers"]).all()
    metadata = json.loads((tile / "metadata.json").read_text())
    assert metadata["metres_per_pixel"] == 4.0
    assert metadata["vertical_modes"] == list(VERTICAL_MODE_NAMES)
    city = json.loads((tile / "city.json").read_text())
    assert city["format"] == "urban-city-state-tile"
    assert city["coordinate_system"]["axis_convention"] == "x-east, y-north, z-up"
    assert city["roads"]
    assert "estimated_width_m" in city["roads"][0]["properties"]
    assert city["roads"][0]["properties"]["vertical_mode"] == "surface"
    assert city["rail"]
    assert city["rail"][0]["properties"]["vertical_mode"] == "surface"
    assert city["buildings"]
    assert "estimated_height_m" in city["buildings"][0]["properties"]
    assert "height_confidence" in city["buildings"][0]["properties"]
    assert city["transport_graph"]["edges"]
    assert city["building_solids"]
    assert (tile / "preview.png").exists()
