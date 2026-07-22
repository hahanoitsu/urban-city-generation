import json

import numpy as np

from urban_dataset.demo import run_demo
from urban_dataset.schema import CHANNEL_NAMES


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
        assert np.isfinite(archive["layers"]).all()
    metadata = json.loads((tile / "metadata.json").read_text())
    assert metadata["metres_per_pixel"] == 4.0
    city = json.loads((tile / "city.json").read_text())
    assert city["roads"]
    assert "estimated_width_m" in city["roads"][0]["properties"]
    assert city["buildings"]
    assert "estimated_height_m" in city["buildings"][0]["properties"]
    assert "height_confidence" in city["buildings"][0]["properties"]
    assert (tile / "preview.png").exists()
