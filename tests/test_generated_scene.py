from __future__ import annotations

from pathlib import Path

from urban_ai.scene import compile_generated_city, export_generated_city_obj, render_generated_city
from urban_ai.schema import ProgramConfig
from urban_ai.validation import program_to_city_state


def test_generated_city_compiles_to_blocks_buildings_preview_and_obj(tmp_path: Path) -> None:
    program = {
        "format": "urban-graph-program",
        "version": "0.2.0",
        "bounds_m": [0.0, 0.0, 200.0, 200.0],
        "source": {"kind": "generated", "seed": 3},
        "program_config": ProgramConfig(coordinate_bins=32).to_dict(),
        "style": {"building_coverage": 0.3, "mean_building_height_m": 20.0},
        "commands": [
            {
                "op": "root",
                "node": 0,
                "x_bin": 0,
                "y_bin": 16,
                "transport_mode": "road",
                "vertical_mode": "surface",
                "layer_bin": 5,
            },
            {
                "op": "add",
                "node": 1,
                "parent": 0,
                "x_bin": 31,
                "y_bin": 16,
                "transport_mode": "road",
                "class": "major",
                "width_bin": 12,
                "vertical_mode": "surface",
                "layer_bin": 5,
            },
        ],
    }
    city = compile_generated_city(program_to_city_state(program), seed=3)
    preview = render_generated_city(city, tmp_path / "preview.png")
    obj = export_generated_city_obj(city, tmp_path / "city.obj")
    assert city["statistics"]["blocks"] >= 2
    assert city["statistics"]["parcels"] > 0
    assert city["statistics"]["buildings"] > 0
    assert Path(preview["preview"]).exists()
    assert Path(obj["obj"]).exists()
    assert Path(obj["material"]).exists()
