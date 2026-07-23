from __future__ import annotations

import json
from pathlib import Path

import pytest

from urban_ai.codec import CommandCodecConfig, encode_program
from urban_ai.conversion import city_state_to_program
from urban_ai.prepare import prepare_program_dataset
from urban_ai.schema import ProgramConfig
from urban_ai.validation import program_to_city_state, validate_program


def sample_state() -> dict:
    return {
        "format": "urban-city-state-tile",
        "version": "0.1.0",
        "tile": {"tile_id": "tile-1", "city_id": "test-city", "area_id": "center"},
        "coordinate_system": {
            "units": "metres",
            "local_bounds": [0.0, 0.0, 100.0, 100.0],
            "axis_convention": "x-east, y-north, z-up",
        },
        "transport_graph": {
            "nodes": [
                {"id": "r0", "position_local_m": [10.0, 50.0, 0.0], "degree": 1},
                {"id": "r1", "position_local_m": [50.0, 50.0, 0.0], "degree": 2},
                {"id": "r2", "position_local_m": [90.0, 50.0, 0.0], "degree": 1},
                {"id": "u0", "position_local_m": [50.0, 10.0, -12.0], "degree": 1},
                {"id": "u1", "position_local_m": [50.0, 90.0, -12.0], "degree": 1},
            ],
            "edges": [
                {
                    "id": "road-a",
                    "from_node": "r0",
                    "to_node": "r1",
                    "transport_mode": "road",
                    "class": "major",
                    "vertical_mode": "surface",
                    "layer_order": 0,
                    "width_m": 14.0,
                    "length_m": 40.0,
                    "geometry_local_m": [[10.0, 50.0, 0.0], [50.0, 50.0, 0.0]],
                },
                {
                    "id": "road-b",
                    "from_node": "r1",
                    "to_node": "r2",
                    "transport_mode": "road",
                    "class": "secondary",
                    "vertical_mode": "surface",
                    "layer_order": 0,
                    "width_m": 10.0,
                    "length_m": 40.0,
                    "geometry_local_m": [[50.0, 50.0, 0.0], [90.0, 50.0, 0.0]],
                },
                {
                    "id": "rail-a",
                    "from_node": "u0",
                    "to_node": "u1",
                    "transport_mode": "rail",
                    "class": "subway",
                    "vertical_mode": "underground",
                    "layer_order": -2,
                    "width_m": 6.0,
                    "length_m": 80.0,
                    "geometry_local_m": [[50.0, 10.0, -12.0], [50.0, 90.0, -12.0]],
                },
            ],
        },
        "building_solids": [
            {
                "id": "b1",
                "footprint_local_m": {
                    "type": "Polygon",
                    "coordinates": [[[40, 40], [60, 40], [60, 60], [40, 60], [40, 40]]],
                },
                "height_m": 30.0,
            }
        ],
        "water": [],
        "green": [],
    }


def test_graph_program_roundtrip_preserves_connectivity_and_vertical_mode() -> None:
    config = ProgramConfig(coordinate_bins=128, simplify_tolerance_m=1.0)
    program = city_state_to_program(sample_state(), config)
    validate_program(program, config)
    encoded = encode_program(program, CommandCodecConfig(program=config, maximum_nodes=32))
    city = program_to_city_state(program, config)

    assert encoded["op"][0] == 1
    assert encoded["op"][-1] == 5
    assert city["statistics"]["nodes"] == 5
    assert city["statistics"]["edges"] == 3
    assert city["statistics"]["components"] == 2
    assert {edge["vertical_mode"] for edge in city["transport_graph"]["edges"]} == {
        "surface",
        "underground",
    }
    underground = next(
        edge for edge in city["transport_graph"]["edges"] if edge["vertical_mode"] == "underground"
    )
    assert all(point[2] == -12.0 for point in underground["geometry_local_m"])


def test_program_rejects_cross_component_connection() -> None:
    program = city_state_to_program(sample_state())
    roots = [command["node"] for command in program["commands"] if command["op"] == "root"]
    program["commands"].append(
        {
            "op": "connect",
            "from": roots[0],
            "to": roots[1],
            "transport_mode": "road",
            "class": "local",
            "width_bin": 5,
            "vertical_mode": "surface",
            "layer_bin": 5,
        }
    )
    with pytest.raises(ValueError, match="separate components"):
        validate_program(program)


def test_prepare_program_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "processed"
    manifests = tmp_path / "manifests"
    state_dir = dataset / "test-area" / "tiles" / "tile-1"
    state_dir.mkdir(parents=True)
    (state_dir / "layers.npz").write_bytes(b"placeholder")
    (state_dir / "city.json").write_text(json.dumps(sample_state()), encoding="utf-8")
    manifests.mkdir()
    row = {"tile_id": "tile-1", "sample_path": str(state_dir / "layers.npz")}
    for split in ("train", "validation", "test"):
        (manifests / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    output = tmp_path / "programs"
    summary = prepare_program_dataset(
        dataset,
        manifests,
        output,
        maximum_nodes=32,
        maximum_commands=64,
        minimum_nodes=2,
    )
    assert summary["splits"]["train"]["accepted"] == 1
    assert (output / "programs" / "train" / "tile-1.json").exists()
    assert (output / "style-stats.json").exists()
    assert (output / "city-styles.json").exists()
