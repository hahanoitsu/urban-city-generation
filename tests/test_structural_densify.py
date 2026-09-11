from __future__ import annotations

import math

from urban_ai.conversion import city_state_to_program
from urban_ai.schema import ProgramConfig


def test_long_surface_road_is_subdivided_into_local_steps():
    state = {
        "tile": {"city_id": "x", "area_id": "x", "tile_id": "x"},
        "coordinate_system": {"local_bounds": [0.0, 0.0, 1024.0, 1024.0]},
        "transport_graph": {
            "nodes": [
                {
                    "id": "a",
                    "position_local_m": [100.0, 100.0, 0.0],
                    "transport_mode": "road",
                    "vertical_mode": "surface",
                    "degree": 1,
                },
                {
                    "id": "b",
                    "position_local_m": [700.0, 100.0, 0.0],
                    "transport_mode": "road",
                    "vertical_mode": "surface",
                    "degree": 1,
                },
            ],
            "edges": [
                {
                    "id": "ab",
                    "from_node": "a",
                    "to_node": "b",
                    "transport_mode": "road",
                    "class": "major",
                    "vertical_mode": "surface",
                    "width_m": 18.0,
                    "length_m": 600.0,
                    "geometry_local_m": [
                        [100.0, 100.0, 0.0],
                        [700.0, 100.0, 0.0],
                    ],
                }
            ],
        },
        "building_solids": [],
        "water": [],
        "green": [],
    }

    config = ProgramConfig(
        coordinate_bins=256,
        simplify_tolerance_m=12.0,
        maximum_segment_length_m=200.0,
        relative_add_coordinates=True,
    )
    program = city_state_to_program(state, config)
    positions = {}
    lengths = []

    for command in program["commands"]:
        if command["op"] == "root":
            positions[command["node"]] = (command["x_bin"], command["y_bin"])
        elif command["op"] == "add":
            parent = positions[command["parent"]]
            current = (command["x_bin"], command["y_bin"])
            positions[command["node"]] = current
            lengths.append(math.hypot(current[0] - parent[0], current[1] - parent[1]))

    metres_per_bin = 1024.0 / 255.0
    assert len(lengths) >= 3
    assert max(lengths) * metres_per_bin <= 205.0
