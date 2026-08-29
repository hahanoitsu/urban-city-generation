from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import mapping, box

from urban_analysis.generated_city_audit import audit_state


def _state() -> dict:
    return {
        "tile": {"tile_id": "synthetic"},
        "coordinate_system": {"local_bounds": [0.0, 0.0, 1024.0, 1024.0]},
        "transport_graph": {
            "nodes": [
                {
                    "id": "a",
                    "transport_mode": "road",
                    "vertical_mode": "surface",
                    "position_local_m": [0.0, 100.0, 0.0],
                },
                {
                    "id": "b",
                    "transport_mode": "road",
                    "vertical_mode": "surface",
                    "position_local_m": [500.0, 100.0, 0.0],
                },
                {
                    "id": "c",
                    "transport_mode": "road",
                    "vertical_mode": "surface",
                    "position_local_m": [900.0, 100.0, 0.0],
                },
            ],
            "edges": [
                {
                    "from_node": "a",
                    "to_node": "b",
                    "transport_mode": "road",
                    "class": "major",
                    "vertical_mode": "surface",
                    "length_m": 500.0,
                    "geometry_local_m": [[0.0, 100.0, 0.0], [500.0, 100.0, 0.0]],
                },
                {
                    "from_node": "b",
                    "to_node": "c",
                    "transport_mode": "road",
                    "class": "local",
                    "vertical_mode": "surface",
                    "length_m": 400.0,
                    "geometry_local_m": [[500.0, 100.0, 0.0], [900.0, 100.0, 0.0]],
                },
            ],
        },
        "building_solids": [
            {
                "footprint_local_m": mapping(box(100.0, 110.0, 140.0, 150.0)),
                "height_m": 12.0,
            },
            {
                "footprint_local_m": mapping(box(700.0, 110.0, 740.0, 150.0)),
                "height_m": 18.0,
            },
        ],
    }


def test_generated_city_audit_detects_served_connected_network(tmp_path: Path):
    path = tmp_path / "city.json"
    path.write_text(json.dumps(_state()))

    result = audit_state(path, "generated")

    assert result["road_components"] == 1
    assert result["road_largest_length_fraction"] == 1.0
    assert result["road_interior_component_length_fraction"] == 0.0
    assert result["buildings_within_20m_road_fraction"] == 1.0
    assert result["local_length_connected_to_higher_fraction"] == 1.0
    assert result["road_component_length_serving_buildings_fraction"] == 1.0
