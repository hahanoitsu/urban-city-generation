from __future__ import annotations

import json

from urban_dataset.obj_export import export_city_state_obj


def test_export_city_state_obj_creates_building_and_transport_meshes(tmp_path):
    state = {
        "format": "urban-city-state-tile",
        "version": "0.1.0",
        "building_solids": [
            {
                "id": "building-one",
                "footprint_local_m": {
                    "type": "Polygon",
                    "coordinates": [[[10, 10], [30, 10], [30, 30], [10, 30], [10, 10]]],
                },
                "base_z_m": 0.0,
                "height_m": 18.0,
            }
        ],
        "transport_graph": {
            "edges": [
                {
                    "id": "road-one",
                    "transport_mode": "road",
                    "vertical_mode": "surface",
                    "width_m": 6.0,
                    "geometry_local_m": [[0.0, 50.0, 0.0], [100.0, 50.0, 0.0]],
                },
                {
                    "id": "rail-underground",
                    "transport_mode": "rail",
                    "vertical_mode": "underground",
                    "width_m": 5.0,
                    "geometry_local_m": [[50.0, 0.0, -12.0], [50.0, 100.0, -12.0]],
                },
                {
                    "id": "unknown-edge",
                    "transport_mode": "road",
                    "vertical_mode": "unknown",
                    "width_m": 5.0,
                    "geometry_local_m": [[0.0, 0.0, None], [10.0, 10.0, None]],
                },
            ]
        },
    }
    state_path = tmp_path / "city.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    result = export_city_state_obj(state_path, tmp_path / "preview.obj")

    obj_path = tmp_path / "preview.obj"
    material_path = tmp_path / "preview.mtl"
    assert obj_path.exists()
    assert material_path.exists()
    assert result["vertices"] > 0
    assert result["faces"] > 0
    assert result["building_parts"] == 1
    assert result["transport_ribbons"] == 2
    assert result["skipped_unknown_vertical_edges"] == 1

    obj = obj_path.read_text(encoding="utf-8")
    assert "usemtl building" in obj
    assert "usemtl road_surface" in obj
    assert "usemtl rail_underground" in obj
    assert "\nv " in obj
    assert "\nf " in obj
