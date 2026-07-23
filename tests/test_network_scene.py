import json
from pathlib import Path

from shapely.geometry import box, mapping

from urban_dataset.network_scene import build_network_scene


def _node(
    node_id: str,
    x: float,
    y: float,
    *,
    mode: str = "surface",
    boundary: str | None = None,
):
    return {
        "id": node_id,
        "transport_mode": "road",
        "vertical_mode": mode,
        "layer_order": 0,
        "position_projected_m": [x, y, 8.0 if mode == "elevated" else 0.0],
        "boundary_port_key": boundary,
    }


def _edge(
    edge_id: str,
    start: str,
    end: str,
    coordinates: list[list[float]],
    *,
    mode: str = "surface",
):
    return {
        "id": edge_id,
        "from_node": start,
        "to_node": end,
        "transport_mode": "road",
        "class": "local",
        "vertical_mode": mode,
        "layer_order": 0,
        "width_m": 6.0,
        "geometry_local_m": coordinates,
    }


def _state(
    root: Path,
    tile_id: str,
    origin_x: float,
    nodes: list[dict],
    edges: list[dict],
):
    tile_dir = root / "singapore-central" / "tiles" / tile_id
    tile_dir.mkdir(parents=True)
    payload = {
        "format": "urban-city-state-tile",
        "version": "0.1.0",
        "tile": {
            "tile_id": tile_id,
            "city_id": "singapore",
            "area_id": "central",
        },
        "coordinate_system": {
            "units": "metres",
            "origin_projected": [origin_x, 0.0],
            "source_projected_crs": "EPSG:3414",
            "local_bounds": [0.0, 0.0, 100.0, 100.0],
        },
        "transport_graph": {"nodes": nodes, "edges": edges},
        "water": [],
        "building_solids": [
            {
                "id": f"building-{tile_id}",
                "source_id": tile_id,
                "building_type": "yes",
                "base_z_m": 0.0,
                "height_m": 12.0,
                "footprint_local_m": mapping(box(20, 20, 35, 35)),
            }
        ],
    }
    (tile_dir / "city.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_scene_stitches_ports_and_preserves_grade_separation(tmp_path):
    dataset = tmp_path / "corpus"
    shared_port = "port-shared"
    _state(
        dataset,
        "west",
        0.0,
        [
            _node("w0", 10, 50),
            _node("w1", 100, 50, boundary=shared_port),
            _node("e0", 50, 0, mode="elevated"),
            _node("e1", 50, 100, mode="elevated"),
        ],
        [
            _edge("west-road", "w0", "w1", [[10, 50, 0], [100, 50, 0]]),
            _edge(
                "flyover",
                "e0",
                "e1",
                [[50, 0, 8], [50, 100, 8]],
                mode="elevated",
            ),
        ],
    )
    _state(
        dataset,
        "east",
        100.0,
        [
            _node("e2", 100, 50, boundary=shared_port),
            _node("e3", 190, 50),
        ],
        [_edge("east-road", "e2", "e3", [[0, 50, 0], [90, 50, 0]])],
    )

    output = tmp_path / "scene"
    result = build_network_scene(
        dataset,
        output,
        city_id="singapore",
        area_id="central",
        minimum_block_area_m2=100.0,
        target_parcel_area_m2=500.0,
        minimum_parcel_area_m2=50.0,
    )

    assert result["tiles"] == 2
    assert result["stitched_nodes"] == 1
    assert result["grade_separated_edges"] == 1
    assert result["blocks"] >= 1
    assert result["parcels"] >= result["blocks"]
    assert result["buildings"] == 2
    assert (output / "network.json").exists()
    assert (output / "plan.png").exists()
    assert (output / "city.obj").exists()
    assert (output / "city.mtl").exists()

    network = json.loads((output / "network.json").read_text(encoding="utf-8"))
    stitched = [
        node
        for node in network["transport_graph"]["nodes"]
        if node["stitched_sources"] > 1
    ]
    assert len(stitched) == 1
    assert stitched[0]["degree"] == 2
