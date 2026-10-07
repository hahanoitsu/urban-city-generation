from urban_analysis.structured_generation_audit import audit_scene, summarise


def square(x, y, size):
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]


def scene():
    bowtie = [[0, 0], [10, 10], [10, 0], [0, 10]]
    return {
        "nodes": [
            {"id": 0, "xyz_m": [10.0, 10.0, None]},
            {"id": 1, "xyz_m": [100.0, 10.0, None]},
            {"id": 2, "xyz_m": [50.0, 50.0, None]},
        ],
        "edges": [
            {
                "from": 0,
                "to": 1,
                "spline_xyz_m": [[10.0, 10.0, None], [55.0, 10.0, None], [100.0, 10.0, None]],
            }
        ],
        "buildings": [
            {"footprint_xy_m": square(20, 20, 10), "height_m": 12.0},
            {"footprint_xy_m": bowtie, "height_m": -3.0},
            {"footprint_xy_m": square(505, 505, 20), "height_m": 5.0},
        ],
        "areas": [{"kind": "green", "polygon_xy_m": square(100, 100, 50)}],
    }


def test_polygons_and_heights():
    row = audit_scene(scene(), 512.0)
    assert row["buildings"]["count"] == 3
    assert row["buildings"]["valid"] == 2
    assert row["buildings"]["outside"] == 1
    assert row["negative_heights"] == 1
    assert row["areas"]["valid"] == 1


def test_transport():
    row = audit_scene(scene(), 512.0)
    transport = row["transport"]
    assert transport["nodes"] == 3
    assert transport["isolated_nodes"] == 1
    assert transport["components"] == 2
    assert transport["largest_component_nodes"] == 2
    assert abs(transport["tortuosity"][0] - 1.0) < 1e-9


def test_summary():
    summary = summarise([audit_scene(scene(), 512.0)])
    assert abs(summary["buildings"]["valid_fraction"] - 2 / 3) < 1e-9
    assert abs(summary["transport"]["isolated_node_fraction"] - 1 / 3) < 1e-9
