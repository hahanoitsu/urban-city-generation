import numpy as np

from urban_analysis.surface_roundtrip import audit_classes


def test_surface_roundtrip_keeps_main_structure():
    classes = np.zeros((64, 64), dtype=np.uint8)

    classes[5:20, 5:20] = 1
    classes[35:50, 5:20] = 7
    classes[8:18, 40:50] = 2
    classes[30:42, 38:52] = 2

    classes[30, 5:59] = 3
    classes[12:31, 30] = 4
    classes[30:55, 45] = 5

    _state, metrics, _masks = audit_classes(classes)

    assert metrics["vegetation_iou"] > 0.95
    assert metrics["building_iou"] > 0.95
    assert metrics["water_iou"] > 0.95
    assert metrics["raw_road_components"] == 1
    assert metrics["cleaned_road_components"] == 1
    assert metrics["vector_road_components"] == 1


def test_short_junction_branch_is_not_lost():
    classes = np.zeros((64, 64), dtype=np.uint8)
    classes[32, 8:56] = 3
    classes[24:33, 32] = 5

    state, metrics, _masks = audit_classes(classes)

    road_edges = [
        edge
        for edge in state["transport_graph"]["edges"]
        if edge["transport_mode"] == "road"
    ]

    assert metrics["vector_road_components"] == 1
    assert len(road_edges) >= 3
