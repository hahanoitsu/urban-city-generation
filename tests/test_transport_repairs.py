import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import torch
from shapely.geometry import LineString, box

from urban_dataset.city_state import build_transport_graph
from urban_dataset.spatial_world import _context_cells, simplify_transport_graph
from urban_dataset.tile import TileSpec
from urban_dataset.transport_graph import simple_transport_graph
from urban_model.plan_cell_graph import (
    PlanCellGraphArchitect,
    PlanCellGraphConfig,
    build_query_layout,
)
from urban_model.plan_cell_graph_loss import _edge_targets
from urban_model.spatial_world_data import (
    SpatialTensorConfig,
    _prepare_target_graph,
    _resample_line,
)


def roads_graph(lines, **columns):
    count = len(lines)
    roads = gpd.GeoDataFrame(
        {
            "id": list(range(count)),
            "road_class": ["local"] * count,
            "vertical_mode": ["surface"] * count,
            "estimated_width_m": [5.0] * count,
            **columns,
            "geometry": [LineString(line) for line in lines],
        },
        crs="EPSG:3857",
    )
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=roads.crs)
    tile = TileSpec(city_id="test", column=0, row=0, minx=0, miny=0, maxx=100, maxy=100)
    return build_transport_graph(roads, empty, tile)


def test_missing_layer_and_explicit_zero_share_a_junction():
    graph = roads_graph(
        [[(10, 50), (50, 50)], [(50, 50), (90, 50)], [(50, 50), (50, 90)]], layer=[None, 0, None]
    )
    assert len(graph["nodes"]) == 4
    assert sorted(node["degree"] for node in graph["nodes"]) == [1, 1, 1, 3]


def test_source_class_width_and_layer_survive_simplification():
    graph = roads_graph(
        [[(10, 50), (40, 50)], [(40, 50), (60, 50)], [(60, 50), (90, 50)]],
        road_class=["local", "major", "major"],
        estimated_width_m=[5, 10, 12],
        layer=[0, 0, 0],
    )
    graph = simplify_transport_graph(graph)
    assert len(graph["edges"]) == 3
    assert sorted((edge["class"], edge["width_m"]) for edge in graph["edges"]) == [
        ("local", 5),
        ("major", 10),
        ("major", 12),
    ]
    assert all(edge["layer_order"] == 0 for edge in graph["edges"])


def test_unknown_stacking_does_not_join_coincident_endpoints():
    graph = roads_graph(
        [[(10, 50), (50, 50)], [(50, 50), (90, 50)]], vertical_mode=["unknown", "unknown"]
    )
    assert len(graph["nodes"]) == 4


def test_partial_context_cell_does_not_reveal_target_statistics():
    cells = _context_cells(box(0, 0, 100, 100), box(25, 25, 75, 75), 100, {(0, 0): {"roads": 12}})
    assert cells[0]["masked_fraction"] == 0.25
    assert cells[0]["features"] == {}


def test_loop_and_parallel_paths_survive_pair_encoding():
    graph = roads_graph([[(10, 10), (90, 10)]])
    first = graph["edges"][0]
    parallel = {
        **first,
        "id": "parallel",
        "geometry_local_m": [[10, 10, 0], [50, 30, 0], [90, 10, 0]],
    }
    loop = {
        **first,
        "id": "loop",
        "to_node": first["from_node"],
        "geometry_local_m": [[10, 10, 0], [10, 40, 0], [40, 40, 0], [10, 10, 0]],
    }
    graph["edges"].extend([parallel, loop])
    simple = simple_transport_graph(graph)
    pairs = [tuple(sorted((edge["from_node"], edge["to_node"]))) for edge in simple["edges"]]
    assert len(set(pairs)) == len(pairs)
    assert all(a != b for a, b in pairs)
    assert len(simple["edges"]) - len(simple["nodes"]) == len(graph["edges"]) - len(graph["nodes"])
    before = sum(
        LineString([p[:2] for p in edge["geometry_local_m"]]).length for edge in graph["edges"]
    )
    after = sum(edge["length_m"] for edge in simple["edges"])
    assert after == pytest.approx(before)
    assert simple_transport_graph(simple)["edges"] == simple["edges"]


def test_xy_curve_codec_preserves_backtracking_after_resampling():
    graph = roads_graph([[(10, 10), (80, 40), (20, 70), (90, 90)]])
    config = SpatialTensorConfig(target_size_m=100, max_nodes=8, max_edges=8, simple_graph=True)
    sample = _prepare_target_graph({"target": {"transport_graph": graph}}, config)
    batch = {key: value[None] for key, value in sample.items()}
    exists, _, _, _, curve = _edge_targets(batch, 0, 2, torch.tensor([0, 1]))
    spec = importlib.util.spec_from_file_location(
        "sample_graph", Path(__file__).parents[1] / "scripts/sample_plan_cell_graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reconstructed = torch.stack(
        module.curve_points(sample["node_xy"][0], sample["node_xy"][1], curve[0, 1])
    )
    metres = (reconstructed + 1) * 50
    expected = _resample_line(graph["edges"][0]["geometry_local_m"], 10)
    if not np.allclose(expected[0], metres[0]):
        expected = expected[::-1].copy()
    assert exists.sum() == 1
    np.testing.assert_allclose(metres.numpy(), expected, atol=2e-5)


def test_position_arithmetic_does_not_quantize_to_bfloat16_grid():
    model = PlanCellGraphArchitect(
        PlanCellGraphConfig(
            plan_dimensions=8,
            orientation_dimensions=4,
            global_dimensions=8,
            model_dimensions=32,
            edge_dimensions=8,
            heads=4,
            plan_layers=1,
            node_layers=1,
            feedforward_dimensions=64,
        )
    )
    local = torch.tensor([[[0.50390625, 0.6015625]]], dtype=torch.bfloat16)
    xy = model._node_positions(local, torch.tensor([[255]]))
    expected = (torch.tensor([[[15.0, 15.0]]]) + local.float()) / 16 * 2 - 1
    assert xy.dtype == torch.float32
    torch.testing.assert_close(xy, expected, rtol=0, atol=0)


def test_query_layout_rejects_excess_counts():
    with pytest.raises(RuntimeError, match="counts"):
        build_query_layout(torch.ones(1, 4, 1), torch.tensor([2]), max_slots_per_cell=8)
