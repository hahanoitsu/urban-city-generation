from __future__ import annotations

import random

import networkx as nx

from urban_ai.planner import _build_topology, _state_from_topology
from urban_analysis.structural_generator_audit import _crossing_metrics


def _profile():
    return {
        "tile_id": "synthetic",
        "road_length_m": 8000.0,
        "boundary_endpoints": 8,
        "junctions_per_km": 6.0,
        "interior_dead_ends_per_km": 1.5,
        "style": {
            "road_length_per_km2": 8.0,
            "rail_length_per_km2": 0.0,
            "major_fraction": 0.20,
            "secondary_fraction": 0.18,
            "local_fraction": 0.62,
            "surface_fraction": 1.0,
            "underground_fraction": 0.0,
            "elevated_fraction": 0.0,
            "intersection_density_per_km2": 48.0,
            "mean_edge_length_m": 70.0,
            "building_coverage": 0.2,
            "mean_building_height_m": 16.0,
            "water_coverage": 0.0,
            "green_coverage": 0.2,
        },
    }


def test_planner_builds_connected_city_spanning_planar_network():
    bounds = [0.0, 0.0, 1024.0, 1024.0]
    graph = _build_topology(
        _profile(),
        bounds,
        random.Random(41),
        orientation_strength=0.35,
        margin_m=45.0,
        maximum_attempts=12,
    )

    assert nx.is_connected(graph)

    state = _state_from_topology(graph, _profile(), bounds, seed=41)
    nodes = state["transport_graph"]["nodes"]
    edges = state["transport_graph"]["edges"]

    boundary = [
        node
        for node in nodes
        if node["degree"] == 1
        and (
            abs(node["position_local_m"][0]) < 1e-6
            or abs(node["position_local_m"][0] - 1024.0) < 1e-6
            or abs(node["position_local_m"][1]) < 1e-6
            or abs(node["position_local_m"][1] - 1024.0) < 1e-6
        )
    ]
    assert len(boundary) == 8
    assert _crossing_metrics(edges)["unnoded_crossings"] == 0

    xs = [node["position_local_m"][0] for node in nodes]
    ys = [node["position_local_m"][1] for node in nodes]
    assert (max(xs) - min(xs)) / 1024.0 > 0.9
    assert (max(ys) - min(ys)) / 1024.0 > 0.9


def test_planner_assigns_all_three_road_classes():
    bounds = [0.0, 0.0, 1024.0, 1024.0]
    graph = _build_topology(
        _profile(),
        bounds,
        random.Random(71),
        orientation_strength=0.35,
        margin_m=45.0,
        maximum_attempts=12,
    )
    state = _state_from_topology(graph, _profile(), bounds, seed=71)
    classes = {edge["class"] for edge in state["transport_graph"]["edges"]}
    assert classes == {"major", "secondary", "local"}


def test_planner_handles_sparse_realistic_main_road_profile():
    bounds = [0.0, 0.0, 1024.0, 1024.0]
    profile = _profile()
    profile["road_length_m"] = 1700.0
    profile["boundary_endpoints"] = 2
    profile["junctions_per_km"] = 1.8
    profile["interior_dead_ends_per_km"] = 0.5

    graph = _build_topology(
        profile,
        bounds,
        random.Random(19),
        orientation_strength=0.35,
        margin_m=45.0,
        maximum_attempts=12,
    )

    assert nx.is_connected(graph)
    total = sum(float(data["length_m"]) for *_ends, data in graph.edges(data=True))
    assert 850.0 <= total <= 3060.0
