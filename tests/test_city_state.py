from __future__ import annotations

import json

import geopandas as gpd
from shapely.geometry import LineString, box

from urban_dataset.bundle import compile_city_state_bundle
from urban_dataset.city_state import build_transport_graph, building_solids
from urban_dataset.tile import TileSpec


def _tile() -> TileSpec:
    return TileSpec(
        city_id="test-city",
        column=0,
        row=0,
        minx=0.0,
        miny=0.0,
        maxx=100.0,
        maxy=100.0,
    )


def test_transport_graph_nodes_surface_crossing_without_joining_elevated_line():
    roads = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "road_class": ["local", "local", "secondary"],
            "estimated_width_m": [5.0, 5.0, 7.0],
            "vertical_mode": ["surface", "surface", "elevated"],
            "bridge": [None, None, "yes"],
            "geometry": [
                LineString([(10.0, 50.0), (90.0, 50.0)]),
                LineString([(50.0, 10.0), (50.0, 90.0)]),
                LineString([(30.0, 10.0), (30.0, 90.0)]),
            ],
        },
        crs="EPSG:3857",
    )
    rail = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=roads.crs)

    graph = build_transport_graph(roads, rail, _tile())
    surface_intersections = [
        node
        for node in graph["nodes"]
        if node["vertical_mode"] == "surface" and node["node_type"] == "intersection"
    ]
    elevated_nodes = [
        node for node in graph["nodes"] if node["vertical_mode"] == "elevated"
    ]

    assert len(surface_intersections) == 1
    assert surface_intersections[0]["position_projected_m"][:2] == [50.0, 50.0]
    assert len(elevated_nodes) == 2
    assert all(node["position_projected_m"][2] == 8.0 for node in elevated_nodes)
    assert not any(
        node["position_projected_m"][:2] == [30.0, 50.0] for node in elevated_nodes
    )


def test_underground_rail_and_surface_building_can_share_xy_space():
    rail = gpd.GeoDataFrame(
        {
            "id": [7],
            "railway": ["subway"],
            "vertical_mode": ["underground"],
            "tunnel": ["yes"],
            "geometry": [LineString([(10.0, 50.0), (90.0, 50.0)])],
        },
        crs="EPSG:3857",
    )
    roads = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=rail.crs)
    buildings = gpd.GeoDataFrame(
        {
            "id": [11],
            "building": ["apartments"],
            "estimated_height_m": [24.0],
            "height_source": ["height"],
            "height_confidence": [3],
            "geometry": [box(40.0, 40.0, 60.0, 60.0)],
        },
        crs=rail.crs,
    )

    graph = build_transport_graph(roads, rail, _tile())
    solids = building_solids(buildings, _tile())

    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["vertical_mode"] == "underground"
    assert {point[2] for point in graph["edges"][0]["geometry_local_m"]} == {-12.0}
    assert solids[0]["base_z_m"] == 0.0
    assert solids[0]["height_m"] == 24.0


def test_compile_city_state_bundle_indexes_multiple_cities(tmp_path):
    for city_id in ("alpha", "beta"):
        tile_dir = tmp_path / f"{city_id}-centre" / "tiles" / f"{city_id}-tile"
        tile_dir.mkdir(parents=True)
        state = {
            "tile": {"tile_id": f"{city_id}-tile", "city_id": city_id, "area_id": "centre"},
            "coordinate_system": {
                "origin_projected": [0.0, 0.0],
                "source_projected_crs": "EPSG:3857",
                "local_bounds": [0.0, 0.0, 100.0, 100.0],
            },
            "transport_graph": {"statistics": {"nodes": 2, "edges": 1}},
            "building_solids": [{"id": "building"}],
        }
        (tile_dir / "city.json").write_text(json.dumps(state), encoding="utf-8")
        (tile_dir / "metadata.json").write_text(
            json.dumps({"city_id": city_id, "area_id": "centre"}), encoding="utf-8"
        )

    result = compile_city_state_bundle(tmp_path, tmp_path / "city-state-index.json")

    assert result["summary"]["tiles"] == 2
    assert result["summary"]["cities"] == {"alpha": 1, "beta": 1}
    assert result["summary"]["transport_edges"] == 2
    assert result["summary"]["buildings"] == 2
