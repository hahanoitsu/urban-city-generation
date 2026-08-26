from __future__ import annotations

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

from urban_analysis.passenger_connectivity import (
    _add_station_transfers,
    _rail_station_kind,
    attach_station_groups,
    group_station_features,
)


def test_rail_station_tag_classification():
    assert _rail_station_kind({"railway": "station"}) == "station"
    assert _rail_station_kind({"railway": "halt"}) == "station"
    assert _rail_station_kind({"railway": "stop"}) == "stop_position"
    assert (
        _rail_station_kind(
            {
                "public_transport": "station",
                "station": "subway",
            }
        )
        == "station"
    )
    assert (
        _rail_station_kind(
            {
                "public_transport": "stop_position",
                "train": "yes",
            }
        )
        == "stop_position"
    )
    assert _rail_station_kind({"public_transport": "station"}) is None


def test_station_grouping_uses_name_and_distance():
    frame = gpd.GeoDataFrame(
        {
            "source_index": ["1", "2", "3"],
            "kind": ["station", "stop_position", "station"],
            "name": ["Central", "Central", "Central"],
            "name_key": ["central", "central", "central"],
            "geometry": [Point(0, 0), Point(20, 0), Point(1000, 0)],
        },
        geometry="geometry",
        crs="EPSG:3414",
    )

    groups = group_station_features(frame, named_distance_m=300)

    assert len(groups) == 2
    assert sorted(len(group["members"]) for group in groups) == [1, 2]


def test_station_transfer_connects_separate_track_components():
    graph = nx.Graph()
    graph.add_node(
        "a0",
        tile_id="tile-a",
        position_projected_m=[0.0, 0.0, 0.0],
    )
    graph.add_node(
        "a1",
        tile_id="tile-a",
        position_projected_m=[100.0, 0.0, 0.0],
    )
    graph.add_node(
        "b0",
        tile_id="tile-a",
        position_projected_m=[0.0, 20.0, -12.0],
    )
    graph.add_node(
        "b1",
        tile_id="tile-a",
        position_projected_m=[100.0, 20.0, -12.0],
    )
    graph.add_edge("a0", "a1", kind="transport", length_m=100.0)
    graph.add_edge("b0", "b1", kind="transport", length_m=100.0)

    edge_records = [
        {
            "tile_id": "tile-a",
            "left": "a0",
            "right": "a1",
            "component": 0,
            "vertical_mode": "surface",
            "length_m": 100.0,
            "geometry": LineString([(0, 0), (100, 0)]),
        },
        {
            "tile_id": "tile-a",
            "left": "b0",
            "right": "b1",
            "component": 1,
            "vertical_mode": "underground",
            "length_m": 100.0,
            "geometry": LineString([(0, 20), (100, 20)]),
        },
    ]
    groups = [
        {
            "group_id": "station_0000",
            "name": "Interchange",
            "name_key": "interchange",
            "geometry": Point(50, 10),
            "members": [
                {
                    "source_index": "station",
                    "kind": "station",
                    "name": "Interchange",
                    "point": Point(50, 10),
                }
            ],
        }
    ]

    attachments = attach_station_groups(groups, edge_records, graph, radius_m=15.0)
    assert len(attachments[0]["attachments"]) == 2
    assert {value["component"] for value in attachments[0]["attachments"]} == {0, 1}

    passenger, transfers = _add_station_transfers(
        graph,
        attachments,
        {"tile-a": 1},
    )

    assert nx.number_connected_components(graph) == 2
    assert transfers == 1
    assert nx.number_connected_components(passenger) == 1


def test_station_outside_radius_does_not_create_transfer():
    graph = nx.Graph()
    graph.add_node("a", tile_id="tile", position_projected_m=[0.0, 0.0, 0.0])
    graph.add_node("b", tile_id="tile", position_projected_m=[100.0, 0.0, 0.0])
    graph.add_edge("a", "b", kind="transport", length_m=100.0)

    edges = [
        {
            "tile_id": "tile",
            "left": "a",
            "right": "b",
            "component": 0,
            "vertical_mode": "surface",
            "length_m": 100.0,
            "geometry": LineString([(0, 0), (100, 0)]),
        }
    ]
    groups = [
        {
            "group_id": "station_0000",
            "name": "Far",
            "name_key": "far",
            "geometry": Point(50, 100),
            "members": [
                {
                    "source_index": "station",
                    "kind": "station",
                    "name": "Far",
                    "point": Point(50, 100),
                }
            ],
        }
    ]

    result = attach_station_groups(groups, edges, graph, radius_m=20.0)
    assert result[0]["attachments"] == []
