from __future__ import annotations

import gzip
import json

import geopandas as gpd
from shapely.geometry import LineString, box

from urban_dataset.context_graph import ContextGraphConfig, build_context_graph
from urban_dataset.extract import CityLayers
from urban_dataset.prepared import save_city_gpkg


def _empty(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def test_context_graph_keeps_cross_region_transport(tmp_path):
    crs = "EPSG:3857"
    roads = gpd.GeoDataFrame(
        {
            "id": ["road-main"],
            "road_class": ["major"],
            "estimated_width_m": [8.0],
            "vertical_mode": ["surface"],
            "geometry": [LineString([(0.0, 1000.0), (4096.0, 1000.0)])],
        },
        geometry="geometry",
        crs=crs,
    )
    rail = gpd.GeoDataFrame(
        {
            "id": ["rail-main"],
            "railway": ["rail"],
            "vertical_mode": ["surface"],
            "geometry": [LineString([(2200.0, 0.0), (2200.0, 4096.0)])],
        },
        geometry="geometry",
        crs=crs,
    )
    buildings = gpd.GeoDataFrame(
        {
            "id": ["building-a", "building-b"],
            "estimated_height_m": [18.0, 30.0],
            "height_confidence": [3, 2],
            "height_source": ["height", "building:levels"],
            "geometry": [
                box(300.0, 300.0, 420.0, 420.0),
                box(2500.0, 2500.0, 2600.0, 2650.0),
            ],
        },
        geometry="geometry",
        crs=crs,
    )
    water = gpd.GeoDataFrame(
        {"geometry": [box(0.0, 3900.0, 4096.0, 4096.0)]},
        geometry="geometry",
        crs=crs,
    )
    green = gpd.GeoDataFrame(
        {"geometry": [box(3800.0, 0.0, 4096.0, 4096.0)]},
        geometry="geometry",
        crs=crs,
    )

    layers = CityLayers(
        roads=roads,
        buildings=buildings,
        landuse=_empty(crs),
        landuse_known=_empty(crs),
        water=water,
        green=green,
        rail=rail,
    )

    city = tmp_path / "test-city.gpkg"
    save_city_gpkg(
        layers,
        city,
        {"city_id": "test-city"},
        overwrite=True,
    )

    output = tmp_path / "context"
    summary = build_context_graph(
        city,
        output,
        config=ContextGraphConfig(
            region_size_m=2048.0,
            target_size_m=512.0,
            target_stride_m=512.0,
            minimum_transport_length_m=10.0,
        ),
    )

    assert summary["region_nodes"] == 4
    assert summary["region_edges_with_roads"] > 0
    assert summary["region_edges_with_rail"] > 0
    assert summary["targets"] > 0
    assert summary["targets_with_boundary_ports"] > 0

    graph = json.loads((output / "context-graph.json").read_text())
    road_edges = [edge for edge in graph["edges"] if edge["road_port_count"]]
    assert any(
        any(port["source_id"] == "road-main" for port in edge["transport_ports"])
        for edge in road_edges
    )

    rows = [
        json.loads(line)
        for line in (output / "targets.jsonl").read_text().splitlines()
        if line.strip()
    ]
    row = max(rows, key=lambda value: value["boundary_ports"])
    with gzip.open(output / row["sample_path"], "rt", encoding="utf-8") as handle:
        sample = json.load(handle)

    assert sample["parent_region_id"] in sample["input"]["masked_region_ids"]
    assert sample["parent_region_id"] not in sample["input"]["visible_region_ids"]
    assert "visible_context_features" in sample["input"]
    assert "context_features" not in sample["input"]
    assert sample["target"]["z_supervision"]["metric_transport_z"] is False
    assert sample["input"]["boundary_ports"]


def test_target_geometry_stays_vector(tmp_path):
    crs = "EPSG:3857"
    roads = gpd.GeoDataFrame(
        {
            "id": ["curved-road"],
            "road_class": ["local"],
            "estimated_width_m": [5.0],
            "vertical_mode": ["surface"],
            "geometry": [
                LineString(
                    [
                        (0.0, 100.0),
                        (180.0, 130.0),
                        (310.0, 250.0),
                        (520.0, 300.0),
                    ]
                )
            ],
        },
        geometry="geometry",
        crs=crs,
    )
    layers = CityLayers(
        roads=roads,
        buildings=_empty(crs),
        landuse=_empty(crs),
        landuse_known=_empty(crs),
        water=_empty(crs),
        green=_empty(crs),
        rail=_empty(crs),
    )
    city = tmp_path / "curve.gpkg"
    save_city_gpkg(layers, city, {"city_id": "curve"}, overwrite=True)

    output = tmp_path / "context"
    build_context_graph(
        city,
        output,
        config=ContextGraphConfig(
            region_size_m=2048.0,
            target_size_m=512.0,
            target_stride_m=512.0,
            minimum_transport_length_m=10.0,
        ),
    )

    row = json.loads((output / "targets.jsonl").read_text().splitlines()[0])
    with gzip.open(output / row["sample_path"], "rt", encoding="utf-8") as handle:
        sample = json.load(handle)

    road = sample["target"]["roads"][0]
    assert road["id"] == "curved-road"
    assert len(road["geometry_local_m"]) >= 3
