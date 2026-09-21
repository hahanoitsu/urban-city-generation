from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from shapely.geometry import LineString

from .vectorize import (
    ROAD_WIDTHS_M,
    _pixel_xy,
    _polygon_features,
    _road_class_for_path,
    _skeleton_paths,
)

SURFACE_NAMES = (
    "terrain",
    "vegetation",
    "building",
    "road_major",
    "road_secondary",
    "road_local",
    "rail",
    "water",
)


def _append_surface_network(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_lookup: dict[tuple[str, tuple[int, int]], str],
    mask: np.ndarray,
    bounds: list[float],
    transport_mode: str,
    class_map: np.ndarray | None = None,
    minimum_pixels: int = 4,
) -> None:
    paths, _cleaned = _skeleton_paths(mask, minimum_pixels)
    shape_value = mask.shape

    def node_id(pixel: tuple[int, int]) -> str:
        key = (transport_mode, pixel)
        if key in node_lookup:
            return node_lookup[key]
        x, y = _pixel_xy(pixel, shape_value=shape_value, bounds=bounds)
        identifier = f"node_{len(nodes):05d}"
        node_lookup[key] = identifier
        nodes.append(
            {
                "id": identifier,
                "transport_mode": transport_mode,
                "vertical_mode": "surface",
                "position_local_m": [x, y, 0.0],
            }
        )
        return identifier

    for path in paths:
        coords = [
            [*_pixel_xy(pixel, shape_value=shape_value, bounds=bounds), 0.0]
            for pixel in path
        ]
        if len(coords) < 2:
            continue

        line = LineString([(x, y) for x, y, _z in coords])
        if line.length <= 0:
            continue

        if transport_mode == "road":
            road_class = _road_class_for_path(path, class_map)
            if road_class is None:
                continue
            width = ROAD_WIDTHS_M[road_class]
            edge_class = road_class
        else:
            width = 6.0
            edge_class = "rail"

        edges.append(
            {
                "id": f"edge_{len(edges):05d}",
                "from_node": node_id(path[0]),
                "to_node": node_id(path[-1]),
                "transport_mode": transport_mode,
                "class": edge_class,
                "vertical_mode": "surface",
                "width_m": width,
                "length_m": float(line.length),
                "geometry_local_m": coords,
            }
        )


def surface_classes_to_city_state(
    classes: np.ndarray,
    *,
    bounds_m: Iterable[float] = (0.0, 0.0, 1000.0, 1000.0),
    minimum_component_pixels: int = 4,
    seed: int | None = None,
) -> dict[str, Any]:
    surface = np.asarray(classes, dtype=np.uint8)
    if surface.ndim != 2:
        raise ValueError("Expected a 2D surface class map")

    bounds = [float(value) for value in bounds_m]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_lookup: dict[tuple[str, tuple[int, int]], str] = {}

    road_class_map = np.zeros_like(surface, dtype=np.uint8)
    road_class_map[surface == 3] = 1
    road_class_map[surface == 4] = 2
    road_class_map[surface == 5] = 3

    _append_surface_network(
        nodes=nodes,
        edges=edges,
        node_lookup=node_lookup,
        mask=road_class_map > 0,
        class_map=road_class_map,
        bounds=bounds,
        transport_mode="road",
        minimum_pixels=minimum_component_pixels,
    )
    _append_surface_network(
        nodes=nodes,
        edges=edges,
        node_lookup=node_lookup,
        mask=surface == 6,
        bounds=bounds,
        transport_mode="rail",
        minimum_pixels=minimum_component_pixels,
    )

    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge["from_node"]] += 1
        degree[edge["to_node"]] += 1
    for node in nodes:
        node["degree"] = degree[node["id"]]

    buildings = []
    for item in _polygon_features(surface == 2, bounds=bounds, minimum_area_m2=20.0):
        buildings.append(
            {
                "id": f"building_{len(buildings):05d}",
                "footprint_local_m": item["geometry"],
                "area_m2": item["area_m2"],
            }
        )

    water = _polygon_features(surface == 7, bounds=bounds, minimum_area_m2=20.0)
    green = _polygon_features(surface == 1, bounds=bounds, minimum_area_m2=20.0)

    return {
        "format": "urban-city-state-surface",
        "version": "0.1.0",
        "tile": {"tile_id": f"generated_{seed}"},
        "coordinate_system": {
            "units": "metres",
            "axis_convention": "x-east, y-north, z-up",
            "local_bounds": bounds,
        },
        "generation": {
            "kind": "surface_diffusion",
            "seed": seed,
        },
        "transport_graph": {
            "nodes": nodes,
            "edges": edges,
        },
        "building_footprints": buildings,
        "water": water,
        "green": green,
        "statistics": {
            "nodes": len(nodes),
            "edges": len(edges),
            "buildings": len(buildings),
            "water_polygons": len(water),
            "green_polygons": len(green),
        },
    }
