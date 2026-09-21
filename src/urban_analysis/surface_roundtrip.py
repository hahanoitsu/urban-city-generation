from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import LineString, shape
from shapely.ops import unary_union

from urban_model.surface_vectorize import surface_classes_to_city_state
from urban_model.vectorize import _connected_components, _require_image_tools


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def _vector_masks(
    state: dict[str, Any],
    shape_value: tuple[int, int],
) -> dict[str, np.ndarray]:
    bounds = state["coordinate_system"]["local_bounds"]
    transform = from_bounds(*bounds, width=shape_value[1], height=shape_value[0])

    masks = {
        "vegetation": np.zeros(shape_value, dtype=bool),
        "building": np.zeros(shape_value, dtype=bool),
        "road_major": np.zeros(shape_value, dtype=bool),
        "road_secondary": np.zeros(shape_value, dtype=bool),
        "road_local": np.zeros(shape_value, dtype=bool),
        "rail": np.zeros(shape_value, dtype=bool),
        "water": np.zeros(shape_value, dtype=bool),
    }

    def burn(geometries) -> np.ndarray:
        geometries = list(geometries)
        if not geometries:
            return np.zeros(shape_value, dtype=bool)
        return rasterize(
            [(geometry, 1) for geometry in geometries],
            out_shape=shape_value,
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)

    masks["vegetation"] = burn(
        shape(item["geometry"]) for item in state.get("green", [])
    )
    masks["water"] = burn(
        shape(item["geometry"]) for item in state.get("water", [])
    )
    masks["building"] = burn(
        shape(item["footprint_local_m"])
        for item in state.get("building_footprints", [])
    )

    grouped: dict[str, list] = {
        "road_major": [],
        "road_secondary": [],
        "road_local": [],
        "rail": [],
    }
    for edge in state.get("transport_graph", {}).get("edges", []):
        coords = edge.get("geometry_local_m", [])
        if len(coords) < 2:
            continue
        line = LineString([(float(x), float(y)) for x, y, *_ in coords])
        width = float(edge.get("width_m", 0.0))
        geometry = line.buffer(max(width / 2.0, 0.5), cap_style="flat")
        if edge.get("transport_mode") == "rail":
            grouped["rail"].append(geometry)
        else:
            key = f"road_{edge.get('class')}"
            if key in grouped:
                grouped[key].append(geometry)

    for name, geometries in grouped.items():
        masks[name] = burn(geometries)

    return masks


def _road_raster_stats(classes: np.ndarray) -> dict[str, float | int]:
    _closing, _distance, _label, _disk, skeletonize = _require_image_tools()
    skeleton = skeletonize((classes >= 3) & (classes <= 5))
    active = {tuple(value) for value in np.argwhere(skeleton)}
    components = _connected_components(active)
    sizes = [len(component) for component in components]
    total = sum(sizes)
    return {
        "raster_road_components": len(components),
        "raster_road_largest_fraction": max(sizes, default=0) / total if total else 0.0,
    }


def _road_graph_stats(state: dict[str, Any]) -> dict[str, float | int]:
    graph = nx.Graph()
    nodes = {
        str(node["id"]): node
        for node in state.get("transport_graph", {}).get("nodes", [])
        if node.get("transport_mode") == "road"
    }
    graph.add_nodes_from(nodes)

    for edge in state.get("transport_graph", {}).get("edges", []):
        if edge.get("transport_mode") != "road":
            continue
        graph.add_edge(
            str(edge["from_node"]),
            str(edge["to_node"]),
            length_m=float(edge.get("length_m", 0.0)),
        )

    graph.remove_nodes_from(list(nx.isolates(graph)))
    components = list(nx.connected_components(graph))
    lengths = []
    for component in components:
        lengths.append(
            sum(
                float(data.get("length_m", 0.0))
                for left, right, data in graph.edges(component, data=True)
                if left in component and right in component
            )
        )
    total = sum(lengths)
    return {
        "vector_road_components": len(components),
        "vector_road_largest_fraction": max(lengths, default=0.0) / total if total else 0.0,
    }


def _building_access(state: dict[str, Any]) -> dict[str, float]:
    buildings = [
        shape(item["footprint_local_m"])
        for item in state.get("building_footprints", [])
    ]
    roads = []
    for edge in state.get("transport_graph", {}).get("edges", []):
        if edge.get("transport_mode") != "road":
            continue
        coords = edge.get("geometry_local_m", [])
        if len(coords) >= 2:
            roads.append(LineString([(float(x), float(y)) for x, y, *_ in coords]))

    if not buildings or not roads:
        return {
            "buildings_within_20m_road_fraction": 0.0,
            "buildings_within_40m_road_fraction": 0.0,
            "building_road_distance_median_m": float("nan"),
        }

    road_union = unary_union(roads)
    distances = np.asarray([polygon.distance(road_union) for polygon in buildings])
    return {
        "buildings_within_20m_road_fraction": float((distances <= 20.0).mean()),
        "buildings_within_40m_road_fraction": float((distances <= 40.0).mean()),
        "building_road_distance_median_m": float(np.median(distances)),
    }


def audit_classes(
    classes: np.ndarray,
    *,
    seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    source = np.asarray(classes, dtype=np.uint8)
    state = surface_classes_to_city_state(source, seed=seed)
    vector = _vector_masks(state, source.shape)

    source_masks = {
        "vegetation": source == 1,
        "building": source == 2,
        "road_major": source == 3,
        "road_secondary": source == 4,
        "road_local": source == 5,
        "rail": source == 6,
        "water": source == 7,
    }

    metrics = {
        f"{name}_iou": _iou(source_masks[name], vector[name])
        for name in source_masks
    }
    metrics.update(_road_raster_stats(source))
    metrics.update(_road_graph_stats(state))
    metrics.update(_building_access(state))
    metrics["road_component_inflation"] = (
        metrics["vector_road_components"]
        / max(metrics["raster_road_components"], 1)
    )
    metrics["road_largest_fraction_retention"] = (
        metrics["vector_road_largest_fraction"]
        / max(metrics["raster_road_largest_fraction"], 1e-9)
    )

    return state, metrics, vector


def write_state(path: str | Path, state: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, indent=2) + "\n")
