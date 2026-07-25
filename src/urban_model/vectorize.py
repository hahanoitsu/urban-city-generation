from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Iterable

import numpy as np
import torch
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import LineString, shape

from .data import model_space_to_layers

ROAD_WIDTHS_M = {"major": 18.0, "secondary": 12.0, "local": 7.0}


def _require_image_tools():
    try:
        from scipy.ndimage import binary_closing, distance_transform_edt, label
        from skimage.morphology import disk, skeletonize
    except ImportError as exc:
        raise RuntimeError(
            "Install the project with the 'diffusion' extra to vectorise generated layers"
        ) from exc
    return binary_closing, distance_transform_edt, label, disk, skeletonize


def _neighbours(pixel: tuple[int, int], active: set[tuple[int, int]]) -> list[tuple[int, int]]:
    row, column = pixel
    result = []
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            if row_offset == 0 and column_offset == 0:
                continue
            candidate = (row + row_offset, column + column_offset)
            if candidate in active:
                result.append(candidate)
    return sorted(result)


def _connected_components(active: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(active)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        component = {start}
        remaining.remove(start)
        while queue:
            current = queue.popleft()
            for neighbour in _neighbours(current, active):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def _skeleton_paths(mask: np.ndarray, minimum_pixels: int) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    binary_closing, _distance_transform_edt, _label, disk, skeletonize = _require_image_tools()
    cleaned = binary_closing(mask.astype(bool), structure=disk(1))
    skeleton = skeletonize(cleaned)
    active = {tuple(value) for value in np.argwhere(skeleton)}
    paths: list[list[tuple[int, int]]] = []

    for component in _connected_components(active):
        if len(component) < minimum_pixels:
            continue
        degree = {pixel: len(_neighbours(pixel, component)) for pixel in component}
        nodes = {pixel for pixel, value in degree.items() if value != 2}
        if not nodes:
            start = min(component)
            neighbours = _neighbours(start, component)
            if not neighbours:
                continue
            path = [start]
            previous = start
            current = neighbours[0]
            while current != start and len(path) <= len(component) + 1:
                path.append(current)
                candidates = [value for value in _neighbours(current, component) if value != previous]
                if not candidates:
                    break
                previous, current = current, candidates[0]
            path.append(start)
            if len(path) >= minimum_pixels:
                paths.append(path)
            continue

        visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for start in sorted(nodes):
            for first in _neighbours(start, component):
                key = tuple(sorted((start, first)))
                if key in visited:
                    continue
                path = [start]
                previous = start
                current = first
                visited.add(key)
                while True:
                    path.append(current)
                    if current in nodes and current != start:
                        break
                    candidates = [
                        value
                        for value in _neighbours(current, component)
                        if value != previous
                        and tuple(sorted((current, value))) not in visited
                    ]
                    if not candidates:
                        break
                    next_pixel = candidates[0]
                    visited.add(tuple(sorted((current, next_pixel))))
                    previous, current = current, next_pixel
                if len(path) >= minimum_pixels:
                    paths.append(path)
    return paths, cleaned.astype(bool)


def _pixel_xy(
    pixel: tuple[int, int],
    *,
    shape_value: tuple[int, int],
    bounds: list[float],
) -> tuple[float, float]:
    row, column = pixel
    height, width = shape_value
    dx = (bounds[2] - bounds[0]) / width
    dy = (bounds[3] - bounds[1]) / height
    return (
        bounds[0] + (column + 0.5) * dx,
        bounds[3] - (row + 0.5) * dy,
    )


def _road_class_from_width(width_m: float) -> str:
    if width_m >= 14.0:
        return "major"
    if width_m >= 9.0:
        return "secondary"
    return "local"


def _road_class_for_path(path: list[tuple[int, int]], class_map: np.ndarray) -> str | None:
    values = [int(class_map[pixel]) for pixel in path if int(class_map[pixel]) > 0]
    if not values:
        return None
    value = Counter(values).most_common(1)[0][0]
    return {1: "major", 2: "secondary", 3: "local"}[value]


def _smooth_grade(z: np.ndarray, xy: np.ndarray, max_grade: float) -> np.ndarray:
    if len(z) <= 2:
        return z
    kernel = np.ones(5, dtype=np.float64) / 5.0
    padded = np.pad(z.astype(np.float64), (2, 2), mode="edge")
    result = np.convolve(padded, kernel, mode="valid")
    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    for index in range(1, len(result)):
        limit = max_grade * max(float(distances[index - 1]), 1e-6)
        result[index] = np.clip(result[index], result[index - 1] - limit, result[index - 1] + limit)
    for index in range(len(result) - 2, -1, -1):
        limit = max_grade * max(float(distances[index]), 1e-6)
        result[index] = np.clip(result[index], result[index + 1] - limit, result[index + 1] + limit)
    return result.astype(np.float32)


def _coordinates_3d(
    path: list[tuple[int, int]],
    *,
    z_field: np.ndarray,
    shape_value: tuple[int, int],
    bounds: list[float],
    max_grade: float,
) -> list[list[float]]:
    xy = np.asarray(
        [_pixel_xy(pixel, shape_value=shape_value, bounds=bounds) for pixel in path],
        dtype=np.float64,
    )
    z = np.asarray([float(z_field[pixel]) for pixel in path], dtype=np.float32)
    z = _smooth_grade(z, xy, max_grade)
    if len(path) <= 2:
        return [[float(x), float(y), float(height)] for (x, y), height in zip(xy, z, strict=True)]

    metres_per_pixel = max(
        (bounds[2] - bounds[0]) / shape_value[1],
        (bounds[3] - bounds[1]) / shape_value[0],
    )
    simplified = list(LineString(xy).simplify(metres_per_pixel * 0.65, preserve_topology=False).coords)
    if len(simplified) < 2:
        simplified = [tuple(xy[0]), tuple(xy[-1])]

    result: list[list[float]] = []
    lower = 0
    for x, y in simplified:
        candidate = xy[lower:]
        distances = np.square(candidate[:, 0] - x) + np.square(candidate[:, 1] - y)
        relative = int(np.argmin(distances))
        index = lower + relative
        result.append([float(x), float(y), float(z[index])])
        lower = min(index, len(xy) - 1)
    return result


def _append_network(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_lookup: dict[tuple[str, str, tuple[int, int]], str],
    mask: np.ndarray,
    class_map: np.ndarray | None,
    z_field: np.ndarray,
    transport_mode: str,
    vertical_mode: str,
    bounds: list[float],
    minimum_pixels: int,
    max_grade: float,
) -> None:
    _binary_closing, distance_transform_edt, _label, _disk, _skeletonize = _require_image_tools()
    paths, cleaned = _skeleton_paths(mask, minimum_pixels)
    distances = distance_transform_edt(cleaned)
    shape_value = mask.shape
    metres_per_pixel = (
        (bounds[2] - bounds[0]) / shape_value[1]
        + (bounds[3] - bounds[1]) / shape_value[0]
    ) / 2.0

    def node_id(pixel: tuple[int, int], position: list[float]) -> str:
        key = (transport_mode, vertical_mode, pixel)
        existing = node_lookup.get(key)
        if existing is not None:
            return existing
        identifier = f"node_{len(nodes):05d}"
        node_lookup[key] = identifier
        nodes.append(
            {
                "id": identifier,
                "transport_mode": transport_mode,
                "vertical_mode": vertical_mode,
                "position_local_m": position,
            }
        )
        return identifier

    for path in paths:
        coordinates = _coordinates_3d(
            path,
            z_field=z_field,
            shape_value=shape_value,
            bounds=bounds,
            max_grade=max_grade,
        )
        if len(coordinates) < 2:
            continue
        line = LineString([(value[0], value[1]) for value in coordinates])
        if line.length <= 1e-6:
            continue
        sampled_width = float(
            np.median([max(1.0, distances[pixel] * 2.0 * metres_per_pixel) for pixel in path])
        )
        if transport_mode == "road":
            road_class = (
                _road_class_for_path(path, class_map)
                if class_map is not None
                else _road_class_from_width(sampled_width)
            )
            if road_class is None:
                road_class = _road_class_from_width(sampled_width)
            width_m = (
                ROAD_WIDTHS_M[road_class]
                if class_map is not None
                else float(np.clip(sampled_width, 4.0, 30.0))
            )
            edge_class = road_class
        else:
            width_m = float(np.clip(sampled_width, 4.0, 10.0))
            edge_class = "rail"

        left = node_id(path[0], coordinates[0])
        right = node_id(path[-1], coordinates[-1])
        z_values = [value[2] for value in coordinates]
        grades = []
        for first, second in zip(coordinates[:-1], coordinates[1:], strict=True):
            horizontal = float(np.hypot(second[0] - first[0], second[1] - first[1]))
            if horizontal > 1e-6:
                grades.append(abs(second[2] - first[2]) / horizontal)
        edges.append(
            {
                "id": f"edge_{len(edges):05d}",
                "from_node": left,
                "to_node": right,
                "transport_mode": transport_mode,
                "class": edge_class,
                "vertical_mode": vertical_mode,
                "width_m": width_m,
                "length_m": float(line.length),
                "minimum_z_m": float(min(z_values)),
                "maximum_z_m": float(max(z_values)),
                "maximum_grade": float(max(grades, default=0.0)),
                "z_source": "generated_profile",
                "geometry_local_m": coordinates,
            }
        )


def _merge_transition_nodes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    tolerance_m: float,
) -> list[dict[str, Any]]:
    replacements: dict[str, str] = {}
    for index, first in enumerate(nodes):
        if first["id"] in replacements:
            continue
        first_position = np.asarray(first["position_local_m"], dtype=float)
        for second in nodes[index + 1 :]:
            if second["id"] in replacements:
                continue
            if first["transport_mode"] != second["transport_mode"]:
                continue
            if first["vertical_mode"] == second["vertical_mode"]:
                continue
            second_position = np.asarray(second["position_local_m"], dtype=float)
            if np.linalg.norm(first_position[:2] - second_position[:2]) > tolerance_m:
                continue
            if abs(float(first_position[2] - second_position[2])) > 2.5:
                continue
            canonical, duplicate = (
                (first, second)
                if first["vertical_mode"] == "surface"
                else (second, first)
                if second["vertical_mode"] == "surface"
                else (first, second)
            )
            replacements[duplicate["id"]] = canonical["id"]
            canonical["vertical_mode"] = "transition"

    if not replacements:
        return nodes
    lookup = {node["id"]: node for node in nodes}
    for edge in edges:
        for key, coordinate_index in (("from_node", 0), ("to_node", -1)):
            node_id = edge[key]
            while node_id in replacements:
                node_id = replacements[node_id]
            edge[key] = node_id
            edge["geometry_local_m"][coordinate_index] = lookup[node_id]["position_local_m"]
    return [node for node in nodes if node["id"] not in replacements]


def _polygon_features(
    mask: np.ndarray,
    *,
    bounds: list[float],
    minimum_area_m2: float,
) -> list[dict[str, Any]]:
    transform = from_bounds(*bounds, width=mask.shape[1], height=mask.shape[0])
    result: list[dict[str, Any]] = []
    for geometry, value in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform):
        if int(value) != 1:
            continue
        polygon = shape(geometry)
        if polygon.is_empty or polygon.area < minimum_area_m2:
            continue
        result.append({"geometry": geometry, "area_m2": float(polygon.area)})
    return result


def _building_solids(
    building_mask: np.ndarray,
    height_values: np.ndarray,
    *,
    bounds: list[float],
    max_height_m: float,
) -> list[dict[str, Any]]:
    _binary_closing, _distance_transform_edt, label, _disk, _skeletonize = _require_image_tools()
    labels, count = label(building_mask.astype(bool))
    transform = from_bounds(*bounds, width=building_mask.shape[1], height=building_mask.shape[0])
    solids: list[dict[str, Any]] = []
    for geometry, value in shapes(labels.astype(np.int32), mask=labels > 0, transform=transform):
        component = int(value)
        if component <= 0 or component > count:
            continue
        polygon = shape(geometry)
        if polygon.is_empty or polygon.area < 20.0:
            continue
        pixels = labels == component
        normalized = float(height_values[pixels].mean()) if pixels.any() else 0.05
        height_m = float(np.clip(normalized * max_height_m, 3.2, max_height_m))
        solids.append(
            {
                "id": f"building_{len(solids):05d}",
                "footprint_local_m": geometry,
                "base_z_m": 0.0,
                "height_m": height_m,
                "height_source": "generated_layer",
            }
        )
    return solids


def generated_layers_to_city_state(
    values: torch.Tensor,
    *,
    bounds_m: Iterable[float] = (0.0, 0.0, 1024.0, 1024.0),
    max_height_m: float = 180.0,
    max_surface_offset_m: float = 12.0,
    max_underground_depth_m: float = 40.0,
    max_elevated_height_m: float = 30.0,
    road_max_grade: float = 0.08,
    rail_max_grade: float = 0.035,
    auxiliary_threshold: float = 0.35,
    minimum_component_pixels: int = 4,
    seed: int | None = None,
) -> dict[str, Any]:
    decoded = model_space_to_layers(
        values,
        auxiliary_threshold=auxiliary_threshold,
        max_surface_offset_m=max_surface_offset_m,
        max_underground_depth_m=max_underground_depth_m,
        max_elevated_height_m=max_elevated_height_m,
    )
    surface = decoded["surface"].detach().cpu().numpy().astype(np.int16)
    bounds = [float(value) for value in bounds_m]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_lookup: dict[tuple[str, str, tuple[int, int]], str] = {}

    surface_class_map = np.zeros_like(surface, dtype=np.uint8)
    surface_class_map[surface == 3] = 1
    surface_class_map[surface == 4] = 2
    surface_class_map[surface == 5] = 3

    networks = [
        (
            surface_class_map > 0,
            surface_class_map,
            decoded["road_surface_offset_m"],
            "road",
            "surface",
            road_max_grade,
        ),
        (
            surface == 6,
            None,
            decoded["rail_surface_offset_m"],
            "rail",
            "surface",
            rail_max_grade,
        ),
        (
            decoded["road_underground"],
            None,
            -decoded["road_underground_depth_m"],
            "road",
            "underground",
            road_max_grade,
        ),
        (
            decoded["road_elevated"],
            None,
            decoded["road_elevated_height_m"],
            "road",
            "elevated",
            road_max_grade,
        ),
        (
            decoded["rail_underground"],
            None,
            -decoded["rail_underground_depth_m"],
            "rail",
            "underground",
            rail_max_grade,
        ),
        (
            decoded["rail_elevated"],
            None,
            decoded["rail_elevated_height_m"],
            "rail",
            "elevated",
            rail_max_grade,
        ),
    ]
    for mask, class_map, z_field, transport_mode, vertical_mode, max_grade in networks:
        _append_network(
            nodes=nodes,
            edges=edges,
            node_lookup=node_lookup,
            mask=mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask,
            class_map=class_map,
            z_field=z_field.detach().cpu().numpy(),
            transport_mode=transport_mode,
            vertical_mode=vertical_mode,
            bounds=bounds,
            minimum_pixels=minimum_component_pixels,
            max_grade=max_grade,
        )

    metres_per_pixel = max(
        (bounds[2] - bounds[0]) / surface.shape[1],
        (bounds[3] - bounds[1]) / surface.shape[0],
    )
    nodes = _merge_transition_nodes(nodes, edges, tolerance_m=metres_per_pixel * 1.5)

    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge["from_node"]] += 1
        degree[edge["to_node"]] += 1
    for node in nodes:
        node["degree"] = degree[node["id"]]
        node["node_type"] = (
            "endpoint"
            if node["degree"] <= 1
            else "intersection"
            if node["degree"] >= 3
            else "continuation"
        )

    height_values = decoded["building_height"].detach().cpu().numpy()
    buildings = _building_solids(
        surface == 2,
        height_values,
        bounds=bounds,
        max_height_m=max_height_m,
    )
    water = _polygon_features(surface == 7, bounds=bounds, minimum_area_m2=20.0)
    green = _polygon_features(surface == 1, bounds=bounds, minimum_area_m2=20.0)

    return {
        "format": "urban-city-state-tile",
        "version": "0.3.0",
        "tile": {
            "city_id": "generated",
            "area_id": "generated",
            "tile_id": f"generated_{seed}",
        },
        "coordinate_system": {
            "units": "metres",
            "axis_convention": "x-east, y-north, z-up",
            "local_bounds": bounds,
        },
        "generation": {
            "kind": "multilayer_diffusion",
            "seed": seed,
            "vertical_profiles": "generated_continuous_z",
        },
        "transport_graph": {
            "nodes": nodes,
            "edges": edges,
            "statistics": {"nodes": len(nodes), "edges": len(edges)},
        },
        "building_solids": buildings,
        "water": water,
        "green": green,
        "statistics": {
            "nodes": len(nodes),
            "edges": len(edges),
            "buildings": len(buildings),
            "surface_edges": sum(edge["vertical_mode"] == "surface" for edge in edges),
            "underground_edges": sum(
                edge["vertical_mode"] == "underground" for edge in edges
            ),
            "elevated_edges": sum(edge["vertical_mode"] == "elevated" for edge in edges),
            "transition_nodes": sum(node["vertical_mode"] == "transition" for node in nodes),
        },
    }
