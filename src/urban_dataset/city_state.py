from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiLineString, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, transform, unary_union

from .classify import clean_tag, first_number
from .tile import TileSpec
from .vertical import VERTICAL_MODE_NAMES, vertical_mode_name

CITY_STATE_VERSION = "0.1.0"

# These are compilation defaults, not measured elevations. They keep the city state
# immediately usable by a 3D importer while preserving that the source data did not
# supply metric depth or deck height.
DEFAULT_VERTICAL_Z_M: dict[str, float | None] = {
    "surface": 0.0,
    "underground": -12.0,
    "elevated": 8.0,
    "unknown": None,
}

_RAIL_WIDTH_M = {
    "rail": 6.0,
    "subway": 6.0,
    "light_rail": 5.0,
    "tram": 4.0,
}


def _clean_properties(row: Any, geometry_column: str = "geometry") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key == geometry_column or value is None:
            continue
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        if missing:
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            continue
        if isinstance(value, (str, int, float, bool)):
            result[str(key)] = value
    return result


def _local_geometry(geometry: BaseGeometry, tile: TileSpec) -> BaseGeometry:
    return transform(lambda x, y, z=None: (x - tile.minx, y - tile.miny), geometry)


def _iter_lines(geometry: BaseGeometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
        return
    if isinstance(geometry, MultiLineString | GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_lines(part)


def _stable_id(prefix: str, *parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _vertical_mode(properties: dict[str, Any]) -> str:
    value = clean_tag(properties.get("vertical_mode"))
    if value in VERTICAL_MODE_NAMES:
        return value
    return vertical_mode_name(properties)


def _layer_order(properties: dict[str, Any]) -> float | None:
    return first_number(properties.get("layer"))


def _width(properties: dict[str, Any], transport_mode: str) -> tuple[float, str]:
    if transport_mode == "road":
        value = first_number(properties.get("estimated_width_m"))
        if value is not None and value > 0:
            return float(value), "estimated_width_m"
        return 5.0, "road_default"
    railway = clean_tag(properties.get("railway"))
    return float(_RAIL_WIDTH_M.get(railway, 5.0)), "rail_mode_default"


def _source_records(frame: gpd.GeoDataFrame, transport_mode: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        properties = _clean_properties(row)
        mode = _vertical_mode(properties)
        source_id = properties.get("id", index)
        for part_index, geometry in enumerate(_iter_lines(row.geometry)):
            records.append(
                {
                    "geometry": geometry,
                    "properties": properties,
                    "transport_mode": transport_mode,
                    "vertical_mode": mode,
                    "layer_order": _layer_order(properties),
                    "source_key": f"{source_id}:{part_index}",
                }
            )
    return records


def _group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    if record["vertical_mode"] == "unknown":
        # Unknown stacking must not be connected to another feature merely because
        # their two-dimensional lines cross.
        return (
            record["transport_mode"],
            record["vertical_mode"],
            record["source_key"],
        )
    layer = record["layer_order"]
    return (
        record["transport_mode"],
        record["vertical_mode"],
        0.0 if layer is None else round(float(layer), 6),
    )


def _noded_lines(records: list[dict[str, Any]]) -> list[LineString]:
    union = unary_union([record["geometry"] for record in records])
    if isinstance(union, LineString):
        return [union]
    merged = linemerge(union)
    return list(_iter_lines(merged))


def _nearest_source(segment: LineString, records: list[dict[str, Any]]) -> dict[str, Any]:
    midpoint = segment.interpolate(0.5, normalized=True)
    return min(records, key=lambda record: record["geometry"].distance(midpoint))


def _node_key(
    point: tuple[float, float],
    transport_mode: str,
    vertical_mode: str,
    layer_order: float | None,
) -> tuple[Any, ...]:
    return (
        transport_mode,
        vertical_mode,
        None if layer_order is None else round(float(layer_order), 6),
        round(float(point[0]), 3),
        round(float(point[1]), 3),
    )


def _boundary_node(x: float, y: float, tile: TileSpec, tolerance: float = 0.05) -> bool:
    return (
        abs(x - tile.minx) <= tolerance
        or abs(x - tile.maxx) <= tolerance
        or abs(y - tile.miny) <= tolerance
        or abs(y - tile.maxy) <= tolerance
    )


def build_transport_graph(
    roads: gpd.GeoDataFrame,
    rail: gpd.GeoDataFrame,
    tile: TileSpec,
) -> dict[str, Any]:
    records = [*_source_records(roads, "road"), *_source_records(rail, "rail")]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_group_key(record)].append(record)

    node_lookup: dict[tuple[Any, ...], str] = {}
    node_values: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for group_key in sorted(groups, key=str):
        group = groups[group_key]
        for segment in _noded_lines(group):
            if segment.length <= 1e-6:
                continue
            source = _nearest_source(segment, group)
            properties = source["properties"]
            transport_mode = source["transport_mode"]
            vertical_mode = source["vertical_mode"]
            layer_order = source["layer_order"]
            z = DEFAULT_VERTICAL_Z_M[vertical_mode]
            start = tuple(segment.coords[0][:2])
            end = tuple(segment.coords[-1][:2])

            node_ids: list[str] = []
            for point in (start, end):
                key = _node_key(point, transport_mode, vertical_mode, layer_order)
                node_id = node_lookup.get(key)
                if node_id is None:
                    node_id = _stable_id(tile.tile_id, "node", *key)
                    node_lookup[key] = node_id
                    local = [float(point[0] - tile.minx), float(point[1] - tile.miny)]
                    port_key = _stable_id("port", *key)
                    node_values[node_id] = {
                        "id": node_id,
                        "transport_mode": transport_mode,
                        "vertical_mode": vertical_mode,
                        "layer_order": layer_order,
                        "position_local_m": [*local, z],
                        "position_projected_m": [float(point[0]), float(point[1]), z],
                        "boundary_port_key": port_key if _boundary_node(*point, tile) else None,
                        "requires_vertical_review": vertical_mode == "unknown",
                    }
                node_ids.append(node_id)

            width_m, width_source = _width(properties, transport_mode)
            local_coordinates = [
                [float(x - tile.minx), float(y - tile.miny), z]
                for x, y, *_rest in segment.coords
            ]
            edge_id = _stable_id(
                tile.tile_id,
                "edge",
                node_ids[0],
                node_ids[1],
                source["source_key"],
                round(float(segment.length), 3),
            )
            edges.append(
                {
                    "id": edge_id,
                    "from_node": node_ids[0],
                    "to_node": node_ids[1],
                    "transport_mode": transport_mode,
                    "class": properties.get("road_class")
                    if transport_mode == "road"
                    else properties.get("railway"),
                    "vertical_mode": vertical_mode,
                    "layer_order": layer_order,
                    "width_m": width_m,
                    "width_source": width_source,
                    "length_m": float(segment.length),
                    "oneway": properties.get("oneway"),
                    "bridge": properties.get("bridge"),
                    "tunnel": properties.get("tunnel"),
                    "source_id": properties.get("id"),
                    "source_osm_type": properties.get("osm_type"),
                    "geometry_local_m": local_coordinates,
                    "z_source": "vertical_mode_default" if z is not None else "unknown",
                    "requires_vertical_review": vertical_mode == "unknown",
                }
            )

    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge["from_node"]] += 1
        degree[edge["to_node"]] += 1

    for node in node_values.values():
        count = degree[node["id"]]
        if node["boundary_port_key"] is not None:
            kind = "boundary_port"
        elif count <= 1:
            kind = "endpoint"
        elif count >= 3:
            kind = "intersection"
        else:
            kind = "continuation"
        node["degree"] = count
        node["node_type"] = kind

    nodes = sorted(node_values.values(), key=lambda value: value["id"])
    edges.sort(key=lambda value: value["id"])
    return {
        "nodes": nodes,
        "edges": edges,
        "statistics": {
            "nodes": len(nodes),
            "edges": len(edges),
            "boundary_ports": sum(node["node_type"] == "boundary_port" for node in nodes),
            "intersections": sum(node["node_type"] == "intersection" for node in nodes),
            "unknown_vertical_edges": sum(edge["vertical_mode"] == "unknown" for edge in edges),
        },
    }


def building_solids(buildings: gpd.GeoDataFrame, tile: TileSpec) -> list[dict[str, Any]]:
    solids: list[dict[str, Any]] = []
    for index, row in buildings.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        properties = _clean_properties(row)
        height = first_number(properties.get("estimated_height_m"))
        if height is None or height <= 0:
            height = 9.3
            height_source = "fallback"
        else:
            height_source = str(properties.get("height_source", "estimated_height_m"))
        source_id = properties.get("id", index)
        solids.append(
            {
                "id": _stable_id(tile.tile_id, "building", source_id),
                "source_id": properties.get("id"),
                "source_osm_type": properties.get("osm_type"),
                "building_type": properties.get("building"),
                "footprint_local_m": mapping(_local_geometry(geometry, tile)),
                "base_z_m": 0.0,
                "height_m": float(height),
                "height_source": height_source,
                "height_confidence": properties.get("height_confidence"),
            }
        )
    return solids


def city_state_header(
    tile: TileSpec,
    crs: str,
    *,
    area_id: str | None,
) -> dict[str, Any]:
    return {
        "format": "urban-city-state-tile",
        "version": CITY_STATE_VERSION,
        "tile": {
            "tile_id": tile.tile_id,
            "city_id": tile.city_id,
            "area_id": area_id,
        },
        "coordinate_system": {
            "units": "metres",
            "origin_projected": [tile.minx, tile.miny],
            "source_projected_crs": crs,
            "local_bounds": [0.0, 0.0, tile.maxx - tile.minx, tile.maxy - tile.miny],
            "axis_convention": "x-east, y-north, z-up",
        },
        "vertical_defaults": {
            "values_m": DEFAULT_VERTICAL_Z_M,
            "status": "procedural compilation defaults, not measured elevations",
        },
    }
