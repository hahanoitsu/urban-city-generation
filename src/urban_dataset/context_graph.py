from __future__ import annotations

import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    box,
    mapping,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .extract import CityLayers
from .prepared import load_city_gpkg


@dataclass(frozen=True)
class ContextGraphConfig:
    region_size_m: float = 2048.0
    target_size_m: float = 512.0
    target_stride_m: float = 512.0
    minimum_transport_length_m: float = 40.0
    port_probe_m: float = 8.0
    include_diagonal_region_edges: bool = True


def _iter_lines(geometry: BaseGeometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
        return
    if isinstance(geometry, MultiLineString | GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_lines(part)


def _iter_points(geometry: BaseGeometry) -> Iterable[Point]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Point):
        yield geometry
        return
    if isinstance(geometry, MultiPoint | GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_points(part)
        return
    if isinstance(geometry, LineString):
        if len(geometry.coords) >= 2:
            yield Point(geometry.coords[0])
            yield Point(geometry.coords[-1])


def _frame_bounds(layers: CityLayers) -> tuple[float, float, float, float]:
    bounds = []
    for _name, frame in layers.items():
        if frame.empty:
            continue
        values = frame.total_bounds
        if len(values) == 4 and np.isfinite(values).all():
            bounds.append(values)
    if not bounds:
        raise ValueError("Prepared city contains no usable geometry")
    array = np.asarray(bounds, dtype=np.float64)
    return (
        float(array[:, 0].min()),
        float(array[:, 1].min()),
        float(array[:, 2].max()),
        float(array[:, 3].max()),
    )


def _clip_frame(frame: gpd.GeoDataFrame, geometry: BaseGeometry) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    indexes = list(frame.sindex.query(geometry, predicate="intersects"))
    if not indexes:
        return frame.iloc[0:0].copy()
    result = frame.iloc[indexes].copy()
    result["geometry"] = result.geometry.intersection(geometry)
    return result[result.geometry.notna() & ~result.geometry.is_empty].copy()


def _text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    try:
        if bool(pd.isna(value)):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return text or fallback


def _number(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _feature_id(row: Any, fallback: str) -> str:
    for name in ("id", "osm_id", "osmid"):
        value = row.get(name)
        if value is not None:
            try:
                if not bool(pd.isna(value)):
                    return str(value)
            except (TypeError, ValueError):
                return str(value)
    return fallback


def _road_class(row: Any) -> str:
    value = _text(row.get("road_class"), "")
    if value in {"major", "secondary", "local"}:
        return value
    highway = _text(row.get("highway"), "")
    if highway in {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}:
        return "major"
    if highway in {"secondary", "secondary_link", "tertiary", "tertiary_link"}:
        return "secondary"
    return "local"


def _rail_class(row: Any) -> str:
    value = _text(row.get("railway"), "rail")
    return value if value in {"rail", "subway", "light_rail", "tram", "monorail"} else "rail"


def _vertical_mode(row: Any) -> str:
    value = _text(row.get("vertical_mode"), "unknown")
    return value if value in {"surface", "underground", "elevated", "unknown"} else "unknown"


def _union_area(frame: gpd.GeoDataFrame) -> float:
    geometries = [geometry for geometry in frame.geometry if geometry is not None and not geometry.is_empty]
    if not geometries:
        return 0.0
    return float(unary_union(geometries).area)


def _line_length(frame: gpd.GeoDataFrame) -> float:
    return float(frame.geometry.length.sum()) if not frame.empty else 0.0


def _region_stats(layers: CityLayers, geometry: Polygon) -> dict[str, float]:
    area = max(float(geometry.area), 1.0)
    area_km2 = area / 1_000_000.0

    roads = _clip_frame(layers.roads, geometry)
    rail = _clip_frame(layers.rail, geometry)
    buildings = _clip_frame(layers.buildings, geometry)
    water = _clip_frame(layers.water, geometry)
    green = _clip_frame(layers.green, geometry)

    road_lengths = {"major": 0.0, "secondary": 0.0, "local": 0.0}
    road_vertical = {"surface": 0.0, "underground": 0.0, "elevated": 0.0, "unknown": 0.0}
    for index, row in roads.iterrows():
        length = float(row.geometry.length)
        road_lengths[_road_class(row)] += length
        road_vertical[_vertical_mode(row)] += length

    rail_vertical = {"surface": 0.0, "underground": 0.0, "elevated": 0.0, "unknown": 0.0}
    for index, row in rail.iterrows():
        rail_vertical[_vertical_mode(row)] += float(row.geometry.length)

    road_total = sum(road_lengths.values())
    rail_total = _line_length(rail)

    heights = []
    for value in buildings.get("estimated_height_m", pd.Series(dtype=float)):
        number = _number(value, math.nan)
        if math.isfinite(number) and number > 0:
            heights.append(number)

    return {
        "road_length_km_per_km2": road_total / 1000.0 / area_km2,
        "rail_length_km_per_km2": rail_total / 1000.0 / area_km2,
        "road_major_share": road_lengths["major"] / road_total if road_total else 0.0,
        "road_secondary_share": road_lengths["secondary"] / road_total if road_total else 0.0,
        "road_local_share": road_lengths["local"] / road_total if road_total else 0.0,
        "road_surface_share": road_vertical["surface"] / road_total if road_total else 0.0,
        "road_underground_share": road_vertical["underground"] / road_total if road_total else 0.0,
        "road_elevated_share": road_vertical["elevated"] / road_total if road_total else 0.0,
        "rail_surface_share": rail_vertical["surface"] / rail_total if rail_total else 0.0,
        "rail_underground_share": rail_vertical["underground"] / rail_total if rail_total else 0.0,
        "rail_elevated_share": rail_vertical["elevated"] / rail_total if rail_total else 0.0,
        "building_coverage": _union_area(buildings) / area,
        "mean_building_height_m": float(np.mean(heights)) if heights else 0.0,
        "water_coverage": _union_area(water) / area,
        "green_coverage": _union_area(green) / area,
    }


def _grid_cells(
    city_id: str,
    bounds: tuple[float, float, float, float],
    size_m: float,
) -> list[dict[str, Any]]:
    minx, miny, maxx, maxy = bounds
    first_col = math.floor(minx / size_m)
    first_row = math.floor(miny / size_m)
    last_col = math.floor((maxx - 1e-9) / size_m)
    last_row = math.floor((maxy - 1e-9) / size_m)

    cells = []
    for row in range(first_row, last_row + 1):
        for column in range(first_col, last_col + 1):
            cell_minx = column * size_m
            cell_miny = row * size_m
            cell = box(cell_minx, cell_miny, cell_minx + size_m, cell_miny + size_m)
            cells.append(
                {
                    "id": f"{city_id}_r{row:+06d}_c{column:+06d}",
                    "row": row,
                    "column": column,
                    "geometry": cell,
                }
            )
    return cells


def _heading_at(geometry: BaseGeometry, point: Point, probe_m: float) -> tuple[float, float]:
    lines = list(_iter_lines(geometry))
    if not lines:
        return (0.0, 0.0)
    line = min(lines, key=lambda value: value.distance(point))
    distance = float(line.project(point))
    before = line.interpolate(max(0.0, distance - probe_m))
    after = line.interpolate(min(float(line.length), distance + probe_m))
    dx = float(after.x - before.x)
    dy = float(after.y - before.y)
    norm = math.hypot(dx, dy)
    if norm <= 1e-8:
        return (0.0, 0.0)
    return (dx / norm, dy / norm)


def _boundary_ports(
    layers: CityLayers,
    boundary: BaseGeometry,
    *,
    direction_hint: tuple[float, float],
    probe_m: float,
) -> list[dict[str, Any]]:
    ports: list[dict[str, Any]] = []
    for mode, frame in (("road", layers.roads), ("rail", layers.rail)):
        if frame.empty:
            continue
        query = boundary.buffer(0.05)
        indexes = list(frame.sindex.query(query, predicate="intersects"))
        for index in indexes:
            row = frame.iloc[index]
            geometry = row.geometry
            intersection = geometry.intersection(boundary)
            for point_index, point in enumerate(_iter_points(intersection)):
                hx, hy = _heading_at(geometry, point, probe_m)
                if hx * direction_hint[0] + hy * direction_hint[1] < 0:
                    hx, hy = -hx, -hy
                port = {
                    "mode": mode,
                    "class": _road_class(row) if mode == "road" else _rail_class(row),
                    "vertical_mode": _vertical_mode(row),
                    "position_projected_m": [float(point.x), float(point.y)],
                    "heading": [hx, hy],
                    "source_id": _feature_id(row, f"{mode}:{index}:{point_index}"),
                }
                if mode == "road":
                    port["width_m"] = _number(row.get("estimated_width_m"), 5.0)
                ports.append(port)

    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for port in ports:
        key = (
            port["mode"],
            port["class"],
            port["vertical_mode"],
            round(port["position_projected_m"][0], 2),
            round(port["position_projected_m"][1], 2),
        )
        unique[key] = port
    return sorted(
        unique.values(),
        key=lambda value: (
            value["mode"],
            value["class"],
            value["position_projected_m"][0],
            value["position_projected_m"][1],
        ),
    )


def _shared_boundary(left: Polygon, right: Polygon) -> BaseGeometry:
    return left.boundary.intersection(right.boundary)


def _region_graph(
    layers: CityLayers,
    city_id: str,
    city_bounds: tuple[float, float, float, float],
    config: ContextGraphConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, int], str]]:
    cells = _grid_cells(city_id, city_bounds, config.region_size_m)
    lookup = {(cell["row"], cell["column"]): cell for cell in cells}
    id_lookup = {(cell["row"], cell["column"]): cell["id"] for cell in cells}

    city_minx, city_miny, city_maxx, city_maxy = city_bounds
    city_width = max(city_maxx - city_minx, 1.0)
    city_height = max(city_maxy - city_miny, 1.0)

    nodes = []
    for cell in cells:
        geometry = cell["geometry"]
        center = geometry.centroid
        nodes.append(
            {
                "id": cell["id"],
                "row": cell["row"],
                "column": cell["column"],
                "bounds_projected_m": list(map(float, geometry.bounds)),
                "center_projected_m": [float(center.x), float(center.y)],
                "center_city_normalized": [
                    ((center.x - city_minx) / city_width) * 2.0 - 1.0,
                    ((center.y - city_miny) / city_height) * 2.0 - 1.0,
                ],
                "features": _region_stats(layers, geometry),
            }
        )

    offsets = [(1, 0), (0, 1)]
    if config.include_diagonal_region_edges:
        offsets.extend([(1, 1), (1, -1)])

    edges = []
    for (row, column), cell in lookup.items():
        left_center = cell["geometry"].centroid
        for dc, dr in offsets:
            other = lookup.get((row + dr, column + dc))
            if other is None:
                continue
            right_center = other["geometry"].centroid
            dx = float(right_center.x - left_center.x)
            dy = float(right_center.y - left_center.y)
            distance = max(math.hypot(dx, dy), 1e-8)
            shared = _shared_boundary(cell["geometry"], other["geometry"])
            cardinal = not shared.is_empty and float(shared.length) > 0
            ports = []
            if cardinal:
                ports = _boundary_ports(
                    layers,
                    shared,
                    direction_hint=(dx / distance, dy / distance),
                    probe_m=config.port_probe_m,
                )
            edges.append(
                {
                    "from": cell["id"],
                    "to": other["id"],
                    "offset": [dc, dr],
                    "distance_m": distance,
                    "shared_boundary": cardinal,
                    "transport_ports": ports,
                    "road_port_count": sum(port["mode"] == "road" for port in ports),
                    "rail_port_count": sum(port["mode"] == "rail" for port in ports),
                }
            )
    return nodes, edges, id_lookup


def _local_line_payload(
    frame: gpd.GeoDataFrame,
    target: Polygon,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    clipped = _clip_frame(frame, target)
    minx, miny, _maxx, _maxy = target.bounds
    records = []
    for index, row in clipped.iterrows():
        for part_index, line in enumerate(_iter_lines(row.geometry)):
            coordinates = [
                [float(x - minx), float(y - miny)]
                for x, y, *_rest in line.coords
            ]
            if len(coordinates) < 2:
                continue
            record = {
                "id": _feature_id(row, f"{mode}:{index}:{part_index}"),
                "mode": mode,
                "class": _road_class(row) if mode == "road" else _rail_class(row),
                "vertical_mode": _vertical_mode(row),
                "geometry_local_m": coordinates,
                "length_m": float(line.length),
            }
            if mode == "road":
                record["width_m"] = _number(row.get("estimated_width_m"), 5.0)
            records.append(record)
    return records


def _local_buildings(frame: gpd.GeoDataFrame, target: Polygon) -> list[dict[str, Any]]:
    clipped = _clip_frame(frame, target)
    minx, miny, _maxx, _maxy = target.bounds
    records = []
    for index, row in clipped.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty or geometry.area <= 1e-6:
            continue
        from shapely import affinity

        local = affinity.translate(geometry, xoff=-minx, yoff=-miny)
        records.append(
            {
                "id": _feature_id(row, f"building:{index}"),
                "footprint_local_m": mapping(local),
                "height_m": _number(row.get("estimated_height_m"), 0.0),
                "height_confidence": int(_number(row.get("height_confidence"), 0.0)),
                "height_source": _text(row.get("height_source"), "unknown"),
            }
        )
    return records


def _target_ports(
    layers: CityLayers,
    target: Polygon,
    probe_m: float,
) -> list[dict[str, Any]]:
    center = target.centroid
    ports = _boundary_ports(
        layers,
        target.boundary,
        direction_hint=(1.0, 0.0),
        probe_m=probe_m,
    )
    for port in ports:
        px, py = port["position_projected_m"]
        hx, hy = port["heading"]
        inward_x = float(center.x - px)
        inward_y = float(center.y - py)
        if hx * inward_x + hy * inward_y < 0:
            port["heading"] = [-hx, -hy]
        port["position_local_m"] = [
            float(px - target.bounds[0]),
            float(py - target.bounds[1]),
        ]
    return ports


def _target_samples(
    layers: CityLayers,
    city_id: str,
    city_bounds: tuple[float, float, float, float],
    region_lookup: dict[tuple[int, int], str],
    config: ContextGraphConfig,
    output_dir: Path,
    *,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    minx, miny, maxx, maxy = city_bounds
    size = config.target_size_m
    stride = config.target_stride_m

    first_col = math.floor(minx / stride)
    first_row = math.floor(miny / stride)
    last_col = math.floor((maxx - 1e-9) / stride)
    last_row = math.floor((maxy - 1e-9) / stride)

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    row_count = last_row - first_row + 1
    column_count = last_col - first_col + 1
    total_cells = row_count * column_count
    scanned = 0

    for row in range(first_row, last_row + 1):
        for column in range(first_col, last_col + 1):
            scanned += 1
            if show_progress and (scanned == 1 or scanned % 250 == 0 or scanned == total_cells):
                print(
                    f"targets: {scanned}/{total_cells} scanned, {len(rows)} kept",
                    flush=True,
                )
            target_minx = column * stride
            target_miny = row * stride
            target = box(
                target_minx,
                target_miny,
                target_minx + size,
                target_miny + size,
            )
            roads = _clip_frame(layers.roads, target)
            rail = _clip_frame(layers.rail, target)
            transport_length = _line_length(roads) + _line_length(rail)
            if transport_length < config.minimum_transport_length_m:
                continue

            center = target.centroid
            region_col = math.floor(float(center.x) / config.region_size_m)
            region_row = math.floor(float(center.y) / config.region_size_m)
            parent_id = region_lookup.get((region_row, region_col))
            if parent_id is None:
                continue

            context = box(
                center.x - config.region_size_m / 2.0,
                center.y - config.region_size_m / 2.0,
                center.x + config.region_size_m / 2.0,
                center.y + config.region_size_m / 2.0,
            )
            target_id = f"{city_id}_t{row:+06d}_{column:+06d}"
            context_min_col = math.floor(float(context.bounds[0]) / config.region_size_m)
            context_min_row = math.floor(float(context.bounds[1]) / config.region_size_m)
            context_max_col = math.floor((float(context.bounds[2]) - 1e-9) / config.region_size_m)
            context_max_row = math.floor((float(context.bounds[3]) - 1e-9) / config.region_size_m)
            context_region_ids = [
                region_lookup[(region_y, region_x)]
                for region_y in range(context_min_row, context_max_row + 1)
                for region_x in range(context_min_col, context_max_col + 1)
                if (region_y, region_x) in region_lookup
            ]
            visible_context = context.difference(target)

            payload = {
                "format": "urban-context-target",
                "version": "0.1.0",
                "id": target_id,
                "city_id": city_id,
                "target_bounds_projected_m": list(map(float, target.bounds)),
                "context_bounds_projected_m": list(map(float, context.bounds)),
                "parent_region_id": parent_id,
                "input": {
                    "context_region_ids": context_region_ids,
                    "masked_region_ids": [parent_id],
                    "visible_region_ids": [
                        region_id for region_id in context_region_ids if region_id != parent_id
                    ],
                    "boundary_ports": _target_ports(layers, target, config.port_probe_m),
                    "visible_context_features": _region_stats(layers, visible_context),
                    "terrain": {
                        "available": False,
                        "note": "No DEM supplied to context-graph-v1 yet.",
                    },
                    "note": (
                        "Target geometry is hidden from local context features. The parent region "
                        "is marked as masked; boundary ports are the explicit continuation signal."
                    ),
                },
                "target": {
                    "roads": _local_line_payload(layers.roads, target, mode="road"),
                    "rail": _local_line_payload(layers.rail, target, mode="rail"),
                    "buildings": _local_buildings(layers.buildings, target),
                    "z_supervision": {
                        "metric_transport_z": False,
                        "policy": (
                            "vertical mode is supervised; exact metric transport Z is masked "
                            "until terrain/elevation evidence is available"
                        ),
                    },
                },
            }

            sample_path = samples_dir / f"{target_id}.json.gz"
            with gzip.open(sample_path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)

            rows.append(
                {
                    "id": target_id,
                    "city_id": city_id,
                    "sample_path": sample_path.relative_to(output_dir).as_posix(),
                    "parent_region_id": parent_id,
                    "target_bounds_projected_m": list(map(float, target.bounds)),
                    "context_bounds_projected_m": list(map(float, context.bounds)),
                    "context_regions": len(context_region_ids),
                    "transport_length_m": transport_length,
                    "boundary_ports": len(payload["input"]["boundary_ports"]),
                    "roads": len(payload["target"]["roads"]),
                    "rail": len(payload["target"]["rail"]),
                    "buildings": len(payload["target"]["buildings"]),
                }
            )
    return rows


def build_context_graph(
    city_path: str | Path,
    output_dir: str | Path,
    *,
    config: ContextGraphConfig | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    city_path = Path(city_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or ContextGraphConfig()

    layers, metadata = load_city_gpkg(city_path)
    city_id = str(metadata.get("city_id") or city_path.stem)
    city_bounds = _frame_bounds(layers)
    crs = str(next(frame.crs for _name, frame in layers.items() if not frame.empty))

    nodes, edges, region_lookup = _region_graph(
        layers,
        city_id,
        city_bounds,
        config,
    )
    graph = {
        "format": "urban-city-context-graph",
        "version": "0.1.0",
        "city_id": city_id,
        "source_city": str(city_path),
        "crs": crs,
        "city_bounds_projected_m": list(city_bounds),
        "config": asdict(config),
        "nodes": nodes,
        "edges": edges,
        "notes": {
            "region_edges": (
                "Spatial adjacency plus data-derived road/rail continuation ports. "
                "The graph does not invent transport."
            ),
            "coordinates": (
                "Projected coordinates are retained for data inspection. Models should prefer "
                "relative region offsets and per-city normalised positions."
            ),
        },
    }
    (output_dir / "context-graph.json").write_text(
        json.dumps(graph, indent=2),
        encoding="utf-8",
    )

    samples = _target_samples(
        layers,
        city_id,
        city_bounds,
        region_lookup,
        config,
        output_dir,
        show_progress=show_progress,
    )
    with (output_dir / "targets.jsonl").open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "city_id": city_id,
        "source_city": str(city_path),
        "crs": crs,
        "city_bounds_projected_m": list(city_bounds),
        "region_nodes": len(nodes),
        "region_edges": len(edges),
        "region_edges_with_roads": sum(edge["road_port_count"] > 0 for edge in edges),
        "region_edges_with_rail": sum(edge["rail_port_count"] > 0 for edge in edges),
        "targets": len(samples),
        "targets_with_boundary_ports": sum(row["boundary_ports"] > 0 for row in samples),
        "targets_with_rail": sum(row["rail"] > 0 for row in samples),
        "config": asdict(config),
        "metric_transport_z_supervised": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
