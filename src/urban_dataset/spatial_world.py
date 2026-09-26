from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon, box
from shapely.geometry.base import BaseGeometry

from urban_dataset.city_state import build_transport_graph
from urban_dataset.context_graph import (
    _clip_frame,
    _feature_id,
    _frame_bounds,
    _local_buildings,
    _local_polygons,
    _rail_class,
    _region_stats,
    _road_class,
    _target_ports,
    _vertical_mode,
)
from urban_dataset.prepared import load_city_gpkg
from urban_dataset.tile import TileSpec


@dataclass(frozen=True)
class SpatialWorldConfig:
    context_size_m: float = 5120.0
    local_vector_size_m: float = 2560.0
    target_size_m: float = 1024.0
    target_stride_m: float = 1024.0
    context_cell_m: float = 512.0
    minimum_transport_length_m: float = 100.0
    port_probe_m: float = 12.0


def _iter_lines(geometry: BaseGeometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
        return
    if isinstance(geometry, MultiLineString | GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_lines(part)


def _line_payload(
    frame: gpd.GeoDataFrame,
    geometry: BaseGeometry,
    *,
    mode: str,
    origin_x: float,
    origin_y: float,
) -> list[dict[str, Any]]:
    clipped = _clip_frame(frame, geometry)
    records = []
    for index, row in clipped.iterrows():
        for part_index, line in enumerate(_iter_lines(row.geometry)):
            coordinates = [
                [float(x - origin_x), float(y - origin_y)]
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
                width = row.get("estimated_width_m")
                try:
                    width = float(width)
                except (TypeError, ValueError):
                    width = None
                if width is not None and math.isfinite(width) and width > 0:
                    record["width_m"] = width
            records.append(record)
    return records


def _context_cells(layers, context: Polygon, target: Polygon, cell_m: float):
    minx, miny, maxx, maxy = context.bounds
    columns = int(round((maxx - minx) / cell_m))
    rows = int(round((maxy - miny) / cell_m))
    values = []
    for row in range(rows):
        for column in range(columns):
            cell = box(
                minx + column * cell_m,
                miny + row * cell_m,
                minx + (column + 1) * cell_m,
                miny + (row + 1) * cell_m,
            )
            visible = cell.difference(target)
            stats = _region_stats(layers, visible) if not visible.is_empty else {}
            values.append(
                {
                    "row": row,
                    "column": column,
                    "center_local_m": [
                        float(cell.centroid.x - target.bounds[0]),
                        float(cell.centroid.y - target.bounds[1]),
                    ],
                    "masked_fraction": float(cell.intersection(target).area / max(cell.area, 1.0)),
                    "features": stats,
                }
            )
    return values


def _edge_signature(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("transport_mode") or "road"),
        str(edge.get("class") or "local"),
        str(edge.get("vertical_mode") or "unknown"),
    )


def _oriented_edge(edge: dict[str, Any], node_id: str):
    if str(edge["from_node"]) == node_id:
        return str(edge["to_node"]), list(edge["geometry_local_m"])
    return str(edge["from_node"]), list(reversed(edge["geometry_local_m"]))


def simplify_transport_graph(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = {str(node["id"]): dict(node) for node in graph["nodes"]}
    edges = [dict(edge) for edge in graph["edges"]]
    adjacency: dict[str, list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        adjacency[str(edge["from_node"])].append(index)
        adjacency[str(edge["to_node"])].append(index)

    critical = set()
    for node_id, indexes in adjacency.items():
        node = nodes[node_id]
        signatures = {_edge_signature(edges[index]) for index in indexes}
        if (
            node.get("boundary_port_key") is not None
            or len(indexes) != 2
            or len(signatures) != 1
        ):
            critical.add(node_id)

    visited = set()
    simplified = []

    def emit(start: str, first_index: int):
        signature = _edge_signature(edges[first_index])
        current = start
        edge_index = first_index
        geometry = []
        source_ids = []
        widths = []
        lengths = []
        while True:
            if edge_index in visited:
                return
            visited.add(edge_index)
            edge = edges[edge_index]
            next_node, coordinates = _oriented_edge(edge, current)
            if geometry:
                geometry.extend(coordinates[1:])
            else:
                geometry.extend(coordinates)
            source_ids.append(str(edge.get("source_id") or edge["id"]))
            widths.append(float(edge.get("width_m", 5.0)))
            lengths.append(float(edge.get("length_m", 0.0)))
            if next_node in critical:
                end = next_node
                break
            candidates = [
                index
                for index in adjacency[next_node]
                if index not in visited and _edge_signature(edges[index]) == signature
            ]
            if len(candidates) != 1:
                end = next_node
                break
            current = next_node
            edge_index = candidates[0]

        simplified.append(
            {
                "id": f"chain:{len(simplified)}",
                "from_node": start,
                "to_node": end,
                "transport_mode": signature[0],
                "class": signature[1],
                "vertical_mode": signature[2],
                "width_m": sum(width * length for width, length in zip(widths, lengths, strict=True))
                / max(sum(lengths), 1e-6),
                "length_m": float(LineString([point[:2] for point in geometry]).length),
                "geometry_local_m": geometry,
                "source_ids": source_ids,
            }
        )

    for node_id in sorted(critical):
        for edge_index in adjacency[node_id]:
            if edge_index not in visited:
                emit(node_id, edge_index)

    for edge_index, edge in enumerate(edges):
        if edge_index in visited:
            continue
        visited.add(edge_index)
        simplified.append(
            {
                "id": f"cycle:{len(simplified)}",
                "from_node": str(edge["from_node"]),
                "to_node": str(edge["to_node"]),
                "transport_mode": str(edge.get("transport_mode") or "road"),
                "class": str(edge.get("class") or "local"),
                "vertical_mode": str(edge.get("vertical_mode") or "unknown"),
                "width_m": float(edge.get("width_m", 5.0)),
                "length_m": float(edge.get("length_m", 0.0)),
                "geometry_local_m": edge["geometry_local_m"],
                "source_ids": [str(edge.get("source_id") or edge["id"])],
            }
        )

    used = {
        node_id
        for edge in simplified
        for node_id in (str(edge["from_node"]), str(edge["to_node"]))
    }
    degree = defaultdict(int)
    for edge in simplified:
        degree[str(edge["from_node"])] += 1
        degree[str(edge["to_node"])] += 1

    result_nodes = []
    for node_id in sorted(used):
        node = nodes[node_id]
        node["degree"] = degree[node_id]
        if node.get("boundary_port_key") is not None:
            node["node_type"] = "boundary_port"
        elif degree[node_id] <= 1:
            node["node_type"] = "endpoint"
        elif degree[node_id] >= 3:
            node["node_type"] = "intersection"
        else:
            node["node_type"] = "continuation"
        result_nodes.append(node)

    return {
        "nodes": result_nodes,
        "edges": simplified,
        "statistics": {
            "nodes": len(result_nodes),
            "edges": len(simplified),
            "raw_nodes": len(graph["nodes"]),
            "raw_edges": len(graph["edges"]),
            "boundary_ports": sum(
                node.get("boundary_port_key") is not None
                for node in result_nodes
            ),
        },
    }


def _target_graph(layers, target: Polygon, city_id: str):
    roads = _clip_frame(layers.roads, target)
    rail = _clip_frame(layers.rail, target)
    tile = TileSpec(
        city_id=city_id,
        column=0,
        row=0,
        minx=float(target.bounds[0]),
        miny=float(target.bounds[1]),
        maxx=float(target.bounds[2]),
        maxy=float(target.bounds[3]),
    )
    return simplify_transport_graph(build_transport_graph(roads, rail, tile))


def _line_length(frame: gpd.GeoDataFrame) -> float:
    return float(frame.geometry.length.sum()) if not frame.empty else 0.0


def build_spatial_world(
    city_path: str | Path,
    output_dir: str | Path,
    *,
    config: SpatialWorldConfig | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    city_path = Path(city_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or SpatialWorldConfig()

    layers, metadata = load_city_gpkg(city_path)
    city_id = str(metadata.get("city_id") or city_path.stem)
    city_bounds = _frame_bounds(layers)
    whole_city = box(*city_bounds)
    city_style = _region_stats(layers, whole_city)

    minx, miny, maxx, maxy = city_bounds
    stride = config.target_stride_m
    first_col = math.floor(minx / stride)
    first_row = math.floor(miny / stride)
    last_col = math.floor((maxx - 1e-9) / stride)
    last_row = math.floor((maxy - 1e-9) / stride)
    total = (last_col - first_col + 1) * (last_row - first_row + 1)

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    scanned = 0

    for row in range(first_row, last_row + 1):
        for column in range(first_col, last_col + 1):
            scanned += 1
            target_minx = column * stride
            target_miny = row * stride
            target = box(
                target_minx,
                target_miny,
                target_minx + config.target_size_m,
                target_miny + config.target_size_m,
            )
            roads = _clip_frame(layers.roads, target)
            rail = _clip_frame(layers.rail, target)
            transport_length = _line_length(roads) + _line_length(rail)
            if transport_length < config.minimum_transport_length_m:
                if show_progress and (scanned == 1 or scanned % 100 == 0 or scanned == total):
                    print(f"spatial world: {scanned}/{total} scanned, {len(rows)} kept", flush=True)
                continue

            center = target.centroid
            context = box(
                center.x - config.context_size_m / 2.0,
                center.y - config.context_size_m / 2.0,
                center.x + config.context_size_m / 2.0,
                center.y + config.context_size_m / 2.0,
            )
            local_vector = box(
                center.x - config.local_vector_size_m / 2.0,
                center.y - config.local_vector_size_m / 2.0,
                center.x + config.local_vector_size_m / 2.0,
                center.y + config.local_vector_size_m / 2.0,
            )
            visible_local = local_vector.difference(target)
            sample_id = f"{city_id}_w{row:+06d}_{column:+06d}"
            ports = _target_ports(layers, target, config.port_probe_m)
            for port in ports:
                port["position_local_m"] = [
                    float(port["position_projected_m"][0] - target.bounds[0]),
                    float(port["position_projected_m"][1] - target.bounds[1]),
                ]

            graph = _target_graph(layers, target, city_id)
            payload = {
                "format": "aether-spatial-world-sample",
                "version": "0.1.0",
                "id": sample_id,
                "city_id": city_id,
                "coordinate_system": {
                    "units": "metres",
                    "source_projected_crs": str(layers.roads.crs),
                    "target_origin_projected_m": [
                        float(target.bounds[0]),
                        float(target.bounds[1]),
                    ],
                },
                "bounds": {
                    "target_projected_m": list(map(float, target.bounds)),
                    "local_vector_projected_m": list(map(float, local_vector.bounds)),
                    "context_projected_m": list(map(float, context.bounds)),
                },
                "style": city_style,
                "controls": _region_stats(layers, target),
                "input": {
                    "context_cells": _context_cells(
                        layers,
                        context,
                        target,
                        config.context_cell_m,
                    ),
                    "visible_transport": {
                        "roads": _line_payload(
                            layers.roads,
                            visible_local,
                            mode="road",
                            origin_x=target.bounds[0],
                            origin_y=target.bounds[1],
                        ),
                        "rail": _line_payload(
                            layers.rail,
                            visible_local,
                            mode="rail",
                            origin_x=target.bounds[0],
                            origin_y=target.bounds[1],
                        ),
                    },
                    "boundary_ports": ports,
                },
                "target": {
                    "transport_graph": graph,
                    "buildings": _local_buildings(layers.buildings, target),
                    "green": _local_polygons(layers.green, target, kind="green"),
                    "water": _local_polygons(layers.water, target, kind="water"),
                    "landuse": _local_polygons(layers.landuse, target, kind="landuse"),
                },
            }

            path = samples_dir / f"{sample_id}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)

            rows.append(
                {
                    "id": sample_id,
                    "city_id": city_id,
                    "sample_path": path.relative_to(output_dir).as_posix(),
                    "transport_length_m": transport_length,
                    "ports": len(ports),
                    "context_cells": len(payload["input"]["context_cells"]),
                    "visible_roads": len(payload["input"]["visible_transport"]["roads"]),
                    "visible_rail": len(payload["input"]["visible_transport"]["rail"]),
                    "nodes": graph["statistics"]["nodes"],
                    "edges": graph["statistics"]["edges"],
                    "raw_nodes": graph["statistics"]["raw_nodes"],
                    "raw_edges": graph["statistics"]["raw_edges"],
                    "buildings": len(payload["target"]["buildings"]),
                    "green": len(payload["target"]["green"]),
                    "water": len(payload["target"]["water"]),
                    "landuse": len(payload["target"]["landuse"]),
                }
            )
            if show_progress and (scanned == 1 or scanned % 100 == 0 or scanned == total):
                print(f"spatial world: {scanned}/{total} scanned, {len(rows)} kept", flush=True)

    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "format": "aether-spatial-world",
        "version": "0.1.0",
        "city_id": city_id,
        "source_city": str(city_path),
        "city_bounds_projected_m": list(map(float, city_bounds)),
        "samples": len(rows),
        "style": city_style,
        "config": asdict(config),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
