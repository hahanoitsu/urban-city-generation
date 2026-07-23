from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    box,
    mapping,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import split, unary_union

from .obj_export import _ObjMesh, _iter_polygons, _write_materials
from .utils import write_json

NETWORK_SCENE_VERSION = "0.1.0"
ROAD_COLOURS = {
    "major": (220, 68, 55),
    "secondary": (235, 142, 65),
    "local": (242, 221, 139),
}


def _stable_id(prefix: str, *parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _iter_lines(geometry: BaseGeometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
        return
    if isinstance(geometry, MultiLineString | GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_lines(part)


def _load_states(
    dataset_root: Path,
    *,
    city_id: str | None,
    area_id: str | None,
    max_tiles: int | None,
) -> list[tuple[Path, dict[str, Any]]]:
    selected: list[tuple[Path, dict[str, Any]]] = []
    for state_path in sorted(dataset_root.glob("**/tiles/*/city.json")):
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        tile = payload.get("tile", {})
        if city_id is not None and str(tile.get("city_id")) != city_id:
            continue
        if area_id is not None and str(tile.get("area_id")) != area_id:
            continue
        if payload.get("format") != "urban-city-state-tile":
            continue
        selected.append((state_path, payload))
        if max_tiles is not None and len(selected) >= max_tiles:
            break
    if not selected:
        filters = {"city": city_id, "area": area_id, "max_tiles": max_tiles}
        raise FileNotFoundError(
            f"No city-state tiles matched below {dataset_root}: {filters}"
        )
    return selected


def _node_key(node: dict[str, Any], tolerance_m: float) -> tuple[Any, ...]:
    position = node.get("position_projected_m")
    if not isinstance(position, list) or len(position) < 2:
        raise ValueError(f"Node is missing projected coordinates: {node.get('id')}")
    layer = node.get("layer_order")
    return (
        node.get("transport_mode"),
        node.get("vertical_mode"),
        None if layer is None else round(float(layer), 4),
        round(float(position[0]) / tolerance_m),
        round(float(position[1]) / tolerance_m),
    )


def _canonical_geometry_key(
    coordinates: list[list[float | None]],
    *,
    precision: int = 3,
) -> tuple[tuple[float | None, ...], ...]:
    rounded = tuple(
        tuple(None if value is None else round(float(value), precision) for value in point)
        for point in coordinates
    )
    reversed_coordinates = tuple(reversed(rounded))
    return min(rounded, reversed_coordinates)


def _translate_geometry_payload(
    geometry_payload: dict[str, Any],
    x_offset: float,
    y_offset: float,
) -> BaseGeometry:
    geometry = shape(geometry_payload)
    return affinity.translate(geometry, xoff=x_offset, yoff=y_offset)


def _recursive_parcels(
    polygon: Polygon,
    *,
    target_area_m2: float,
    minimum_area_m2: float,
    depth: int = 0,
    max_depth: int = 10,
) -> list[Polygon]:
    if polygon.area <= target_area_m2 * 1.5 or depth >= max_depth:
        return [polygon]
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    if min(width, height) < 8.0:
        return [polygon]
    if width >= height:
        coordinate = (minx + maxx) / 2.0
        cutter = LineString([(coordinate, miny - height - 10), (coordinate, maxy + height + 10)])
    else:
        coordinate = (miny + maxy) / 2.0
        cutter = LineString([(minx - width - 10, coordinate), (maxx + width + 10, coordinate)])
    pieces = [part for part in split(polygon, cutter).geoms if isinstance(part, Polygon)]
    pieces = [part for part in pieces if part.area >= minimum_area_m2]
    if len(pieces) < 2:
        return [polygon]
    result: list[Polygon] = []
    for piece in pieces:
        result.extend(
            _recursive_parcels(
                piece,
                target_area_m2=target_area_m2,
                minimum_area_m2=minimum_area_m2,
                depth=depth + 1,
                max_depth=max_depth,
            )
        )
    return result


def _derive_blocks_and_parcels(
    coverage: BaseGeometry,
    surface_roads: list[tuple[LineString, float]],
    water: list[BaseGeometry],
    *,
    scene_origin: tuple[float, float],
    minimum_block_area_m2: float,
    target_parcel_area_m2: float,
    minimum_parcel_area_m2: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], BaseGeometry]:
    road_surfaces = [
        line.buffer(max(width, 1.0) / 2.0, cap_style="flat", join_style="round")
        for line, width in surface_roads
        if not line.is_empty and line.length > 0
    ]
    road_union = unary_union(road_surfaces) if road_surfaces else GeometryCollection()
    water_union = unary_union(water) if water else GeometryCollection()
    remaining = coverage.difference(road_union).difference(water_union)
    boundary = coverage.boundary.buffer(0.5)

    blocks: list[dict[str, Any]] = []
    parcels: list[dict[str, Any]] = []
    for polygon in _iter_polygons(remaining):
        if polygon.area < minimum_block_area_m2:
            continue
        centroid = polygon.centroid
        block_id = _stable_id(
            "block",
            round(centroid.x, 2),
            round(centroid.y, 2),
            round(polygon.area, 1),
        )
        local_polygon = affinity.translate(
            polygon,
            xoff=-scene_origin[0],
            yoff=-scene_origin[1],
        )
        block = {
            "id": block_id,
            "area_m2": float(polygon.area),
            "perimeter_m": float(polygon.length),
            "touches_scene_boundary": bool(polygon.intersects(boundary)),
            "geometry_local_m": mapping(local_polygon),
        }
        blocks.append(block)
        for parcel_index, parcel in enumerate(
            _recursive_parcels(
                polygon,
                target_area_m2=target_parcel_area_m2,
                minimum_area_m2=minimum_parcel_area_m2,
            )
        ):
            if parcel.area < minimum_parcel_area_m2:
                continue
            local_parcel = affinity.translate(
                parcel,
                xoff=-scene_origin[0],
                yoff=-scene_origin[1],
            )
            parcels.append(
                {
                    "id": _stable_id(block_id, "parcel", parcel_index),
                    "block_id": block_id,
                    "area_m2": float(parcel.area),
                    "geometry_local_m": mapping(local_parcel),
                }
            )
    blocks.sort(key=lambda item: item["id"])
    parcels.sort(key=lambda item: item["id"])
    return blocks, parcels, road_union


def compile_network_scene(
    dataset_root: str | Path,
    output_json: str | Path,
    *,
    city_id: str | None = None,
    area_id: str | None = None,
    max_tiles: int | None = None,
    stitch_tolerance_m: float = 0.10,
    minimum_block_area_m2: float = 400.0,
    target_parcel_area_m2: float = 900.0,
    minimum_parcel_area_m2: float = 120.0,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_json = Path(output_json).expanduser().resolve()
    states = _load_states(
        dataset_root,
        city_id=city_id,
        area_id=area_id,
        max_tiles=max_tiles,
    )

    crs_values = {
        str(payload.get("coordinate_system", {}).get("source_projected_crs"))
        for _, payload in states
    }
    if len(crs_values) != 1:
        raise ValueError(
            "A compiled scene must use one projected CRS. Filter to one city before combining."
        )
    source_crs = next(iter(crs_values))

    tile_records: list[dict[str, Any]] = []
    tile_coverage: list[Polygon] = []
    water_geometries: list[BaseGeometry] = []
    node_lookup: dict[tuple[Any, ...], str] = {}
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[Any, ...]] = set()
    buildings: list[dict[str, Any]] = []
    building_keys: set[tuple[Any, ...]] = set()

    origins: list[tuple[float, float]] = []
    for state_path, payload in states:
        coordinate = payload["coordinate_system"]
        origin = tuple(float(value) for value in coordinate["origin_projected"][:2])
        origins.append(origin)
        local_bounds = [float(value) for value in coordinate["local_bounds"]]
        tile_polygon = box(
            origin[0] + local_bounds[0],
            origin[1] + local_bounds[1],
            origin[0] + local_bounds[2],
            origin[1] + local_bounds[3],
        )
        tile_coverage.append(tile_polygon)
        tile_info = payload.get("tile", {})
        tile_records.append(
            {
                "tile_id": tile_info.get("tile_id"),
                "city_id": tile_info.get("city_id"),
                "area_id": tile_info.get("area_id"),
                "state_path": state_path.relative_to(dataset_root).as_posix(),
                "origin_projected_m": list(origin),
            }
        )

        local_to_global: dict[str, str] = {}
        graph = payload.get("transport_graph", {})
        for node in graph.get("nodes", []):
            key = _node_key(node, stitch_tolerance_m)
            global_id = node_lookup.get(key)
            if global_id is None:
                position = node["position_projected_m"]
                global_id = _stable_id("node", *key)
                node_lookup[key] = global_id
                nodes[global_id] = {
                    "id": global_id,
                    "transport_mode": node.get("transport_mode"),
                    "vertical_mode": node.get("vertical_mode"),
                    "layer_order": node.get("layer_order"),
                    "position_projected_m": [
                        float(position[0]),
                        float(position[1]),
                        position[2],
                    ],
                    "source_nodes": [],
                    "boundary_port_keys": [],
                }
            local_to_global[str(node["id"])] = global_id
            nodes[global_id]["source_nodes"].append(str(node["id"]))
            port_key = node.get("boundary_port_key")
            if port_key and port_key not in nodes[global_id]["boundary_port_keys"]:
                nodes[global_id]["boundary_port_keys"].append(port_key)

        for edge in graph.get("edges", []):
            local_coordinates = edge.get("geometry_local_m", [])
            coordinates = [
                [
                    origin[0] + float(point[0]),
                    origin[1] + float(point[1]),
                    point[2] if len(point) >= 3 else None,
                ]
                for point in local_coordinates
            ]
            if len(coordinates) < 2:
                continue
            edge_key = (
                edge.get("transport_mode"),
                edge.get("vertical_mode"),
                edge.get("class"),
                _canonical_geometry_key(coordinates),
            )
            if edge_key in edge_keys:
                continue
            edge_keys.add(edge_key)
            from_node = local_to_global.get(str(edge.get("from_node")))
            to_node = local_to_global.get(str(edge.get("to_node")))
            if from_node is None or to_node is None:
                continue
            edges.append(
                {
                    "id": _stable_id("edge", *edge_key),
                    "from_node": from_node,
                    "to_node": to_node,
                    "transport_mode": edge.get("transport_mode"),
                    "class": edge.get("class"),
                    "vertical_mode": edge.get("vertical_mode"),
                    "layer_order": edge.get("layer_order"),
                    "width_m": float(edge.get("width_m", 5.0)),
                    "length_m": float(LineString([point[:2] for point in coordinates]).length),
                    "geometry_projected_m": coordinates,
                    "source_tiles": [tile_info.get("tile_id")],
                    "source_id": edge.get("source_id"),
                }
            )

        for feature in payload.get("water", []):
            geometry_payload = feature.get("geometry")
            if geometry_payload:
                water_geometries.append(
                    _translate_geometry_payload(geometry_payload, origin[0], origin[1])
                )

        for building in payload.get("building_solids", []):
            footprint = _translate_geometry_payload(
                building["footprint_local_m"],
                origin[0],
                origin[1],
            )
            if footprint.is_empty:
                continue
            centroid = footprint.centroid
            key = (
                building.get("source_id"),
                round(centroid.x, 2),
                round(centroid.y, 2),
                round(footprint.area, 1),
            )
            if key in building_keys:
                continue
            building_keys.add(key)
            buildings.append(
                {
                    "id": _stable_id("building", *key),
                    "source_id": building.get("source_id"),
                    "building_type": building.get("building_type"),
                    "base_z_m": float(building.get("base_z_m", 0.0)),
                    "height_m": float(building.get("height_m", 9.3)),
                    "height_source": building.get("height_source"),
                    "footprint_projected_m": mapping(footprint),
                }
            )

    scene_origin = (min(value[0] for value in origins), min(value[1] for value in origins))
    coverage = unary_union(tile_coverage)
    surface_roads = [
        (
            LineString([point[:2] for point in edge["geometry_projected_m"]]),
            edge["width_m"],
        )
        for edge in edges
        if edge["transport_mode"] == "road" and edge["vertical_mode"] == "surface"
    ]
    blocks, parcels, road_union = _derive_blocks_and_parcels(
        coverage,
        surface_roads,
        water_geometries,
        scene_origin=scene_origin,
        minimum_block_area_m2=minimum_block_area_m2,
        target_parcel_area_m2=target_parcel_area_m2,
        minimum_parcel_area_m2=minimum_parcel_area_m2,
    )

    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge["from_node"]] += 1
        degree[edge["to_node"]] += 1
    for node in nodes.values():
        position = node.pop("position_projected_m")
        node["position_local_m"] = [
            float(position[0]) - scene_origin[0],
            float(position[1]) - scene_origin[1],
            position[2],
        ]
        node["degree"] = degree[node["id"]]
        node["stitched_sources"] = len(node["source_nodes"])
        node["boundary_port_keys"].sort()
        node["source_nodes"].sort()

    for edge in edges:
        coordinates = edge.pop("geometry_projected_m")
        edge["geometry_local_m"] = [
            [
                float(point[0]) - scene_origin[0],
                float(point[1]) - scene_origin[1],
                point[2],
            ]
            for point in coordinates
        ]

    for building in buildings:
        footprint = shape(building.pop("footprint_projected_m"))
        local = affinity.translate(
            footprint,
            xoff=-scene_origin[0],
            yoff=-scene_origin[1],
        )
        building["footprint_local_m"] = mapping(local)

    local_coverage = affinity.translate(
        coverage,
        xoff=-scene_origin[0],
        yoff=-scene_origin[1],
    )
    local_roads = affinity.translate(
        road_union,
        xoff=-scene_origin[0],
        yoff=-scene_origin[1],
    )
    minx, miny, maxx, maxy = local_coverage.bounds
    result = {
        "format": "urban-network-scene",
        "version": NETWORK_SCENE_VERSION,
        "coordinate_system": {
            "units": "metres",
            "axis_convention": "x-east, y-north, z-up",
            "source_projected_crs": source_crs,
            "origin_projected_m": list(scene_origin),
            "local_bounds_m": [minx, miny, maxx, maxy],
        },
        "filters": {
            "city_id": city_id,
            "area_id": area_id,
            "max_tiles": max_tiles,
        },
        "tiles": tile_records,
        "transport_graph": {
            "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
            "edges": sorted(edges, key=lambda item: item["id"]),
        },
        "coverage_local_m": mapping(local_coverage),
        "surface_road_reservation_local_m": mapping(local_roads),
        "blocks": blocks,
        "parcels": parcels,
        "buildings": sorted(buildings, key=lambda item: item["id"]),
        "statistics": {
            "tiles": len(states),
            "nodes": len(nodes),
            "edges": len(edges),
            "stitched_nodes": sum(
                int(node["stitched_sources"] > 1) for node in nodes.values()
            ),
            "surface_road_edges": sum(
                edge["transport_mode"] == "road"
                and edge["vertical_mode"] == "surface"
                for edge in edges
            ),
            "grade_separated_edges": sum(
                edge["vertical_mode"] in {"underground", "elevated"}
                for edge in edges
            ),
            "blocks": len(blocks),
            "boundary_blocks": sum(block["touches_scene_boundary"] for block in blocks),
            "parcels": len(parcels),
            "buildings": len(buildings),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, result)
    result["output"] = str(output_json)
    return result


def _point_transform(
    x: float,
    y: float,
    *,
    bounds: tuple[float, float, float, float],
    scale: float,
    padding: int,
) -> tuple[int, int]:
    minx, _miny, _maxx, maxy = bounds
    return (
        int(round((x - minx) * scale + padding)),
        int(round((maxy - y) * scale + padding)),
    )


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: Polygon,
    *,
    bounds: tuple[float, float, float, float],
    scale: float,
    padding: int,
    fill: tuple[int, int, int] | None,
    outline: tuple[int, int, int] | None,
    width: int = 1,
) -> None:
    points = [
        _point_transform(x, y, bounds=bounds, scale=scale, padding=padding)
        for x, y in polygon.exterior.coords
    ]
    draw.polygon(points, fill=fill, outline=outline)
    if outline is not None and width > 1:
        draw.line(points, fill=outline, width=width, joint="curve")


def render_network_plan(
    network_path: str | Path,
    output_png: str | Path,
    *,
    maximum_size: int = 1800,
) -> dict[str, Any]:
    network_path = Path(network_path).expanduser().resolve()
    output_png = Path(output_png).expanduser().resolve()
    payload = json.loads(network_path.read_text(encoding="utf-8"))
    bounds = tuple(float(value) for value in payload["coordinate_system"]["local_bounds_m"])
    width_m = max(bounds[2] - bounds[0], 1.0)
    height_m = max(bounds[3] - bounds[1], 1.0)
    padding = 24
    scale = (maximum_size - padding * 2) / max(width_m, height_m)
    image_width = max(64, int(math.ceil(width_m * scale)) + padding * 2)
    image_height = max(64, int(math.ceil(height_m * scale)) + padding * 2)
    image = Image.new("RGB", (image_width, image_height), (12, 14, 17))
    draw = ImageDraw.Draw(image)

    for block in payload.get("blocks", []):
        for polygon in _iter_polygons(shape(block["geometry_local_m"])):
            _draw_polygon(
                draw,
                polygon,
                bounds=bounds,
                scale=scale,
                padding=padding,
                fill=(31, 40, 44),
                outline=(61, 75, 78),
            )
    for parcel in payload.get("parcels", []):
        for polygon in _iter_polygons(shape(parcel["geometry_local_m"])):
            _draw_polygon(
                draw,
                polygon,
                bounds=bounds,
                scale=scale,
                padding=padding,
                fill=None,
                outline=(91, 104, 104),
            )
    for building in payload.get("buildings", []):
        for polygon in _iter_polygons(shape(building["footprint_local_m"])):
            _draw_polygon(
                draw,
                polygon,
                bounds=bounds,
                scale=scale,
                padding=padding,
                fill=(151, 158, 163),
                outline=(185, 190, 194),
            )
    for edge in payload.get("transport_graph", {}).get("edges", []):
        if edge.get("transport_mode") != "road":
            continue
        coordinates = edge.get("geometry_local_m", [])
        points = [
            _point_transform(
                float(point[0]),
                float(point[1]),
                bounds=bounds,
                scale=scale,
                padding=padding,
            )
            for point in coordinates
        ]
        if len(points) < 2:
            continue
        colour = ROAD_COLOURS.get(str(edge.get("class")), (170, 170, 170))
        road_width = max(1, int(round(float(edge.get("width_m", 3.0)) * scale)))
        if edge.get("vertical_mode") == "underground":
            colour = tuple(max(0, value // 3) for value in colour)
        elif edge.get("vertical_mode") == "elevated":
            colour = tuple(min(255, int(value * 1.15)) for value in colour)
        draw.line(points, fill=colour, width=road_width, joint="curve")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png, optimize=True)
    return {
        "network": str(network_path),
        "preview": str(output_png),
        "size": [image_width, image_height],
    }


def export_network_scene_obj(
    network_path: str | Path,
    output_obj: str | Path,
) -> dict[str, Any]:
    network_path = Path(network_path).expanduser().resolve()
    output_obj = Path(output_obj).expanduser().resolve()
    if output_obj.suffix.lower() != ".obj":
        output_obj = output_obj.with_suffix(".obj")
    payload = json.loads(network_path.read_text(encoding="utf-8"))
    mesh = _ObjMesh()

    for block in payload.get("blocks", []):
        for polygon in _iter_polygons(shape(block["geometry_local_m"])):
            mesh.prism(
                polygon,
                bottom_z=-0.12,
                top_z=-0.08,
                group=str(block["id"]),
                material="ground_block",
            )
    for parcel in payload.get("parcels", []):
        geometry = shape(parcel["geometry_local_m"])
        outline = geometry.boundary.buffer(0.20, cap_style="flat", join_style="mitre")
        for polygon in _iter_polygons(outline):
            mesh.prism(
                polygon,
                bottom_z=-0.07,
                top_z=-0.03,
                group=str(parcel["id"]),
                material="parcel_line",
            )
    for building in payload.get("buildings", []):
        footprint = shape(building["footprint_local_m"])
        bottom = float(building.get("base_z_m", 0.0))
        top = bottom + float(building.get("height_m", 9.3))
        for polygon in _iter_polygons(footprint):
            mesh.prism(
                polygon,
                bottom_z=bottom,
                top_z=top,
                group=str(building["id"]),
                material="building",
            )
    for edge in payload.get("transport_graph", {}).get("edges", []):
        coordinates = edge.get("geometry_local_m", [])
        z_values = [
            float(point[2])
            for point in coordinates
            if len(point) >= 3 and point[2] is not None
        ]
        if len(coordinates) < 2 or not z_values:
            continue
        line = LineString([(float(point[0]), float(point[1])) for point in coordinates])
        if line.is_empty or line.length <= 0:
            continue
        ribbon = line.buffer(
            max(float(edge.get("width_m", 5.0)), 0.5) / 2.0,
            cap_style="flat",
            join_style="round",
        )
        z = sum(z_values) / len(z_values)
        material = f"{edge.get('transport_mode', 'road')}_{edge.get('vertical_mode', 'surface')}"
        for polygon in _iter_polygons(ribbon):
            mesh.prism(
                polygon,
                bottom_z=z - 0.10,
                top_z=z + 0.10,
                group=str(edge["id"]),
                material=material,
            )

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    material_path = output_obj.with_suffix(".mtl")
    _write_materials(material_path)
    mesh.write(output_obj, material_path.name)
    return {
        "network": str(network_path),
        "obj": str(output_obj),
        "material": str(material_path),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
    }


def build_network_scene(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    city_id: str | None = None,
    area_id: str | None = None,
    max_tiles: int | None = None,
    stitch_tolerance_m: float = 0.10,
    minimum_block_area_m2: float = 400.0,
    target_parcel_area_m2: float = 900.0,
    minimum_parcel_area_m2: float = 120.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    network_path = output_dir / "network.json"
    result = compile_network_scene(
        dataset_root,
        network_path,
        city_id=city_id,
        area_id=area_id,
        max_tiles=max_tiles,
        stitch_tolerance_m=stitch_tolerance_m,
        minimum_block_area_m2=minimum_block_area_m2,
        target_parcel_area_m2=target_parcel_area_m2,
        minimum_parcel_area_m2=minimum_parcel_area_m2,
    )
    preview = render_network_plan(network_path, output_dir / "plan.png")
    obj = export_network_scene_obj(network_path, output_dir / "city.obj")
    summary = {
        **result["statistics"],
        "network": str(network_path),
        "preview": preview["preview"],
        "obj": obj["obj"],
        "material": obj["material"],
    }
    write_json(output_dir / "summary.json", summary)
    return summary
