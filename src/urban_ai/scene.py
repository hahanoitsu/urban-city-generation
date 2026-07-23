from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import split, unary_union

from urban_dataset.obj_export import _ObjMesh, _iter_polygons, _write_materials

_ROAD_COLOURS = {
    "major": (225, 67, 52),
    "secondary": (235, 141, 63),
    "local": (244, 223, 143),
}
_RAIL_COLOURS = {
    "rail": (199, 125, 255),
    "subway": (120, 175, 255),
    "light_rail": (115, 223, 207),
    "tram": (94, 200, 143),
}


def _polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon | GeometryCollection):
        for part in geometry.geoms:
            yield from _polygons(part)


def _recursive_parcels(
    polygon: Polygon,
    *,
    target_area_m2: float,
    minimum_area_m2: float,
    depth: int = 0,
) -> list[Polygon]:
    if polygon.area <= target_area_m2 * 1.5 or depth >= 10:
        return [polygon]
    minx, miny, maxx, maxy = polygon.bounds
    width, height = maxx - minx, maxy - miny
    if min(width, height) < 12.0:
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
            )
        )
    return result


def _largest_polygon(geometry: BaseGeometry) -> Polygon | None:
    values = list(_polygons(geometry))
    return max(values, key=lambda value: value.area) if values else None


def compile_generated_city(
    city: dict[str, Any],
    *,
    seed: int = 5132,
    minimum_block_area_m2: float = 400.0,
    target_parcel_area_m2: float = 1200.0,
    minimum_parcel_area_m2: float = 160.0,
) -> dict[str, Any]:
    result = json.loads(json.dumps(city))
    bounds = [float(value) for value in result["coordinate_system"]["local_bounds"]]
    coverage = box(*bounds)
    graph = result.get("transport_graph", {})

    surface_reservations: list[BaseGeometry] = []
    for edge in graph.get("edges", []):
        if edge.get("vertical_mode") != "surface":
            continue
        coordinates = edge.get("geometry_local_m", [])
        if len(coordinates) < 2:
            continue
        line = LineString([(float(point[0]), float(point[1])) for point in coordinates])
        if line.length <= 1e-6:
            continue
        width = max(1.0, float(edge.get("width_m", 5.0)))
        surface_reservations.append(
            line.buffer(width / 2.0, cap_style="flat", join_style="round")
        )
    reservation = unary_union(surface_reservations) if surface_reservations else GeometryCollection()
    remaining = coverage.difference(reservation)

    blocks: list[dict[str, Any]] = []
    parcels: list[dict[str, Any]] = []
    for block_index, polygon in enumerate(_polygons(remaining)):
        if polygon.area < minimum_block_area_m2:
            continue
        block_id = f"block_{block_index:04d}"
        blocks.append(
            {
                "id": block_id,
                "area_m2": float(polygon.area),
                "geometry_local_m": mapping(polygon),
                "touches_city_boundary": bool(polygon.intersects(coverage.boundary.buffer(0.1))),
            }
        )
        for parcel in _recursive_parcels(
            polygon,
            target_area_m2=target_parcel_area_m2,
            minimum_area_m2=minimum_parcel_area_m2,
        ):
            if parcel.area < minimum_parcel_area_m2:
                continue
            parcels.append(
                {
                    "id": f"parcel_{len(parcels):05d}",
                    "block_id": block_id,
                    "area_m2": float(parcel.area),
                    "geometry_local_m": mapping(parcel),
                    "touches_transport": bool(parcel.boundary.distance(reservation) <= 1.0),
                }
            )

    style = result.get("generation", {}).get("style", {})
    target_coverage = max(0.12, min(0.70, float(style.get("building_coverage", 0.28))))
    base_height = max(6.0, min(180.0, float(style.get("mean_building_height_m", 18.0))))
    rng = random.Random(int(seed))
    buildings: list[dict[str, Any]] = []
    for parcel in parcels:
        parcel_geometry = shape(parcel["geometry_local_m"])
        if parcel_geometry.area < 220.0:
            continue
        scale = math.sqrt(target_coverage)
        scaled = affinity.scale(parcel_geometry, xfact=scale, yfact=scale, origin="centroid")
        footprint = _largest_polygon(scaled.intersection(parcel_geometry.buffer(-1.5)))
        if footprint is None or footprint.area < 80.0:
            continue
        height = base_height * rng.uniform(0.65, 1.35)
        buildings.append(
            {
                "id": f"building_{len(buildings):05d}",
                "parcel_id": parcel["id"],
                "footprint_local_m": mapping(footprint),
                "base_z_m": 0.0,
                "height_m": round(height, 2),
                "height_source": "procedural_style_baseline",
            }
        )

    result["coverage_local_m"] = mapping(coverage)
    result["surface_transport_reservation_local_m"] = mapping(reservation)
    result["blocks"] = blocks
    result["parcels"] = parcels
    result["building_solids"] = buildings
    statistics = result.setdefault("statistics", {})
    statistics.update(
        {
            "blocks": len(blocks),
            "parcels": len(parcels),
            "buildings": len(buildings),
            "surface_reservation_area_m2": float(reservation.area),
        }
    )
    return result


def render_generated_city(
    city: dict[str, Any] | str | Path,
    output_path: str | Path,
    *,
    maximum_size: int = 1400,
) -> dict[str, Any]:
    if not isinstance(city, dict):
        city = json.loads(Path(city).read_text(encoding="utf-8"))
    output_path = Path(output_path).expanduser().resolve()
    bounds = [float(value) for value in city["coordinate_system"]["local_bounds"]]
    width_m = max(bounds[2] - bounds[0], 1.0)
    height_m = max(bounds[3] - bounds[1], 1.0)
    padding = 24
    scale = (maximum_size - 2 * padding) / max(width_m, height_m)
    width = max(64, int(math.ceil(width_m * scale)) + 2 * padding)
    height = max(64, int(math.ceil(height_m * scale)) + 2 * padding)
    image = Image.new("RGB", (width, height), (9, 11, 14))
    draw = ImageDraw.Draw(image)

    def point(x: float, y: float) -> tuple[int, int]:
        return (
            int(round((x - bounds[0]) * scale + padding)),
            int(round((bounds[3] - y) * scale + padding)),
        )

    def polygon_points(polygon: Polygon) -> list[tuple[int, int]]:
        return [point(float(x), float(y)) for x, y in polygon.exterior.coords]

    for block in city.get("blocks", []):
        for polygon in _polygons(shape(block["geometry_local_m"])):
            draw.polygon(polygon_points(polygon), fill=(28, 37, 41), outline=(56, 68, 72))
    for parcel in city.get("parcels", []):
        for polygon in _polygons(shape(parcel["geometry_local_m"])):
            draw.line(polygon_points(polygon), fill=(76, 88, 91), width=1)
    for building in city.get("building_solids", []):
        for polygon in _polygons(shape(building["footprint_local_m"])):
            draw.polygon(
                polygon_points(polygon), fill=(145, 153, 160), outline=(190, 195, 200)
            )

    edges = city.get("transport_graph", {}).get("edges", [])
    order = {"underground": 0, "surface": 1, "elevated": 2, "unknown": 3}
    for edge in sorted(edges, key=lambda item: order.get(str(item.get("vertical_mode")), 3)):
        coordinates = edge.get("geometry_local_m", [])
        points = [point(float(value[0]), float(value[1])) for value in coordinates]
        if len(points) < 2:
            continue
        mode = edge.get("transport_mode")
        edge_class = str(edge.get("class"))
        colour = (
            _ROAD_COLOURS.get(edge_class, (190, 190, 190))
            if mode == "road"
            else _RAIL_COLOURS.get(edge_class, (150, 150, 230))
        )
        vertical = edge.get("vertical_mode")
        if vertical == "underground":
            colour = tuple(max(20, value // 3) for value in colour)
        elif vertical == "elevated":
            colour = tuple(min(255, int(value * 1.15)) for value in colour)
        line_width = max(1, int(round(float(edge.get("width_m", 4.0)) * scale)))
        draw.line(points, fill=colour, width=line_width, joint="curve")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)
    return {"preview": str(output_path), "size": [width, height], "edges": len(edges)}


def export_generated_city_obj(city: dict[str, Any] | str | Path, output_obj: str | Path) -> dict[str, Any]:
    if not isinstance(city, dict):
        city = json.loads(Path(city).read_text(encoding="utf-8"))
    output_obj = Path(output_obj).expanduser().resolve()
    if output_obj.suffix.lower() != ".obj":
        output_obj = output_obj.with_suffix(".obj")
    mesh = _ObjMesh()

    for block in city.get("blocks", []):
        for polygon in _iter_polygons(shape(block["geometry_local_m"])):
            mesh.prism(
                polygon,
                bottom_z=-0.12,
                top_z=-0.08,
                group=str(block["id"]),
                material="ground_block",
            )
    for building in city.get("building_solids", []):
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
    skipped = 0
    for edge in city.get("transport_graph", {}).get("edges", []):
        coordinates = edge.get("geometry_local_m", [])
        z_values = [
            float(point[2])
            for point in coordinates
            if len(point) >= 3 and point[2] is not None
        ]
        if len(coordinates) < 2 or not z_values:
            skipped += 1
            continue
        line = LineString([(float(point[0]), float(point[1])) for point in coordinates])
        if line.length <= 1e-6:
            continue
        width = max(0.5, float(edge.get("width_m", 5.0)))
        ribbon = line.buffer(width / 2.0, cap_style="flat", join_style="round")
        material = f"{edge.get('transport_mode', 'road')}_{edge.get('vertical_mode', 'surface')}"
        z = sum(z_values) / len(z_values)
        for polygon in _iter_polygons(ribbon):
            mesh.prism(
                polygon,
                bottom_z=z - 0.10,
                top_z=z + 0.10,
                group=str(edge.get("id", "transport")),
                material=material,
            )

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    material_path = output_obj.with_suffix(".mtl")
    _write_materials(material_path)
    mesh.write(output_obj, material_path.name)
    return {
        "obj": str(output_obj),
        "material": str(material_path),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "skipped_unknown_vertical_edges": skipped,
    }
