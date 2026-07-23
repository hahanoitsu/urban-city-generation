from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import triangulate


_MATERIALS: dict[str, tuple[float, float, float]] = {
    "building": (0.68, 0.68, 0.72),
    "road_surface": (0.16, 0.16, 0.18),
    "road_elevated": (0.24, 0.24, 0.27),
    "road_underground": (0.34, 0.34, 0.38),
    "rail_surface": (0.65, 0.52, 0.12),
    "rail_elevated": (0.76, 0.62, 0.18),
    "rail_underground": (0.48, 0.38, 0.10),
}


def _iter_polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon | GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_polygons(part)


@dataclass
class _ObjMesh:
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    vertex_lookup: dict[tuple[float, float, float], int] = field(default_factory=dict)
    faces: list[tuple[str, str, tuple[int, ...]]] = field(default_factory=list)

    def vertex(self, x: float, y: float, z: float) -> int:
        key = (round(float(x), 6), round(float(y), 6), round(float(z), 6))
        existing = self.vertex_lookup.get(key)
        if existing is not None:
            return existing
        self.vertices.append(key)
        index = len(self.vertices)
        self.vertex_lookup[key] = index
        return index

    def face(self, group: str, material: str, points: Iterable[tuple[float, float, float]]) -> None:
        indexes = tuple(self.vertex(*point) for point in points)
        if len(set(indexes)) >= 3:
            self.faces.append((group, material, indexes))

    def prism(
        self,
        polygon: Polygon,
        *,
        bottom_z: float,
        top_z: float,
        group: str,
        material: str,
    ) -> None:
        triangles = [
            triangle
            for triangle in triangulate(polygon)
            if polygon.covers(triangle.representative_point())
        ]
        for triangle in triangles:
            coordinates = list(triangle.exterior.coords)[:-1]
            self.face(group, material, [(x, y, top_z) for x, y in coordinates])
            self.face(group, material, [(x, y, bottom_z) for x, y in reversed(coordinates)])

        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            coordinates = list(ring.coords)
            for start, end in zip(coordinates, coordinates[1:]):
                self.face(
                    group,
                    material,
                    [
                        (start[0], start[1], bottom_z),
                        (end[0], end[1], bottom_z),
                        (end[0], end[1], top_z),
                        (start[0], start[1], top_z),
                    ],
                )

    def write(self, obj_path: Path, mtl_name: str) -> None:
        lines = [f"mtllib {mtl_name}"]
        lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in self.vertices)
        active_group: str | None = None
        active_material: str | None = None
        for group, material, indexes in self.faces:
            if group != active_group:
                lines.append(f"g {group}")
                active_group = group
            if material != active_material:
                lines.append(f"usemtl {material}")
                active_material = material
            lines.append("f " + " ".join(str(index) for index in indexes))
        obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_materials(path: Path) -> None:
    lines: list[str] = []
    for name, colour in _MATERIALS.items():
        lines.extend(
            [
                f"newmtl {name}",
                f"Kd {colour[0]:.4f} {colour[1]:.4f} {colour[2]:.4f}",
                "Ka 0.0500 0.0500 0.0500",
                "Ks 0.1000 0.1000 0.1000",
                "Ns 8.0",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_city_state_obj(
    state_path: str | Path,
    output_obj: str | Path,
) -> dict[str, Any]:
    state_path = Path(state_path).expanduser().resolve()
    output_obj = Path(output_obj).expanduser().resolve()
    if output_obj.suffix.lower() != ".obj":
        output_obj = output_obj.with_suffix(".obj")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if payload.get("format") != "urban-city-state-tile":
        raise ValueError("Expected an urban-city-state-tile JSON file")

    mesh = _ObjMesh()
    building_count = 0
    edge_count = 0
    skipped_unknown_edges = 0

    for building in payload.get("building_solids", []):
        footprint = shape(building["footprint_local_m"])
        bottom = float(building.get("base_z_m", 0.0))
        top = bottom + float(building.get("height_m", 9.3))
        for polygon in _iter_polygons(footprint):
            mesh.prism(
                polygon,
                bottom_z=bottom,
                top_z=top,
                group=str(building.get("id", "building")),
                material="building",
            )
            building_count += 1

    graph = payload.get("transport_graph", {})
    for edge in graph.get("edges", []):
        coordinates = edge.get("geometry_local_m", [])
        if len(coordinates) < 2:
            continue
        z_values = [point[2] for point in coordinates if len(point) >= 3 and point[2] is not None]
        if not z_values:
            skipped_unknown_edges += 1
            continue
        z = float(sum(z_values) / len(z_values))
        line = LineString([(float(point[0]), float(point[1])) for point in coordinates])
        if line.is_empty or line.length <= 1e-6:
            continue
        width = max(0.5, float(edge.get("width_m", 5.0)))
        ribbon = line.buffer(width / 2.0, cap_style="flat", join_style="round")
        material = f"{edge.get('transport_mode', 'road')}_{edge.get('vertical_mode', 'surface')}"
        if material not in _MATERIALS:
            skipped_unknown_edges += 1
            continue
        for polygon in _iter_polygons(ribbon):
            mesh.prism(
                polygon,
                bottom_z=z - 0.10,
                top_z=z + 0.10,
                group=str(edge.get("id", "transport")),
                material=material,
            )
            edge_count += 1

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    material_path = output_obj.with_suffix(".mtl")
    _write_materials(material_path)
    mesh.write(output_obj, material_path.name)
    return {
        "state": str(state_path),
        "obj": str(output_obj),
        "material": str(material_path),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "building_parts": building_count,
        "transport_ribbons": edge_count,
        "skipped_unknown_vertical_edges": skipped_unknown_edges,
    }
