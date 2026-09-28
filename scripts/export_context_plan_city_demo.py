from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape

from urban_ai.scene import compile_generated_city, export_generated_city_obj, render_generated_city


ROAD_COLOURS = {
    "major": (225, 67, 52),
    "secondary": (235, 141, 63),
    "local": (244, 223, 143),
}
RAIL_COLOURS = {
    "rail": (199, 125, 255),
    "subway": (120, 175, 255),
    "light_rail": (115, 223, 207),
    "tram": (94, 200, 143),
    "monorail": (232, 150, 255),
}
DISPLAY_Z = {
    "surface": 0.05,
    "elevated": 8.0,
    "underground": -8.0,
}


def polygons(value: Any) -> Iterable[Polygon]:
    geometry = value if hasattr(value, "geom_type") else shape(value)
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon | GeometryCollection):
        for part in geometry.geoms:
            yield from polygons(part)


def generation(record: dict[str, Any], decoder: str) -> dict[str, Any]:
    matches = [
        item
        for item in record.get("generations", [])
        if item.get("decoder") == decoder and item.get("seed") is None
    ]
    if not matches:
        raise ValueError(f"No deterministic {decoder} generation in {record.get('sample_id')}")
    return matches[0]


def transport_state(record: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    target_size = float(record.get("coordinate_system", {}).get("target_size_m", 512.0))
    graph = selected["graph"]
    nodes = []
    for node in graph.get("nodes", []):
        nodes.append(
            {
                "id": str(node["id"]),
                "transport_mode": node.get("mode", "road"),
                "vertical_mode": node.get("vertical_mode", "unknown"),
                "position_local_m": [
                    float(node["position_local_m"][0]),
                    float(node["position_local_m"][1]),
                    None,
                ],
            }
        )
    edges = []
    for edge in graph.get("edges", []):
        edges.append(
            {
                "id": str(edge["id"]),
                "from_node": str(edge["from_node"]),
                "to_node": str(edge["to_node"]),
                "transport_mode": edge.get("mode", "road"),
                "class": edge.get("class", "local"),
                "vertical_mode": edge.get("vertical_mode", "unknown"),
                "width_m": float(edge.get("width_m", 5.0)),
                "geometry_local_m": [
                    [float(point[0]), float(point[1]), None]
                    for point in edge.get("geometry_local_m", [])
                ],
            }
        )
    return {
        "format": "urban-city-state-demo",
        "version": "0.1.0",
        "coordinate_system": {
            "units": "metres",
            "local_bounds": [0.0, 0.0, target_size, target_size],
            "metric_elevation": "unavailable",
        },
        "source": {
            "sample_id": record.get("sample_id"),
            "split": record.get("split"),
            "overfit": bool(record.get("overfit")),
            "decoder": selected.get("decoder"),
            "sampling": selected.get("sampling"),
        },
        "transport_graph": {"nodes": nodes, "edges": edges},
    }


def scene_style(
    input_path: Path,
    building_coverage: float | None,
    mean_building_height_m: float | None,
) -> tuple[dict[str, float], str]:
    coverage = building_coverage
    height = mean_building_height_m
    source = "command defaults"
    root = input_path if input_path.is_dir() else input_path.parent
    experiment = root.parent / "training" / "experiment.json"
    if experiment.exists() and (coverage is None or height is None):
        payload = json.loads(experiment.read_text(encoding="utf-8"))
        normalization = payload.get("normalization", {})
        names = normalization.get("feature_names", [])
        means = normalization.get("feature_mean", [])
        features = dict(zip(names, means, strict=False))
        if coverage is None and "building_coverage" in features:
            coverage = float(features["building_coverage"])
        if height is None and "mean_building_height_m" in features:
            height = float(features["mean_building_height_m"])
        source = str(experiment)
    return {
        "building_coverage": 0.28 if coverage is None else float(coverage),
        "mean_building_height_m": 18.0 if height is None else float(height),
    }, source


def display_city(city: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(city)
    for edge in result.get("transport_graph", {}).get("edges", []):
        z = DISPLAY_Z.get(str(edge.get("vertical_mode")))
        edge["geometry_local_m"] = [
            [float(point[0]), float(point[1]), z]
            for point in edge.get("geometry_local_m", [])
        ]
    return result


def project_point(
    x: float,
    y: float,
    z: float,
    *,
    minimum_u: float,
    minimum_v: float,
    scale: float,
    padding: int,
) -> tuple[int, int]:
    u = (x - y) * (math.sqrt(3.0) / 2.0)
    v = (x + y) * 0.5 - z * 1.4
    return (
        int(round(padding + (u - minimum_u) * scale)),
        int(round(padding + (v - minimum_v) * scale)),
    )


def render_isometric_city(
    city: dict[str, Any],
    output_path: Path,
    *,
    width: int = 1600,
    height: int = 1100,
) -> None:
    bounds = [float(value) for value in city["coordinate_system"]["local_bounds"]]
    buildings = city.get("building_solids", [])
    projected = []
    for x, y in (
        (bounds[0], bounds[1]),
        (bounds[2], bounds[1]),
        (bounds[2], bounds[3]),
        (bounds[0], bounds[3]),
    ):
        projected.append(((x - y) * math.sqrt(3.0) / 2.0, (x + y) * 0.5))
    for building in buildings:
        top = float(building.get("base_z_m", 0.0)) + float(building.get("height_m", 9.3))
        for polygon in polygons(building["footprint_local_m"]):
            for x, y in polygon.exterior.coords:
                projected.append(
                    ((x - y) * math.sqrt(3.0) / 2.0, (x + y) * 0.5 - top * 1.4)
                )
    minimum_u = min(value[0] for value in projected)
    maximum_u = max(value[0] for value in projected)
    minimum_v = min(value[1] for value in projected)
    maximum_v = max(value[1] for value in projected)
    padding = 60
    scale = min(
        (width - 2 * padding) / max(maximum_u - minimum_u, 1.0),
        (height - 2 * padding) / max(maximum_v - minimum_v, 1.0),
    )

    def point(x: float, y: float, z: float = 0.0) -> tuple[int, int]:
        return project_point(
            x,
            y,
            z,
            minimum_u=minimum_u,
            minimum_v=minimum_v,
            scale=scale,
            padding=padding,
        )

    image = Image.new("RGB", (width, height), (8, 10, 13))
    draw = ImageDraw.Draw(image)
    ground = [
        point(bounds[0], bounds[1]),
        point(bounds[2], bounds[1]),
        point(bounds[2], bounds[3]),
        point(bounds[0], bounds[3]),
    ]
    draw.polygon(ground, fill=(18, 23, 26), outline=(65, 76, 80))
    for block in city.get("blocks", []):
        for polygon in polygons(block["geometry_local_m"]):
            draw.polygon(
                [point(float(x), float(y), 0.02) for x, y in polygon.exterior.coords],
                fill=(25, 32, 35),
                outline=(45, 55, 58),
            )

    edges = city.get("transport_graph", {}).get("edges", [])
    for edge in edges:
        if edge.get("vertical_mode") == "elevated":
            continue
        z = DISPLAY_Z.get(str(edge.get("vertical_mode")), 0.05)
        points = [point(float(value[0]), float(value[1]), z) for value in edge["geometry_local_m"]]
        if len(points) < 2:
            continue
        colours = ROAD_COLOURS if edge.get("transport_mode") == "road" else RAIL_COLOURS
        colour = colours.get(str(edge.get("class")), (190, 190, 190))
        if edge.get("vertical_mode") == "underground":
            colour = tuple(max(20, value // 3) for value in colour)
        line_width = max(1, int(round(float(edge.get("width_m", 5.0)) * scale * 0.75)))
        draw.line(points, fill=(18, 18, 20), width=line_width + 4, joint="curve")
        draw.line(points, fill=colour, width=line_width, joint="curve")

    ordered_buildings = sorted(
        buildings,
        key=lambda item: shape(item["footprint_local_m"]).centroid.x
        + shape(item["footprint_local_m"]).centroid.y,
    )
    for building in ordered_buildings:
        bottom_z = float(building.get("base_z_m", 0.0))
        top_z = bottom_z + float(building.get("height_m", 9.3))
        for polygon in polygons(building["footprint_local_m"]):
            coordinates = list(polygon.exterior.coords)[:-1]
            bottom = [point(float(x), float(y), bottom_z) for x, y in coordinates]
            top = [point(float(x), float(y), top_z) for x, y in coordinates]
            for index in range(len(coordinates)):
                following = (index + 1) % len(coordinates)
                dx = coordinates[following][0] - coordinates[index][0]
                dy = coordinates[following][1] - coordinates[index][1]
                shade = (96, 104, 112) if abs(dx) >= abs(dy) else (112, 120, 128)
                draw.polygon(
                    [bottom[index], bottom[following], top[following], top[index]],
                    fill=shade,
                )
            draw.polygon(top, fill=(160, 168, 175), outline=(200, 205, 210))

    for edge in edges:
        if edge.get("vertical_mode") != "elevated":
            continue
        points = [
            point(float(value[0]), float(value[1]), DISPLAY_Z["elevated"])
            for value in edge["geometry_local_m"]
        ]
        if len(points) < 2:
            continue
        colours = ROAD_COLOURS if edge.get("transport_mode") == "road" else RAIL_COLOURS
        colour = colours.get(str(edge.get("class")), (190, 190, 190))
        colour = tuple(min(255, int(value * 1.15)) for value in colour)
        line_width = max(1, int(round(float(edge.get("width_m", 5.0)) * scale * 0.75)))
        draw.line(points, fill=(18, 18, 20), width=line_width + 5, joint="curve")
        draw.line(points, fill=colour, width=line_width, joint="curve")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def export_one(
    input_path: Path,
    output: Path,
    *,
    decoder: str,
    style: dict[str, float],
    style_source: str,
    scene_seed: int,
) -> dict[str, Any]:
    record = json.loads(input_path.read_text(encoding="utf-8"))
    selected = generation(record, decoder)
    city = transport_state(record, selected)
    city["generation"] = {
        "kind": "context-plan-graph-v1",
        "style": style,
        "scene_seed": scene_seed,
    }
    city = compile_generated_city(city, seed=scene_seed)
    city["demo"] = {
        "learned": ["transport plan", "junction geometry", "edge classes", "edge widths"],
        "procedural_visualisation": ["blocks", "parcels", "building footprints", "building heights"],
        "metric_z": "unavailable from the learned model",
        "obj_vertical_offsets": "surface 0 m, elevated +8 m, underground -8 m for display only",
    }
    output.mkdir(parents=True, exist_ok=True)
    city_path = output / "city.json"
    city_path.write_text(json.dumps(city, indent=2) + "
", encoding="utf-8")
    render_generated_city(city, output / "plan.png")
    render_isometric_city(city, output / "city-isometric.png")
    obj = export_generated_city_obj(display_city(city), output / "city.obj")
    manifest = {
        "sample_id": record.get("sample_id"),
        "source_file": str(input_path),
        "transport": "deterministic model generation",
        "decoder": decoder,
        "overfit": bool(record.get("overfit")),
        "style": style,
        "style_source": style_source,
        "scene_seed": scene_seed,
        "generation_statistics": selected.get("statistics", {}),
        "scene_statistics": city.get("statistics", {}),
        "files": {
            "city": str(city_path),
            "plan": str(output / "plan.png"),
            "isometric": str(output / "city-isometric.png"),
            "obj": obj["obj"],
            "material": obj["material"],
        },
        "limitations": [
            "The current checkpoint is an overfit architecture-debugging run on 16 samples.",
            "Buildings and parcels in this demo are procedural visualisation, not model outputs.",
            "OBJ vertical offsets are display values because metric elevation is not learned yet.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "
", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder", choices=["learned_degree", "compatible"], default="learned_degree")
    parser.add_argument("--building-coverage", type=float)
    parser.add_argument("--mean-building-height-m", type=float)
    parser.add_argument("--scene-seed", type=int, default=5132)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    style, style_source = scene_style(
        input_path,
        args.building_coverage,
        args.mean_building_height_m,
    )
    files = (
        sorted(path for path in input_path.glob("*.json") if path.name != "summary.json")
        if input_path.is_dir()
        else [input_path]
    )
    if not files:
        raise ValueError(f"No sample JSON files found in {input_path}")
    results = []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        destination = output / str(record.get("sample_id")) if input_path.is_dir() else output
        result = export_one(
            path,
            destination,
            decoder=args.decoder,
            style=style,
            style_source=style_source,
            scene_seed=args.scene_seed,
        )
        results.append(result)
        print(f"Saved {result['sample_id']} to {destination}", flush=True)
    print(json.dumps({"samples": len(results), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
