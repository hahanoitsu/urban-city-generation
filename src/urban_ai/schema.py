from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from shapely.geometry import shape

GRAPH_PROGRAM_VERSION = "0.2.0"
STYLE_FIELDS = (
    "road_length_per_km2",
    "rail_length_per_km2",
    "major_fraction",
    "secondary_fraction",
    "local_fraction",
    "surface_fraction",
    "underground_fraction",
    "elevated_fraction",
    "intersection_density_per_km2",
    "mean_edge_length_m",
    "building_coverage",
    "mean_building_height_m",
    "water_coverage",
    "green_coverage",
)

VERTICAL_Z_M: dict[str, float | None] = {
    "surface": 0.0,
    "underground": -12.0,
    "elevated": 8.0,
    "unknown": None,
}
ROAD_CLASSES = {"major", "secondary", "local"}
RAIL_CLASSES = {"rail", "subway", "light_rail", "tram"}


@dataclass(frozen=True)
class ProgramConfig:
    coordinate_bins: int = 256
    width_quantum_m: float = 1.0
    maximum_width_m: float = 32.0
    simplify_tolerance_m: float = 6.0
    layer_min: int = -5
    layer_max: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalise_mode(value: Any) -> str:
    return "rail" if str(value).strip().lower() == "rail" else "road"


def normalise_class(value: Any, mode: str) -> str:
    text = str(value or "").strip().lower()
    if mode == "road":
        return text if text in ROAD_CLASSES else "local"
    return text if text in RAIL_CLASSES else "rail"


def normalise_vertical(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in VERTICAL_Z_M else "unknown"


def quantise(value: float, lower: float, upper: float, bins: int) -> int:
    if bins < 2:
        raise ValueError("coordinate_bins must be at least 2")
    if upper <= lower:
        return 0
    ratio = (float(value) - lower) / (upper - lower)
    return max(0, min(bins - 1, int(round(ratio * (bins - 1)))))


def dequantise(value: int, lower: float, upper: float, bins: int) -> float:
    if bins < 2 or upper <= lower:
        return float(lower)
    value = max(0, min(bins - 1, int(value)))
    return float(lower + value / (bins - 1) * (upper - lower))


def quantise_width(width_m: float, config: ProgramConfig) -> int:
    maximum_bin = max(1, int(round(config.maximum_width_m / config.width_quantum_m)))
    value = int(round(float(width_m) / config.width_quantum_m))
    return max(1, min(maximum_bin, value))


def dequantise_width(width_bin: int, config: ProgramConfig) -> float:
    return max(config.width_quantum_m, int(width_bin) * config.width_quantum_m)


def quantise_layer(layer: Any, config: ProgramConfig) -> int:
    if layer is None:
        layer = 0
    try:
        value = int(round(float(layer)))
    except (TypeError, ValueError):
        value = 0
    value = max(config.layer_min, min(config.layer_max, value))
    return value - config.layer_min


def dequantise_layer(layer_bin: int, config: ProgramConfig) -> int:
    value = int(layer_bin) + config.layer_min
    return max(config.layer_min, min(config.layer_max, value))


def _iter_feature_area(features: Iterable[dict[str, Any]]) -> float:
    total = 0.0
    for feature in features:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        try:
            total += float(shape(geometry).area)
        except Exception:
            continue
    return total


def city_style(payload: dict[str, Any]) -> dict[str, float]:
    bounds = payload.get("coordinate_system", {}).get(
        "local_bounds", [0.0, 0.0, 1024.0, 1024.0]
    )
    minx, miny, maxx, maxy = [float(value) for value in bounds]
    area_m2 = max((maxx - minx) * (maxy - miny), 1.0)
    area_km2 = area_m2 / 1_000_000.0
    graph = payload.get("transport_graph", {})
    edges = list(graph.get("edges", []))
    nodes = list(graph.get("nodes", []))

    mode_lengths: dict[str, float] = defaultdict(float)
    class_lengths: dict[str, float] = defaultdict(float)
    vertical_lengths: dict[str, float] = defaultdict(float)
    lengths: list[float] = []
    for edge in edges:
        length = max(0.0, float(edge.get("length_m", 0.0)))
        mode = normalise_mode(edge.get("transport_mode"))
        edge_class = normalise_class(edge.get("class"), mode)
        vertical = normalise_vertical(edge.get("vertical_mode"))
        mode_lengths[mode] += length
        class_lengths[edge_class] += length
        vertical_lengths[vertical] += length
        if length > 0:
            lengths.append(length)

    road_length = mode_lengths["road"]
    transport_length = sum(mode_lengths.values()) or 1.0
    intersections = sum(int(node.get("degree", 0)) >= 3 for node in nodes)
    building_area = 0.0
    heights: list[float] = []
    for building in payload.get("building_solids", []):
        geometry = building.get("footprint_local_m")
        if geometry:
            try:
                building_area += float(shape(geometry).area)
            except Exception:
                pass
        try:
            height = float(building.get("height_m", 0.0))
        except (TypeError, ValueError):
            height = 0.0
        if height > 0:
            heights.append(height)

    return {
        "road_length_per_km2": road_length / area_km2,
        "rail_length_per_km2": mode_lengths["rail"] / area_km2,
        "major_fraction": class_lengths["major"] / road_length if road_length else 0.0,
        "secondary_fraction": class_lengths["secondary"] / road_length if road_length else 0.0,
        "local_fraction": class_lengths["local"] / road_length if road_length else 0.0,
        "surface_fraction": vertical_lengths["surface"] / transport_length,
        "underground_fraction": vertical_lengths["underground"] / transport_length,
        "elevated_fraction": vertical_lengths["elevated"] / transport_length,
        "intersection_density_per_km2": intersections / area_km2,
        "mean_edge_length_m": sum(lengths) / len(lengths) if lengths else 0.0,
        "building_coverage": min(1.0, building_area / area_m2),
        "mean_building_height_m": sum(heights) / len(heights) if heights else 0.0,
        "water_coverage": min(1.0, _iter_feature_area(payload.get("water", [])) / area_m2),
        "green_coverage": min(1.0, _iter_feature_area(payload.get("green", [])) / area_m2),
    }


def style_vector(style: dict[str, Any]) -> list[float]:
    return [float(style.get(name, 0.0)) for name in STYLE_FIELDS]


def style_from_vector(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    if len(values) != len(STYLE_FIELDS):
        raise ValueError(f"Expected {len(STYLE_FIELDS)} style values, got {len(values)}")
    return {name: float(value) for name, value in zip(STYLE_FIELDS, values, strict=True)}
