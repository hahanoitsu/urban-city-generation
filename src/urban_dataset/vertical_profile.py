from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from .classify import clean_tag, first_number
from .config import BuildConfig
from .tile import TileSpec
from .vertical import vertical_mode_name

PROFILE_MODE_NAMES = ("surface", "underground", "elevated")
PROFILE_CONFIDENCE_NAMES = (
    "missing",
    "inferred_from_structure",
    "tag_derived",
    "measured",
)


@dataclass(frozen=True)
class VerticalProfileResult:
    offsets_m: np.ndarray
    confidence: np.ndarray
    evidence_counts: dict[str, int]


def _iter_lines(geometry: BaseGeometry | None) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
        return
    if isinstance(geometry, MultiLineString | GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_lines(part)


def _enabled(value: Any) -> bool:
    return clean_tag(value) not in {"", "no", "false", "0", "none"}


def _metres(value: Any) -> float | None:
    number = first_number(value)
    if number is None or not math.isfinite(number):
        return None
    return float(number)


def _incline_fraction(value: Any) -> float | None:
    text = clean_tag(value).replace(" ", "")
    if not text or text in {"up", "down", "yes", "no"}:
        return None
    try:
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        if text.endswith("°"):
            return math.tan(math.radians(float(text[:-1])))
        number = float(text)
    except ValueError:
        return None
    return number / 100.0 if abs(number) > 1.0 else number


def _mode_index(mode: str) -> int | None:
    try:
        return PROFILE_MODE_NAMES.index(mode)
    except ValueError:
        return None


def _target_offset(
    row: Any,
    *,
    transport_mode: str,
    config: BuildConfig,
) -> tuple[float, int, str]:
    mode = vertical_mode_name(row)
    profile = config.vertical_profiles
    layer = _metres(row.get("layer"))
    incline = _incline_fraction(row.get("incline"))

    if mode == "elevated":
        min_height = _metres(row.get("min_height"))
        if min_height is not None and 0.5 <= min_height <= 80.0:
            return min_height + profile.deck_thickness_m, 2, "min_height"
        height = _metres(row.get("height"))
        if height is not None and 1.0 <= height <= 80.0:
            return height, 2, "height"
        base = (
            profile.rail_default_elevated_m
            if transport_mode == "rail"
            else profile.road_default_elevated_m
        )
        if layer is not None and layer > 0:
            return base + max(0.0, layer - 1.0) * profile.layer_step_m, 1, "layer"
        return base, 1, "bridge_or_location"

    if mode == "underground":
        depth = _metres(row.get("depth"))
        if depth is not None and 1.0 <= depth <= 100.0:
            return -depth, 2, "depth"
        base = (
            profile.rail_default_tunnel_depth_m
            if transport_mode == "rail"
            else profile.road_default_tunnel_depth_m
        )
        if layer is not None and layer < 0:
            return -(base + max(0.0, abs(layer) - 1.0) * profile.layer_step_m), 1, "layer"
        return -base, 1, "tunnel_or_location"

    if mode == "surface":
        if _enabled(row.get("embankment")):
            return profile.embankment_offset_m, 1, "embankment"
        if _enabled(row.get("cutting")):
            return -profile.cutting_depth_m, 1, "cutting"
        if incline is not None and abs(incline) > 1e-6:
            return 0.0, 2, "incline"

    return 0.0, 0, "none"


def _on_tile_boundary(point: tuple[float, float], tile: TileSpec, tolerance: float) -> bool:
    x, y = point
    return (
        abs(x - tile.minx) <= tolerance
        or abs(x - tile.maxx) <= tolerance
        or abs(y - tile.miny) <= tolerance
        or abs(y - tile.maxy) <= tolerance
    )


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _profile_values(
    line: LineString,
    *,
    target_offset_m: float,
    incline: float | None,
    max_grade: float,
    sample_step_m: float,
    open_start: bool,
    open_end: bool,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    length = float(line.length)
    if length <= 1e-6:
        return np.zeros(0, dtype=np.float32), []

    count = max(2, int(math.ceil(length / max(sample_step_m, 0.25))) + 1)
    distances = np.linspace(0.0, length, count, dtype=np.float64)
    points = [line.interpolate(float(distance)).coords[0][:2] for distance in distances]

    if abs(target_offset_m) <= 1e-6:
        if incline is None:
            return np.zeros(count, dtype=np.float32), points
        values = incline * (distances - length / 2.0)
        return values.astype(np.float32), points

    ramp_length = min(length / 2.0, abs(target_offset_m) / max(max_grade, 1e-4))
    start_gate = np.ones_like(distances) if open_start else _smoothstep(distances / ramp_length)
    end_gate = (
        np.ones_like(distances)
        if open_end
        else _smoothstep((length - distances) / ramp_length)
    )
    gate = np.minimum(start_gate, end_gate)
    values = target_offset_m * gate
    if incline is not None:
        values = values + incline * (distances - length / 2.0) * gate
    return values.astype(np.float32), points


def _feature_width(row: Any, transport_mode: str, config: BuildConfig) -> float:
    if transport_mode == "rail":
        return 6.0
    value = _metres(row.get("estimated_width_m"))
    if value is not None and value > 0:
        return float(np.clip(value, config.roads.minimum_width_m, config.roads.maximum_width_m))
    road_class = clean_tag(row.get("road_class")) or "local"
    return float(config.roads.widths_m.get(road_class, config.roads.widths_m["local"]))


def build_vertical_profiles(
    frame: gpd.GeoDataFrame,
    *,
    transport_mode: str,
    tile: TileSpec,
    config: BuildConfig,
    pixels: int,
    transform,
) -> VerticalProfileResult:
    offsets = np.zeros((len(PROFILE_MODE_NAMES), pixels, pixels), dtype=np.float32)
    confidence = np.zeros((len(PROFILE_MODE_NAMES), pixels, pixels), dtype=np.uint8)
    evidence_counts: dict[str, int] = {}
    if frame.empty:
        return VerticalProfileResult(offsets, confidence, evidence_counts)

    boundary_tolerance = max(0.5, config.raster.tile_size_m / pixels)
    max_grade = (
        config.vertical_profiles.rail_max_grade
        if transport_mode == "rail"
        else config.vertical_profiles.road_max_grade
    )

    for _index, row in frame.iterrows():
        mode = vertical_mode_name(row)
        mode_index = _mode_index(mode)
        if mode_index is None:
            continue
        target_offset, level, evidence = _target_offset(
            row,
            transport_mode=transport_mode,
            config=config,
        )
        incline = _incline_fraction(row.get("incline"))
        if level <= 0 and incline is None:
            continue

        width = _feature_width(row, transport_mode, config)
        geometries: list[tuple[BaseGeometry, float]] = []
        for line in _iter_lines(row.geometry):
            start = tuple(line.coords[0][:2])
            end = tuple(line.coords[-1][:2])
            values, points = _profile_values(
                line,
                target_offset_m=target_offset,
                incline=incline,
                max_grade=max_grade,
                sample_step_m=config.vertical_profiles.sample_step_m,
                open_start=_on_tile_boundary(start, tile, boundary_tolerance),
                open_end=_on_tile_boundary(end, tile, boundary_tolerance),
            )
            for point_a, point_b, value_a, value_b in zip(
                points[:-1],
                points[1:],
                values[:-1],
                values[1:],
                strict=True,
            ):
                segment = LineString([point_a, point_b])
                if segment.length <= 1e-6:
                    continue
                geometry = segment.buffer(width / 2.0, cap_style="flat", join_style="round")
                geometries.append((geometry, float((value_a + value_b) / 2.0)))

        if not geometries:
            continue
        profile_raster = rasterize(
            geometries,
            out_shape=(pixels, pixels),
            transform=transform,
            fill=0.0,
            all_touched=True,
            dtype="float32",
        )
        confidence_raster = rasterize(
            [(geometry, level) for geometry, _value in geometries],
            out_shape=(pixels, pixels),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        replace = (confidence_raster > confidence[mode_index]) | (
            (confidence_raster == confidence[mode_index])
            & (np.abs(profile_raster) > np.abs(offsets[mode_index]))
        )
        offsets[mode_index][replace] = profile_raster[replace]
        confidence[mode_index][replace] = confidence_raster[replace]
        evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1

    return VerticalProfileResult(offsets, confidence, evidence_counts)
