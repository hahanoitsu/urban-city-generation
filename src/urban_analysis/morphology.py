from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.ndimage import convolve, label
from skimage.morphology import skeletonize

from urban_dataset.schema import CHANNEL_NAMES

CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNEL_NAMES)}

MORPHOLOGY_FEATURES = (
    "water_coverage",
    "green_coverage",
    "landuse_residential_coverage",
    "landuse_commercial_mixed_coverage",
    "landuse_industrial_coverage",
    "landuse_civic_coverage",
    "building_coverage",
    "building_density_per_km2",
    "building_height_mean_m",
    "building_height_area_mean_m",
    "building_height_area_median_m",
    "building_height_area_std_m",
    "building_height_area_p90_m",
    "road_length_km_per_km2",
    "road_major_share",
    "road_secondary_share",
    "road_local_share",
    "road_junction_density_per_km2",
    "road_dead_end_density_per_km2",
    "road_junctions_per_km",
    "road_dead_ends_per_km",
    "road_components",
    "road_largest_component_fraction",
    "road_elevated_length_km_per_km2",
    "road_underground_length_km_per_km2",
    "road_vertical_length_share",
    "rail_present",
    "rail_length_km_per_km2",
    "rail_components",
    "rail_largest_component_fraction",
    "rail_boundary_sides",
    "rail_surface_share",
    "rail_underground_share",
    "rail_elevated_share",
    "rail_vertical_length_share",
    "road_underground_mean_depth_m",
    "road_elevated_mean_height_m",
    "rail_underground_mean_depth_m",
    "rail_elevated_mean_height_m",
)

# Features used to discover a controllable morphology space. Connectivity and
# physical-validity measures stay in MORPHOLOGY_FEATURES for evaluation but are
# deliberately excluded here: they describe whether a city works, not what kind
# of urban form it has.
MORPHOLOGY_CONTROL_FEATURES = (
    "water_coverage",
    "green_coverage",
    "landuse_residential_coverage",
    "landuse_commercial_mixed_coverage",
    "landuse_industrial_coverage",
    "landuse_civic_coverage",
    "building_coverage",
    "building_density_per_km2",
    "building_height_mean_m",
    "road_length_km_per_km2",
    "road_major_share",
    "road_local_share",
    "road_junctions_per_km",
    "road_dead_ends_per_km",
    "road_vertical_length_share",
    "rail_length_km_per_km2",
    "rail_underground_share",
    "rail_elevated_share",
)


def _coverage(mask: np.ndarray, valid: np.ndarray) -> float:
    denominator = int(valid.sum())
    if denominator == 0:
        return 0.0
    return float((mask & valid).sum() / denominator)


def _skeleton(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return skeletonize(np.asarray(mask, dtype=bool) & valid)


def _skeleton_length_m(skeleton: np.ndarray, metres_per_pixel: float) -> float:
    skeleton = np.asarray(skeleton, dtype=bool)
    height, width = skeleton.shape
    total = 0.0
    root_two = 2.0**0.5
    for dr, dc, scale in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, root_two), (1, -1, root_two)):
        r0 = max(0, -dr)
        r1 = min(height, height - dr)
        c0 = max(0, -dc)
        c1 = min(width, width - dc)
        first = skeleton[r0:r1, c0:c1]
        second = skeleton[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        total += float((first & second).sum()) * metres_per_pixel * scale
    return total


def _component_stats(skeleton: np.ndarray, minimum_pixels: int = 4) -> tuple[int, float]:
    components, count = label(
        np.asarray(skeleton, dtype=bool),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if count == 0:
        return 0, 0.0
    sizes = np.bincount(components.ravel())[1:]
    sizes = sizes[sizes >= minimum_pixels]
    if sizes.size == 0:
        return 0, 0.0
    total = int(sizes.sum())
    return int(sizes.size), float(sizes.max() / total)


def _node_stats(skeleton: np.ndarray) -> tuple[int, int]:
    skeleton = np.asarray(skeleton, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    degree = convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)

    junction_pixels = skeleton & (degree >= 3)
    _, junctions = label(junction_pixels, structure=np.ones((3, 3), dtype=np.uint8))

    endpoints = skeleton & (degree == 1)
    if endpoints.shape[0] > 2 and endpoints.shape[1] > 2:
        interior = np.zeros_like(endpoints)
        interior[1:-1, 1:-1] = True
        endpoints &= interior
    _, dead_ends = label(endpoints, structure=np.ones((3, 3), dtype=np.uint8))
    return int(junctions), int(dead_ends)


def _boundary_sides(skeleton: np.ndarray) -> int:
    skeleton = np.asarray(skeleton, dtype=bool)
    if not skeleton.any():
        return 0
    return sum(
        int(bool(value))
        for value in (
            skeleton[0].any(),
            skeleton[-1].any(),
            skeleton[:, 0].any(),
            skeleton[:, -1].any(),
        )
    )


def _mean_profile(
    profile: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray | None,
) -> float:
    active = np.asarray(mask, dtype=bool)
    if confidence is not None:
        active &= np.asarray(confidence) > 0
    values = np.abs(np.asarray(profile, dtype=np.float32)[active])
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def describe_tile(
    row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    layers = np.asarray(arrays["layers"], dtype=np.float32)
    if layers.shape[0] != len(CHANNEL_NAMES):
        raise ValueError(
            f"Expected {len(CHANNEL_NAMES)} source channels, found {layers.shape[0]}"
        )

    valid = np.asarray(arrays["valid_data_mask"]) > 0
    if not valid.any():
        raise ValueError(f"Tile {row.get('tile_id', '<unknown>')} contains no valid pixels")

    height, width = valid.shape
    tile_size_m = _number(metadata.get("tile_size_m"))
    if tile_size_m <= 0:
        tile_size_m = _number(row.get("maxx")) - _number(row.get("minx"))
    if tile_size_m <= 0:
        raise ValueError("Tile size is missing from both metadata and manifest")

    metres_per_pixel = _number(metadata.get("metres_per_pixel"))
    if metres_per_pixel <= 0:
        metres_per_pixel = tile_size_m / width

    valid_area_km2 = float(valid.sum()) * metres_per_pixel**2 / 1_000_000.0
    if valid_area_km2 <= 0:
        raise ValueError("Valid tile area is zero")

    water = layers[CHANNEL_INDEX["water"]] > 0
    green = layers[CHANNEL_INDEX["green"]] > 0
    building = layers[CHANNEL_INDEX["building_footprint"]] > 0

    max_height_m = _number(
        metadata.get("height", {}).get("normalization_max_m"),
        180.0,
    )
    height_m = (
        layers[CHANNEL_INDEX["building_height_normalized"]].clip(0.0, 1.0)
        * max_height_m
    )
    building_heights = height_m[building & valid]

    road_centerlines = np.asarray(arrays["road_centerlines"]) > 0
    if road_centerlines.shape[0] != 3:
        raise ValueError("road_centerlines must contain major/secondary/local channels")

    class_skeletons = [_skeleton(channel, valid) for channel in road_centerlines]
    road_surface = _skeleton(road_centerlines.any(axis=0), valid)
    road_class_lengths = [
        _skeleton_length_m(skeleton, metres_per_pixel) for skeleton in class_skeletons
    ]
    road_surface_length = _skeleton_length_m(road_surface, metres_per_pixel)
    road_surface_length_km = road_surface_length / 1000.0
    road_components, road_largest = _component_stats(road_surface)
    road_junctions, road_dead_ends = _node_stats(road_surface)

    road_vertical = np.asarray(arrays["road_vertical_masks"]) > 0
    rail_vertical = np.asarray(arrays["rail_vertical_masks"]) > 0
    if road_vertical.shape[0] < 3 or rail_vertical.shape[0] < 3:
        raise ValueError("Vertical transport masks must contain surface/underground/elevated modes")

    road_underground = _skeleton(road_vertical[1], valid)
    road_elevated = _skeleton(road_vertical[2], valid)
    road_underground_length = _skeleton_length_m(road_underground, metres_per_pixel)
    road_elevated_length = _skeleton_length_m(road_elevated, metres_per_pixel)
    road_all_length = road_surface_length + road_underground_length + road_elevated_length

    rail_mode_skeletons = [_skeleton(rail_vertical[index], valid) for index in range(3)]
    rail_mode_lengths = [
        _skeleton_length_m(skeleton, metres_per_pixel) for skeleton in rail_mode_skeletons
    ]
    rail_union = _skeleton(rail_vertical[:3].any(axis=0), valid)
    rail_components, rail_largest = _component_stats(rail_union)
    rail_total_length = sum(rail_mode_lengths)

    road_profiles = np.asarray(
        arrays.get("road_vertical_profiles_m", np.zeros((3, height, width)))
    )
    rail_profiles = np.asarray(
        arrays.get("rail_vertical_profiles_m", np.zeros((3, height, width)))
    )
    road_confidence = arrays.get("road_vertical_profile_confidence")
    rail_confidence = arrays.get("rail_vertical_profile_confidence")

    building_count = int(_number(row.get("building_count"), 0.0))
    building_height_mean_m = _number(row.get("mean_building_height_m"))
    if building_height_mean_m <= 0 and building_heights.size:
        building_height_mean_m = float(building_heights.mean())

    road_class_total = sum(road_class_lengths)
    rail_present = bool(rail_union.any())

    return {
        "tile_id": str(row.get("tile_id", "")),
        "city_id": str(row.get("city_id", metadata.get("city_id", ""))),
        "area_id": str(row.get("area_id", metadata.get("area_id") or "")),
        "split": str(row.get("split", "")),
        "spatial_group": str(row.get("spatial_group", "")),
        "tile_size_m": float(tile_size_m),
        "metres_per_pixel": float(metres_per_pixel),
        "valid_area_km2": valid_area_km2,
        "valid_fraction": float(valid.mean()),
        "water_coverage": _coverage(water, valid),
        "green_coverage": _coverage(green, valid),
        "landuse_residential_coverage": _coverage(
            layers[CHANNEL_INDEX["landuse_residential"]] > 0, valid
        ),
        "landuse_commercial_mixed_coverage": _coverage(
            layers[CHANNEL_INDEX["landuse_commercial_mixed"]] > 0, valid
        ),
        "landuse_industrial_coverage": _coverage(
            layers[CHANNEL_INDEX["landuse_industrial"]] > 0, valid
        ),
        "landuse_civic_coverage": _coverage(
            layers[CHANNEL_INDEX["landuse_civic"]] > 0, valid
        ),
        "building_coverage": _coverage(building, valid),
        "building_count": building_count,
        "building_density_per_km2": float(building_count / valid_area_km2),
        "building_height_mean_m": building_height_mean_m,
        "building_height_area_mean_m": (
            float(building_heights.mean()) if building_heights.size else 0.0
        ),
        "building_height_area_median_m": (
            float(np.median(building_heights)) if building_heights.size else 0.0
        ),
        "building_height_area_std_m": (
            float(building_heights.std()) if building_heights.size else 0.0
        ),
        "building_height_area_p90_m": (
            float(np.quantile(building_heights, 0.9)) if building_heights.size else 0.0
        ),
        "road_length_km_per_km2": road_surface_length_km / valid_area_km2,
        "road_major_length_km_per_km2": road_class_lengths[0] / 1000.0 / valid_area_km2,
        "road_secondary_length_km_per_km2": (
            road_class_lengths[1] / 1000.0 / valid_area_km2
        ),
        "road_local_length_km_per_km2": road_class_lengths[2] / 1000.0 / valid_area_km2,
        "road_major_share": road_class_lengths[0] / road_class_total if road_class_total else 0.0,
        "road_secondary_share": (
            road_class_lengths[1] / road_class_total if road_class_total else 0.0
        ),
        "road_local_share": road_class_lengths[2] / road_class_total if road_class_total else 0.0,
        "road_junction_count": road_junctions,
        "road_junction_density_per_km2": road_junctions / valid_area_km2,
        "road_junctions_per_km": (
            road_junctions / road_surface_length_km if road_surface_length_km else 0.0
        ),
        "road_dead_end_count": road_dead_ends,
        "road_dead_end_density_per_km2": road_dead_ends / valid_area_km2,
        "road_dead_ends_per_km": (
            road_dead_ends / road_surface_length_km if road_surface_length_km else 0.0
        ),
        "road_components": road_components,
        "road_largest_component_fraction": road_largest,
        "road_boundary_sides": _boundary_sides(road_surface),
        "road_underground_length_km_per_km2": (
            road_underground_length / 1000.0 / valid_area_km2
        ),
        "road_elevated_length_km_per_km2": road_elevated_length / 1000.0 / valid_area_km2,
        "road_vertical_length_share": (
            (road_underground_length + road_elevated_length) / road_all_length
            if road_all_length
            else 0.0
        ),
        "rail_present": int(rail_present),
        "rail_length_km_per_km2": rail_total_length / 1000.0 / valid_area_km2,
        "rail_components": rail_components,
        "rail_largest_component_fraction": rail_largest,
        "rail_boundary_sides": _boundary_sides(rail_union),
        "rail_surface_length_km_per_km2": rail_mode_lengths[0] / 1000.0 / valid_area_km2,
        "rail_underground_length_km_per_km2": rail_mode_lengths[1] / 1000.0 / valid_area_km2,
        "rail_elevated_length_km_per_km2": rail_mode_lengths[2] / 1000.0 / valid_area_km2,
        "rail_surface_share": rail_mode_lengths[0] / rail_total_length if rail_total_length else 0.0,
        "rail_underground_share": (
            rail_mode_lengths[1] / rail_total_length if rail_total_length else 0.0
        ),
        "rail_elevated_share": (
            rail_mode_lengths[2] / rail_total_length if rail_total_length else 0.0
        ),
        "rail_vertical_length_share": (
            (rail_mode_lengths[1] + rail_mode_lengths[2]) / rail_total_length
            if rail_total_length
            else 0.0
        ),
        "road_underground_mean_depth_m": _mean_profile(
            road_profiles[1],
            road_vertical[1] & valid,
            None if road_confidence is None else np.asarray(road_confidence)[1],
        ),
        "road_elevated_mean_height_m": _mean_profile(
            road_profiles[2],
            road_vertical[2] & valid,
            None if road_confidence is None else np.asarray(road_confidence)[2],
        ),
        "rail_underground_mean_depth_m": _mean_profile(
            rail_profiles[1],
            rail_vertical[1] & valid,
            None if rail_confidence is None else np.asarray(rail_confidence)[1],
        ),
        "rail_elevated_mean_height_m": _mean_profile(
            rail_profiles[2],
            rail_vertical[2] & valid,
            None if rail_confidence is None else np.asarray(rail_confidence)[2],
        ),
    }
