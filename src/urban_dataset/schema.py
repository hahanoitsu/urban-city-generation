from __future__ import annotations

DATASET_VERSION = "0.4.0"

# Surface occupancy channels. Underground and elevated transport are stored in
# auxiliary vertical-mode arrays so they can overlap buildings and surface roads.
CHANNEL_NAMES: tuple[str, ...] = (
    "water",
    "road_major_surface",
    "road_secondary_surface",
    "road_local_surface",
    "landuse_residential",
    "landuse_commercial_mixed",
    "landuse_industrial",
    "green",
    "building_footprint",
    "building_height_normalized",
    "landuse_civic",
    "rail_surface",
)

ROAD_CHANNELS: tuple[int, int, int] = (1, 2, 3)
LANDUSE_CHANNELS: tuple[int, int, int, int, int] = (4, 5, 6, 7, 10)
BINARY_CHANNELS: tuple[int, ...] = tuple(i for i in range(len(CHANNEL_NAMES)) if i != 9)

HEIGHT_CONFIDENCE_LABELS: tuple[str, ...] = (
    "type_default",
    "local_context",
    "levels_derived",
    "explicit_metres",
)

VERTICAL_MODE_NAMES: tuple[str, ...] = (
    "surface",
    "underground",
    "elevated",
    "unknown",
)

VERTICAL_PROFILE_MODE_NAMES: tuple[str, ...] = (
    "surface",
    "underground",
    "elevated",
)

VERTICAL_PROFILE_CONFIDENCE_LABELS: tuple[str, ...] = (
    "missing",
    "inferred_from_structure",
    "tag_derived",
    "measured",
)
