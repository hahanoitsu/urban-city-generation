from __future__ import annotations

DATASET_VERSION = "0.2.0"

# Learned target channels. Auxiliary confidence/coverage arrays are stored next to
# these tensors in layers.npz and should be used to mask uncertain supervision.
CHANNEL_NAMES: tuple[str, ...] = (
    "water",
    "road_major",
    "road_secondary",
    "road_local",
    "landuse_residential",
    "landuse_commercial_mixed",
    "landuse_industrial",
    "green",
    "building_footprint",
    "building_height_normalized",
    "landuse_civic",
    "rail_transit",
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
