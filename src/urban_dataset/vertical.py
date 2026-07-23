from __future__ import annotations

from enum import IntEnum
from typing import Any, Mapping

from .classify import clean_tag, first_number


class VerticalMode(IntEnum):
    SURFACE = 0
    UNDERGROUND = 1
    ELEVATED = 2
    UNKNOWN = 3


VERTICAL_MODE_NAMES: tuple[str, ...] = (
    "surface",
    "underground",
    "elevated",
    "unknown",
)

_FALSE_TAGS = {"", "no", "false", "0"}
_UNDERGROUND_LOCATIONS = {"underground", "subsurface", "below_ground"}
_ELEVATED_LOCATIONS = {"overground", "overhead", "elevated", "above_ground"}


def _positive_tag(value: Any) -> bool:
    return clean_tag(value) not in _FALSE_TAGS


def classify_vertical_mode(feature: Mapping[str, Any]) -> VerticalMode:
    """Classify transport without treating local layer order as metric elevation.

    Explicit tunnel, bridge and location tags decide the vertical mode. A non-zero
    layer without an explicit structural tag remains unknown because OSM layer is a
    local stacking relation, not a guarantee that a feature is underground or elevated.
    Untagged transport defaults to the surface.
    """

    tunnel = _positive_tag(feature.get("tunnel"))
    bridge = _positive_tag(feature.get("bridge"))
    location = clean_tag(feature.get("location"))
    layer = first_number(feature.get("layer"))

    underground = tunnel or location in _UNDERGROUND_LOCATIONS
    elevated = bridge or location in _ELEVATED_LOCATIONS

    if underground and elevated:
        return VerticalMode.UNKNOWN
    if underground:
        return VerticalMode.UNDERGROUND
    if elevated:
        return VerticalMode.ELEVATED
    if layer is not None and abs(layer) > 1e-9:
        return VerticalMode.UNKNOWN
    return VerticalMode.SURFACE


def vertical_mode_name(feature: Mapping[str, Any]) -> str:
    return VERTICAL_MODE_NAMES[int(classify_vertical_mode(feature))]
