from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .classify import clean_tag, first_number


@dataclass(frozen=True)
class HeightEstimate:
    metres: float
    confidence: int
    source: str

    @property
    def observed(self) -> bool:
        """Backward-compatible name: explicit metres or levels-derived."""
        return self.confidence >= 2


def parse_height_metres(value: Any) -> float | None:
    number = first_number(value)
    if number is None or number <= 0:
        return None
    text = str(value).strip().lower()
    if "ft" in text or "feet" in text or "foot" in text:
        number *= 0.3048
    return number if 1.5 <= number <= 1000 else None


def estimate_building_height(
    row: Any,
    *,
    floor_height_m: float,
    default_height_m: float,
    default_by_building: dict[str, float],
) -> HeightEstimate:
    explicit = parse_height_metres(row.get("height"))
    if explicit is not None:
        return HeightEstimate(explicit, 3, "height")

    levels = first_number(row.get("building:levels"))
    if levels is not None and 0 < levels <= 300:
        return HeightEstimate(levels * floor_height_m, 2, "building:levels")

    building_type = clean_tag(row.get("building"))
    fallback = default_by_building.get(building_type, default_height_m)
    return HeightEstimate(float(fallback), 0, "default")
