from __future__ import annotations

import re
from typing import Any, Mapping

_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

MAJOR_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
}
SECONDARY_HIGHWAYS = {
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
}
LOCAL_HIGHWAYS = {
    "residential",
    "living_street",
    "unclassified",
    "service",
    "road",
}

EXCLUDED_SERVICE_TYPES = {
    "parking_aisle",
    "driveway",
    "alley",
    "drive-through",
    "emergency_access",
}
EXCLUDED_ACCESS = {"private", "no"}

# Conservative carriageway widths. OSM commonly represents divided roads as one
# way per carriageway, so very wide class-level buffers create double-width roads.
DEFAULT_WIDTH_BY_HIGHWAY_M: dict[str, float] = {
    "motorway": 8.5,
    "motorway_link": 5.5,
    "trunk": 8.0,
    "trunk_link": 5.5,
    "primary": 7.5,
    "primary_link": 5.5,
    "secondary": 7.0,
    "secondary_link": 5.0,
    "tertiary": 6.5,
    "tertiary_link": 4.8,
    "residential": 5.5,
    "living_street": 4.5,
    "unclassified": 5.0,
    "service": 3.5,
    "road": 5.0,
}

RESIDENTIAL_LANDUSE = {"residential"}
COMMERCIAL_LANDUSE = {"commercial", "retail", "mixed_use", "office"}
INDUSTRIAL_LANDUSE = {"industrial", "construction", "brownfield", "port"}
CIVIC_LANDUSE = {
    "education",
    "school",
    "college",
    "university",
    "hospital",
    "healthcare",
    "civic",
    "institutional",
    "religious",
}
CIVIC_AMENITIES = {
    "school",
    "college",
    "university",
    "kindergarten",
    "hospital",
    "clinic",
    "healthcare",
    "place_of_worship",
    "community_centre",
    "townhall",
    "theatre",
    "arts_centre",
    "social_facility",
    "police",
    "library",
    "monastery",
    "language_school",
    "events_venue",
}
COMMERCIAL_AMENITIES = {
    "marketplace",
    "restaurant",
    "food_court",
    "cafe",
    "bank",
    "bar",
    "pub",
}
GREEN_LANDUSE = {
    "forest",
    "grass",
    "meadow",
    "recreation_ground",
    "village_green",
    "allotments",
    "cemetery",
    "orchard",
    "greenfield",
    "flowerbed",
}
GREEN_LEISURE = {
    "park",
    "garden",
    "nature_reserve",
    "golf_course",
    "pitch",
    "playground",
    "swimming_pool",
    "fitness_station",
    "stadium",
    "sports_hall",
    "sports_centre",
    "dog_park",
}


def clean_tag(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if ";" in text:
        text = text.split(";", 1)[0]
    return text


def first_number(value: Any) -> float | None:
    if value is None:
        return None
    match = _NUMBER.search(str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def parse_width_metres(value: Any) -> float | None:
    number = first_number(value)
    if number is None or number <= 0:
        return None
    text = str(value).strip().lower()
    if "ft" in text or "feet" in text or "foot" in text:
        number *= 0.3048
    return number if 1.5 <= number <= 60 else None


def parse_lane_count(value: Any) -> float | None:
    number = first_number(value)
    if number is None or not (0 < number <= 20):
        return None
    return number


def tag_is_true(value: Any) -> bool:
    return clean_tag(value) in {"yes", "true", "1", "-1"}


def classify_highway(value: Any) -> str | None:
    highway = clean_tag(value)
    if highway in MAJOR_HIGHWAYS:
        return "major"
    if highway in SECONDARY_HIGHWAYS:
        return "secondary"
    if highway in LOCAL_HIGHWAYS:
        return "local"
    return None


def road_is_usable(row: Mapping[str, Any]) -> bool:
    highway = clean_tag(row.get("highway"))
    if classify_highway(highway) is None:
        return False
    if clean_tag(row.get("service")) in EXCLUDED_SERVICE_TYPES:
        return False
    if clean_tag(row.get("access")) in EXCLUDED_ACCESS and highway == "service":
        return False
    return True


def estimate_road_width_metres(
    row: Mapping[str, Any],
    *,
    class_fallbacks: Mapping[str, float],
    lane_width_m: float = 3.25,
    edge_margin_m: float = 0.8,
    minimum_width_m: float = 2.5,
    maximum_width_m: float = 24.0,
) -> float:
    explicit = parse_width_metres(row.get("width"))
    if explicit is not None:
        return float(min(max(explicit, minimum_width_m), maximum_width_m))

    lanes = parse_lane_count(row.get("lanes"))
    if lanes is not None:
        estimated = lanes * lane_width_m + edge_margin_m
        return float(min(max(estimated, minimum_width_m), maximum_width_m))

    highway = clean_tag(row.get("highway"))
    road_class = classify_highway(highway)
    fallback = DEFAULT_WIDTH_BY_HIGHWAY_M.get(
        highway,
        float(class_fallbacks.get(road_class or "local", 5.0)),
    )
    return float(min(max(fallback, minimum_width_m), maximum_width_m))


def classify_landuse(row: Any) -> str | None:
    landuse = clean_tag(row.get("landuse"))
    amenity = clean_tag(row.get("amenity"))
    leisure = clean_tag(row.get("leisure"))
    natural = clean_tag(row.get("natural"))

    if landuse in RESIDENTIAL_LANDUSE:
        return "residential"
    if landuse in COMMERCIAL_LANDUSE or amenity in COMMERCIAL_AMENITIES:
        return "commercial_mixed"
    if landuse in INDUSTRIAL_LANDUSE:
        return "industrial"
    if landuse in CIVIC_LANDUSE or amenity in CIVIC_AMENITIES:
        return "civic"
    if landuse in GREEN_LANDUSE or leisure in GREEN_LEISURE or natural in {"wood", "grassland", "scrub"}:
        return "green"
    return None
