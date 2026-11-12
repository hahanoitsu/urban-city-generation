from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, box
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from .classify import classify_highway, classify_landuse, clean_tag, road_is_usable, tag_is_true


class ExtractionError(RuntimeError):
    pass


TRANSPORT_VERTICAL_TAGS = [
    "bridge",
    "bridge:structure",
    "bridge:movable",
    "tunnel",
    "location",
    "layer",
    "level",
    "incline",
    "ele",
    "height",
    "min_height",
    "depth",
    "maxheight",
    "maxheight:physical",
    "embankment",
    "cutting",
]
ROAD_TAGS = [
    "name",
    "oneway",
    "lanes",
    "width",
    "service",
    "access",
    "junction",
    *TRANSPORT_VERTICAL_TAGS,
]
RAIL_TAGS = ["name", *TRANSPORT_VERTICAL_TAGS]
RAILWAYS = {"rail", "light_rail", "subway", "tram", "monorail"}
STOPS = {"station", "halt", "tram_stop"}
STATION_TAGS = ["name", "railway", "station", "subway", "light_rail", "monorail", "tram"]
BUILDING_TAGS = [
    "height",
    "min_height",
    "building:levels",
    "building:min_level",
    "roof:height",
    "amenity",
    "name",
]

_TAG_PAIR = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')


def _parse_other_tags(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    result: dict[str, str] = {}
    for key, raw_value in _TAG_PAIR.findall(str(value)):
        clean_key = key.replace(r'\"', '"')
        clean_value = raw_value.replace(r'\"', '"')
        result[clean_key] = clean_value
        if clean_key.endswith("_1") and clean_key[:-2] not in result:
            result[clean_key[:-2]] = clean_value
    return result


def _expand_other_tags(frame: gpd.GeoDataFrame, wanted: list[str]) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame
    parsed = frame.get("other_tags", pd.Series([None] * len(frame), index=frame.index)).map(
        _parse_other_tags
    )
    result = frame.copy()
    for key in wanted:
        if key not in result.columns:
            result[key] = parsed.map(lambda tags, name=key: tags.get(name))
        else:
            missing = result[key].isna()
            if missing.any():
                result.loc[missing, key] = parsed[missing].map(
                    lambda tags, name=key: tags.get(name)
                )
    if "id" not in result.columns:
        relation_ids = result.get("osm_id", pd.Series(index=result.index, dtype=object))
        way_ids = result.get("osm_way_id", pd.Series(index=result.index, dtype=object))
        result["id"] = relation_ids.combine_first(way_ids)
    if "osm_type" not in result.columns:
        if "osm_way_id" in result.columns:
            relation_ids = result.get("osm_id", pd.Series(index=result.index, dtype=object))
            result["osm_type"] = relation_ids.notna().map({True: "relation", False: "way"})
        else:
            result["osm_type"] = "way"
    return result


@dataclass
class CityLayers:
    roads: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame
    landuse: gpd.GeoDataFrame
    landuse_known: gpd.GeoDataFrame
    water: gpd.GeoDataFrame
    green: gpd.GeoDataFrame
    rail: gpd.GeoDataFrame
    stations: gpd.GeoDataFrame | None = None

    def __post_init__(self):
        if self.stations is None:
            self.stations = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=self.roads.crs)

    def items(self):
        return {
            "roads": self.roads,
            "buildings": self.buildings,
            "landuse": self.landuse,
            "landuse_known": self.landuse_known,
            "water": self.water,
            "green": self.green,
            "rail": self.rail,
            "stations": self.stations,
        }.items()


def _empty(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _keep_columns(frame: gpd.GeoDataFrame, names: list[str]) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    for column in names:
        if column not in result.columns and column != "geometry":
            result[column] = None
    return result[[name for name in names if name in result.columns]].copy()


def _normalise_candidates(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    candidates = [frame for frame in frames if not frame.empty]
    if not candidates:
        return _empty()
    columns = sorted(set().union(*(frame.columns for frame in candidates)))
    normalised = []
    for frame in candidates:
        copy = frame.copy()
        for column in columns:
            if column not in copy.columns:
                copy[column] = None
        normalised.append(copy[columns])
    return gpd.GeoDataFrame(
        pd.concat(normalised, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )


def _safe_extract(name: str, function: Callable[[], gpd.GeoDataFrame | None]) -> gpd.GeoDataFrame:
    try:
        frame = function()
    except Exception as exc:
        raise ExtractionError(f"Failed while extracting {name}: {exc}") from exc
    if frame is None or frame.empty:
        return _empty()
    frame = frame.copy()
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    elif frame.crs.to_epsg() != 4326:
        frame = frame.to_crs("EPSG:4326")
    return frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()


def _custom(osm, custom_filter: dict, *, extra_attributes: list[str] | None = None):
    return osm.get_data_by_custom_criteria(
        custom_filter=custom_filter,
        filter_type="keep",
        keep_nodes=False,
        keep_ways=True,
        keep_relations=True,
        extra_attributes=extra_attributes,
    )


def _classify_lines(lines: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    roads = lines.copy()
    if "highway" not in roads.columns:
        roads["highway"] = None
    roads["road_class"] = roads["highway"].map(classify_highway)
    roads = roads[roads["road_class"].notna()].copy()
    if not roads.empty:
        roads = roads[roads.apply(road_is_usable, axis=1)].copy()
    roads = _keep_columns(
        roads,
        [
            "id",
            "osm_type",
            "highway",
            *ROAD_TAGS,
            "road_class",
            "geometry",
        ],
    )

    railway = lines.get("railway", pd.Series(index=lines.index, dtype=object)).map(clean_tag)
    rail = lines[railway.isin(RAILWAYS)].copy()
    rail = _keep_columns(
        rail,
        ["id", "osm_type", "railway", *RAIL_TAGS, "geometry"],
    )
    return roads, rail


def station_kind(row: Any) -> str:
    if clean_tag(row.get("railway")) == "tram_stop":
        return "tram"
    kind = clean_tag(row.get("station"))
    if kind in {"subway", "light_rail", "monorail"}:
        return kind
    for name in ("subway", "light_rail", "monorail", "tram"):
        if tag_is_true(row.get(name)):
            return name
    return "rail"


def _classify_stations(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    found = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        railway = frame.get("railway", pd.Series(index=frame.index, dtype=object)).map(clean_tag)
        stops = frame[railway.isin(STOPS)].copy()
        if stops.empty:
            continue
        stops["geometry"] = stops.geometry.representative_point()
        stops["station_kind"] = stops.apply(station_kind, axis=1)
        found.append(
            _keep_columns(stops, ["id", "osm_type", "name", "railway", "station_kind", "geometry"])
        )
    return _normalise_candidates(found)


def _station_points(
    pbf_path: Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    import pyogrio

    points = pyogrio.read_dataframe(
        pbf_path,
        layer="points",
        bbox=bbox_wgs84,
        where="other_tags LIKE '%railway%'",
    )
    points = _expand_other_tags(points, STATION_TAGS)
    if points.crs is None:
        points = points.set_crs("EPSG:4326")
    points["osm_type"] = "node"
    return points


def _line_parts(geometry) -> list:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    return [part for item in getattr(geometry, "geoms", []) for part in _line_parts(item)]


def _sea(
    lines: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    bbox_wgs84: tuple[float, float, float, float],
    probe: float = 5e-5,
    busy: float = 1.0,
    cell: float = 0.02,
) -> gpd.GeoDataFrame:
    if lines.empty or "natural" not in lines.columns:
        return _empty()
    coast = lines[lines["natural"].map(clean_tag).eq("coastline")].geometry
    region = box(*bbox_wgs84)
    pieces = []
    for way in coast:
        for part in _line_parts(way.intersection(region)):
            if part.length < 1e-9:
                continue
            if way.project(Point(part.coords[-1])) < way.project(Point(part.coords[0])):
                part = part.reverse()
            pieces.append(part)
    if not pieces:
        return _empty()
    faces = list(polygonize(unary_union([region.boundary, *pieces])))
    if not faces:
        return _empty()
    tree = STRtree(faces)
    votes = np.zeros((len(faces), 2))
    for part in pieces:
        step = min(part.length * 0.01, 2.0 * probe)
        for share in (0.25, 0.5, 0.75):
            along = share * part.length
            here, ahead = part.interpolate(along), part.interpolate(along + step)
            dx, dy = ahead.x - here.x, ahead.y - here.y
            length = float(np.hypot(dx, dy))
            if length < 1e-12:
                continue
            dx, dy = dx / length * probe, dy / length * probe
            for side, point in enumerate((Point(here.x + dy, here.y - dx), Point(here.x - dy, here.y + dx))):
                for index in tree.query(point, predicate="within"):
                    votes[index, side] += part.length
    wet = [index for index in range(len(faces)) if votes[index, 0] > votes[index, 1]]
    if wet and not roads.empty:
        road_tree = STRtree(roads.geometry.values)
        squash = float(np.cos(np.radians((bbox_wgs84[1] + bbox_wgs84[3]) / 2.0)))
        kept = []
        for index in wet:
            face = faces[index]
            near = road_tree.query(face, predicate="intersects")
            length = float(shapely.length(shapely.intersection(roads.geometry.values[near], face)).sum()) if len(near) else 0.0
            if length * 111.32 * np.sqrt(squash) <= busy * face.area * 111.32**2 * squash:
                kept.append(index)
        wet = kept
    if not wet:
        return _empty()
    sea = unary_union([faces[index] for index in wet])
    if not polygons.empty and "natural" in polygons.columns:
        islands = polygons[polygons["natural"].map(clean_tag).eq("coastline")].geometry
        islands = [island for island in islands if island is not None and not island.is_empty]
        if islands:
            sea = sea.difference(unary_union(islands))
    west, south, east, north = bbox_wgs84
    rows = []
    for x in np.arange(west, east, cell):
        for y in np.arange(south, north, cell):
            part = shapely.clip_by_rect(sea, x, y, x + cell, y + cell)
            for item in getattr(part, "geoms", [part]):
                if item.geom_type == "Polygon" and not item.is_empty:
                    rows.append({"natural": "sea", "geometry": item})
    if not rows:
        return _empty()
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _with_sea(water: gpd.GeoDataFrame, sea: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if sea.empty:
        return water
    if water.empty:
        return sea
    return gpd.GeoDataFrame(pd.concat([water, sea], ignore_index=True), geometry="geometry", crs="EPSG:4326")


def _classify_polygons(polygons: gpd.GeoDataFrame) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    polygons = polygons[polygons.geometry.notna() & ~polygons.geometry.is_empty].copy()
    building_tag = polygons.get("building", pd.Series(index=polygons.index, dtype=object)).map(
        clean_tag
    )
    buildings = polygons[building_tag.ne("") & building_tag.ne("no")].copy()
    buildings = _keep_columns(
        buildings,
        ["id", "osm_type", "building", *BUILDING_TAGS, "geometry"],
    )

    known_mask = pd.Series(False, index=polygons.index)
    for column in ["landuse", "amenity", "leisure", "natural"]:
        known_mask |= polygons.get(column, pd.Series(index=polygons.index, dtype=object)).map(
            clean_tag
        ).ne("")
    landuse_known = _keep_columns(
        polygons[known_mask].copy(),
        ["id", "osm_type", "landuse", "amenity", "leisure", "natural", "name", "geometry"],
    )

    classified = polygons.copy()
    classified["landuse_class"] = classified.apply(classify_landuse, axis=1)
    landuse = classified[classified["landuse_class"].notna()].copy()
    landuse = _keep_columns(
        landuse,
        [
            "id",
            "osm_type",
            "landuse",
            "amenity",
            "leisure",
            "natural",
            "name",
            "landuse_class",
            "geometry",
        ],
    )
    green = landuse[landuse["landuse_class"] == "green"].copy()

    natural = polygons.get("natural", pd.Series(index=polygons.index, dtype=object)).map(clean_tag)
    water_tag = polygons.get("water", pd.Series(index=polygons.index, dtype=object)).map(clean_tag)
    waterway = polygons.get("waterway", pd.Series(index=polygons.index, dtype=object)).map(clean_tag)
    landuse_tag = polygons.get("landuse", pd.Series(index=polygons.index, dtype=object)).map(
        clean_tag
    )
    water_mask = (
        natural.isin({"water", "bay", "wetland", "strait"})
        | water_tag.ne("")
        | waterway.isin({"riverbank", "canal", "dock"})
        | landuse_tag.isin({"reservoir", "basin"})
    )
    water = _keep_columns(
        polygons[water_mask].copy(),
        ["id", "osm_type", "natural", "water", "waterway", "landuse", "name", "geometry"],
    )
    return buildings, landuse, landuse_known, water, green


def _extract_with_gdal(
    pbf_path: Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> CityLayers:
    try:
        import pyogrio
    except ImportError as exc:
        raise ExtractionError(
            "Neither pyrosm nor pyogrio/GDAL's OSM driver is available for PBF extraction."
        ) from exc

    try:
        lines = pyogrio.read_dataframe(pbf_path, layer="lines", bbox=bbox_wgs84)
        polygons = pyogrio.read_dataframe(pbf_path, layer="multipolygons", bbox=bbox_wgs84)
    except Exception as exc:
        raise ExtractionError(f"GDAL OSM extraction failed: {exc}") from exc

    lines = _expand_other_tags(
        lines,
        ["highway", "railway", "natural", *ROAD_TAGS],
    )
    polygons = _expand_other_tags(
        polygons,
        [
            "building",
            *BUILDING_TAGS,
            "landuse",
            "leisure",
            "natural",
            "water",
            "waterway",
            "railway",
            "station",
        ],
    )
    if lines.crs is None:
        lines = lines.set_crs("EPSG:4326")
    if polygons.crs is None:
        polygons = polygons.set_crs("EPSG:4326")
    try:
        points = _station_points(pbf_path, bbox_wgs84)
    except Exception as exc:
        raise ExtractionError(f"GDAL OSM station extraction failed: {exc}") from exc

    roads, rail = _classify_lines(lines)
    buildings, landuse, landuse_known, water, green = _classify_polygons(polygons)
    stations = _classify_stations([points, polygons])
    water = _with_sea(water, _sea(lines, polygons, roads, bbox_wgs84))
    return CityLayers(roads, buildings, landuse, landuse_known, water, green, rail, stations)


def extract_from_pbf(
    pbf_path: Path,
    bbox_wgs84: tuple[float, float, float, float],
    *,
    backend: str = "auto",
    engine: str = "out_of_core",
    workers: int | str | None = "auto",
    keep_metadata: bool = False,
    complete_relations: bool = True,
) -> CityLayers:
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file does not exist: {pbf_path}")
    backend = backend.strip().lower()
    if backend == "gdal":
        return _extract_with_gdal(pbf_path, bbox_wgs84)

    try:
        from pyrosm import OSM
    except ImportError as exc:
        if backend == "pyrosm":
            raise ExtractionError("The pyrosm backend was requested but pyrosm is not installed") from exc
        return _extract_with_gdal(pbf_path, bbox_wgs84)

    osm = OSM(
        str(pbf_path),
        bounding_box=list(bbox_wgs84),
        keep_metadata=keep_metadata,
        complete_relations=complete_relations,
        engine=engine,
        workers=workers,
    )

    roads = _safe_extract(
        "roads",
        lambda: osm.get_network(network_type="all", extra_attributes=ROAD_TAGS),
    )
    if not roads.empty:
        roads, _unused_rail = _classify_lines(roads)

    buildings = _safe_extract(
        "buildings",
        lambda: osm.get_buildings(extra_attributes=BUILDING_TAGS),
    )
    buildings = _keep_columns(
        buildings,
        ["id", "osm_type", "building", *BUILDING_TAGS, "geometry"],
    )

    raw_landuse = _safe_extract(
        "land use",
        lambda: osm.get_landuse(extra_attributes=["amenity", "leisure", "natural", "name"]),
    )
    raw_natural = _safe_extract(
        "natural features",
        lambda: osm.get_natural(
            extra_attributes=["water", "waterway", "landuse", "leisure", "name"]
        ),
    )
    amenities = _safe_extract(
        "civic amenities",
        lambda: _custom(
            osm,
            {
                "amenity": [
                    "school",
                    "college",
                    "university",
                    "hospital",
                    "clinic",
                    "place_of_worship",
                    "community_centre",
                    "townhall",
                ]
            },
            extra_attributes=["landuse", "leisure", "natural", "name"],
        ),
    )
    leisure = _safe_extract(
        "leisure areas",
        lambda: _custom(
            osm,
            {
                "leisure": [
                    "park",
                    "garden",
                    "nature_reserve",
                    "golf_course",
                    "pitch",
                    "playground",
                ]
            },
            extra_attributes=["landuse", "amenity", "natural", "name"],
        ),
    )
    waterways = _safe_extract(
        "water",
        lambda: _custom(
            osm,
            {
                "natural": ["water", "bay", "wetland", "strait"],
                "waterway": ["riverbank", "canal", "dock"],
                "landuse": ["reservoir", "basin"],
            },
            extra_attributes=["name"],
        ),
    )
    rail = _safe_extract(
        "rail",
        lambda: _custom(
            osm,
            {"railway": sorted(RAILWAYS)},
            extra_attributes=RAIL_TAGS,
        ),
    )
    rail = _keep_columns(rail, ["id", "osm_type", "railway", *RAIL_TAGS, "geometry"])

    combined = _normalise_candidates([raw_landuse, raw_natural, amenities, leisure])
    if not combined.empty:
        for column in ["landuse", "amenity", "leisure", "natural"]:
            if column not in combined.columns:
                combined[column] = None
        landuse_known = _keep_columns(
            combined[combined.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy(),
            ["id", "osm_type", "landuse", "amenity", "leisure", "natural", "name", "geometry"],
        )
        combined["landuse_class"] = combined.apply(classify_landuse, axis=1)
        landuse = _keep_columns(
            combined[combined["landuse_class"].notna()].copy(),
            [
                "id",
                "osm_type",
                "landuse",
                "amenity",
                "leisure",
                "natural",
                "name",
                "landuse_class",
                "geometry",
            ],
        )
        green = landuse[landuse["landuse_class"] == "green"].copy()
    else:
        landuse = _empty()
        landuse_known = _empty()
        green = _empty()

    water_parts = []
    for frame in [raw_natural, waterways]:
        if frame.empty:
            continue
        copy = frame.copy()
        for column in ["natural", "water", "waterway", "landuse"]:
            if column not in copy.columns:
                copy[column] = None
        mask = (
            copy["natural"].map(clean_tag).isin({"water", "bay", "wetland", "strait"})
            | copy["water"].map(clean_tag).ne("")
            | copy["waterway"].map(clean_tag).isin({"riverbank", "canal", "dock"})
            | copy["landuse"].map(clean_tag).isin({"reservoir", "basin"})
        )
        water_parts.append(copy[mask])
    if water_parts:
        water = gpd.GeoDataFrame(
            pd.concat(water_parts, ignore_index=True), geometry="geometry", crs="EPSG:4326"
        )
        water = _keep_columns(
            water,
            ["id", "osm_type", "natural", "water", "waterway", "landuse", "name", "geometry"],
        )
    else:
        water = _empty()

    try:
        stations = _classify_stations([_station_points(pbf_path, bbox_wgs84)])
    except Exception:
        stations = _empty()
    return CityLayers(roads, buildings, landuse, landuse_known, water, green, rail, stations)
