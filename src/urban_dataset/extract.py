from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import re

import geopandas as gpd
import pandas as pd

from .classify import classify_highway, classify_landuse, clean_tag, road_is_usable


class ExtractionError(RuntimeError):
    pass



_TAG_PAIR = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')


def _parse_other_tags(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    text = str(value)
    result: dict[str, str] = {}
    for key, raw_value in _TAG_PAIR.findall(text):
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
    copy = frame.copy()
    for key in wanted:
        if key not in copy.columns:
            copy[key] = parsed.map(lambda tags, name=key: tags.get(name))
        else:
            missing = copy[key].isna()
            if missing.any():
                copy.loc[missing, key] = parsed[missing].map(lambda tags, name=key: tags.get(name))
    if "id" not in copy.columns:
        relation_ids = copy.get("osm_id", pd.Series(index=copy.index, dtype=object))
        way_ids = copy.get("osm_way_id", pd.Series(index=copy.index, dtype=object))
        copy["id"] = relation_ids.combine_first(way_ids)
    if "osm_type" not in copy.columns:
        if "osm_way_id" in copy.columns:
            relation_ids = copy.get("osm_id", pd.Series(index=copy.index, dtype=object))
            copy["osm_type"] = relation_ids.notna().map({True: "relation", False: "way"})
        else:
            # GDAL's lines layer uses osm_id for ordinary ways.
            copy["osm_type"] = "way"
    return copy


def _extract_with_gdal(
    pbf_path: Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> CityLayers:
    """Fallback PBF reader using GDAL's OSM driver through pyogrio.

    This avoids a hard dependency on pyrosm and reads only the requested bbox.
    GDAL stores non-promoted OSM tags in an `other_tags` HSTORE-like string,
    which is expanded here for the fields used by the dataset schema.
    """
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

    line_tags = [
        "oneway", "lanes", "width", "service", "access", "junction",
        "bridge", "tunnel", "layer", "railway", "highway", "name",
    ]
    lines = _expand_other_tags(lines, line_tags)
    if lines.crs is None:
        lines = lines.set_crs("EPSG:4326")

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
            "id", "osm_type", "highway", "name", "oneway", "lanes", "width",
            "service", "access", "junction", "bridge", "tunnel", "layer",
            "road_class", "geometry",
        ],
    )

    rail = lines[lines.get("railway", pd.Series(index=lines.index, dtype=object)).map(clean_tag).isin(
        {"rail", "light_rail", "subway", "tram"}
    )].copy()
    rail = _keep_columns(
        rail,
        ["id", "osm_type", "railway", "name", "bridge", "tunnel", "layer", "geometry"],
    )

    polygon_tags = [
        "building", "height", "building:levels", "amenity", "landuse", "leisure",
        "natural", "water", "waterway", "name",
    ]
    polygons = _expand_other_tags(polygons, polygon_tags)
    if polygons.crs is None:
        polygons = polygons.set_crs("EPSG:4326")
    polygons = polygons[polygons.geometry.notna() & ~polygons.geometry.is_empty].copy()

    buildings = polygons[
        polygons.get("building", pd.Series(index=polygons.index, dtype=object)).map(clean_tag).ne("")
        & polygons.get("building", pd.Series(index=polygons.index, dtype=object)).map(clean_tag).ne("no")
    ].copy()
    buildings = _keep_columns(
        buildings,
        ["id", "osm_type", "building", "height", "building:levels", "amenity", "name", "geometry"],
    )

    known_mask = pd.Series(False, index=polygons.index)
    for column in ["landuse", "amenity", "leisure", "natural"]:
        known_mask |= polygons.get(column, pd.Series(index=polygons.index, dtype=object)).map(clean_tag).ne("")
    landuse_known = polygons[known_mask].copy()
    landuse_known = _keep_columns(
        landuse_known,
        ["id", "osm_type", "landuse", "amenity", "leisure", "natural", "name", "geometry"],
    )

    classified = polygons.copy()
    classified["landuse_class"] = classified.apply(classify_landuse, axis=1)
    landuse = classified[classified["landuse_class"].notna()].copy()
    landuse = _keep_columns(
        landuse,
        [
            "id", "osm_type", "landuse", "amenity", "leisure", "natural", "name",
            "landuse_class", "geometry",
        ],
    )
    green = landuse[landuse["landuse_class"] == "green"].copy()

    natural = polygons.get("natural", pd.Series(index=polygons.index, dtype=object)).map(clean_tag)
    water_tag = polygons.get("water", pd.Series(index=polygons.index, dtype=object)).map(clean_tag)
    waterway = polygons.get("waterway", pd.Series(index=polygons.index, dtype=object)).map(clean_tag)
    landuse_tag = polygons.get("landuse", pd.Series(index=polygons.index, dtype=object)).map(clean_tag)
    water_mask = (
        natural.isin({"water", "bay", "wetland", "strait"})
        | water_tag.ne("")
        | waterway.isin({"riverbank", "canal", "dock"})
        | landuse_tag.isin({"reservoir", "basin"})
    )
    water = polygons[water_mask].copy()
    water = _keep_columns(
        water,
        ["id", "osm_type", "natural", "water", "waterway", "landuse", "name", "geometry"],
    )

    return CityLayers(
        roads=roads,
        buildings=buildings,
        landuse=landuse,
        landuse_known=landuse_known,
        water=water,
        green=green,
        rail=rail,
    )

@dataclass
class CityLayers:
    roads: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame
    landuse: gpd.GeoDataFrame
    landuse_known: gpd.GeoDataFrame
    water: gpd.GeoDataFrame
    green: gpd.GeoDataFrame
    rail: gpd.GeoDataFrame

    def items(self):
        return {
            "roads": self.roads,
            "buildings": self.buildings,
            "landuse": self.landuse,
            "landuse_known": self.landuse_known,
            "water": self.water,
            "green": self.green,
            "rail": self.rail,
        }.items()


def _empty(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _safe_extract(name: str, function: Callable[[], gpd.GeoDataFrame | None]) -> gpd.GeoDataFrame:
    try:
        frame = function()
    except Exception as exc:  # pyrosm exception types have varied by release
        raise ExtractionError(f"Failed while extracting {name}: {exc}") from exc
    if frame is None or frame.empty:
        return _empty()
    frame = frame.copy()
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    elif frame.crs.to_epsg() != 4326:
        frame = frame.to_crs("EPSG:4326")
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    return frame


def _custom(osm, custom_filter: dict, *, extra_attributes: list[str] | None = None):
    return osm.get_data_by_custom_criteria(
        custom_filter=custom_filter,
        filter_type="keep",
        keep_nodes=False,
        keep_ways=True,
        keep_relations=True,
        extra_attributes=extra_attributes,
    )


def _keep_columns(frame: gpd.GeoDataFrame, names: list[str]) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame
    for column in names:
        if column not in frame.columns and column != "geometry":
            frame[column] = None
    return frame[[name for name in names if name in frame.columns]].copy()


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
        lambda: osm.get_network(
            network_type="all",
            extra_attributes=[
                "name",
                "oneway",
                "lanes",
                "width",
                "service",
                "access",
                "junction",
                "bridge",
                "tunnel",
                "layer",
            ],
        ),
    )
    if not roads.empty:
        if "highway" not in roads.columns:
            roads["highway"] = None
        for column in ["service", "access"]:
            if column not in roads.columns:
                roads[column] = None
        roads["road_class"] = roads["highway"].map(classify_highway)
        roads = roads[roads["road_class"].notna()].copy()
        usable = roads.apply(road_is_usable, axis=1)
        roads = roads[usable].copy()
        roads = _keep_columns(
            roads,
            [
                "id",
                "osm_type",
                "highway",
                "name",
                "oneway",
                "lanes",
                "width",
                "service",
                "access",
                "junction",
                "bridge",
                "tunnel",
                "layer",
                "road_class",
                "geometry",
            ],
        )

    buildings = _safe_extract(
        "buildings",
        lambda: osm.get_buildings(
            extra_attributes=["height", "building:levels", "amenity", "name"]
        ),
    )
    buildings = _keep_columns(
        buildings,
        [
            "id",
            "osm_type",
            "building",
            "height",
            "building:levels",
            "amenity",
            "name",
            "geometry",
        ],
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
            {"railway": ["rail", "light_rail", "subway", "tram"]},
            extra_attributes=["name", "bridge", "tunnel", "layer"],
        ),
    )
    rail = _keep_columns(
        rail,
        ["id", "osm_type", "railway", "name", "bridge", "tunnel", "layer", "geometry"],
    )

    combined = _normalise_candidates([raw_landuse, raw_natural, amenities, leisure])
    if not combined.empty:
        for column in ["landuse", "amenity", "leisure", "natural"]:
            if column not in combined.columns:
                combined[column] = None
        # This is deliberately broader than the classified target layers. It marks
        # where OSM supplied a polygonal land-use/natural/amenity observation, so
        # zero target pixels can be distinguished from missing mapping coverage.
        landuse_known = combined[
            combined.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()
        landuse_known = _keep_columns(
            landuse_known,
            ["id", "osm_type", "landuse", "amenity", "leisure", "natural", "name", "geometry"],
        )

        combined["landuse_class"] = combined.apply(classify_landuse, axis=1)
        landuse = combined[combined["landuse_class"].notna()].copy()
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
        natural = copy["natural"].map(clean_tag)
        water_tag = copy["water"].map(clean_tag)
        waterway = copy["waterway"].map(clean_tag)
        landuse_tag = copy["landuse"].map(clean_tag)
        mask = (
            natural.isin({"water", "bay", "wetland", "strait"})
            | water_tag.ne("")
            | waterway.isin({"riverbank", "canal", "dock"})
            | landuse_tag.isin({"reservoir", "basin"})
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

    return CityLayers(
        roads=roads,
        buildings=buildings,
        landuse=landuse,
        landuse_known=landuse_known,
        water=water,
        green=green,
        rail=rail,
    )
