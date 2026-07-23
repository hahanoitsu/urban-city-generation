from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import yaml
from pyproj import CRS
from shapely.geometry import box

from .audit import audit_dataset, create_preview_atlas
from .config import (
    BuildConfig,
    HeightConfig,
    InputConfig,
    OutputConfig,
    ProjectConfig,
    QualityConfig,
    RasterConfig,
    RoadConfig,
)
from .extract import CityLayers
from .manifests import build_manifests_from_values
from .pipeline import run_prepared_build
from .prepared import load_city_gpkg
from .utils import write_json


@dataclass(frozen=True)
class AreaConfig:
    area_id: str
    bbox_wgs84: tuple[float, float, float, float]


@dataclass(frozen=True)
class CorpusCityConfig:
    city_id: str
    gpkg_path: Path
    split: str = "auto"
    areas: tuple[AreaConfig, ...] = ()


@dataclass(frozen=True)
class CorpusConfig:
    output_root: Path
    manifest_root: Path
    cities: tuple[CorpusCityConfig, ...]
    raster: RasterConfig = field(default_factory=RasterConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    seed: int = 5132
    spatial_group_tiles: int = 4
    overwrite: bool = False
    save_tile_vectors: bool = False
    atlas_columns: int = 6
    atlas_limit: int = 160


def _config_root(config_path: Path) -> Path:
    for parent in config_path.parents:
        if parent.name == "configs":
            return parent.parent
    return config_path.parent


def _resolve(path: str | Path, config_path: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = (_config_root(config_path) / value).resolve()
    return value


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    bounds = tuple(float(number) for number in value)
    if len(bounds) != 4:
        raise ValueError(f"{label} must contain four numbers")
    min_lon, min_lat, max_lon, max_lat = bounds
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"{label} is not ordered correctly")
    return bounds


def load_corpus_config(path: str | Path) -> CorpusConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    tile = raw.get("tile", {})
    quality = raw.get("quality", {})
    cities: list[CorpusCityConfig] = []
    seen_cities: set[str] = set()
    for city_raw in raw.get("cities", []):
        city_id = str(city_raw.get("id", "")).strip()
        if not city_id or city_id in seen_cities:
            raise ValueError("Each corpus city needs a unique non-empty id")
        seen_cities.add(city_id)
        split = str(city_raw.get("split", "auto")).strip().lower()
        if split not in {"auto", "train", "validation", "test"}:
            raise ValueError(f"Invalid split for {city_id}: {split}")

        areas: list[AreaConfig] = []
        seen_areas: set[str] = set()
        for area_raw in city_raw.get("areas", []):
            area_id = str(area_raw.get("id", "")).strip()
            if not area_id or area_id in seen_areas:
                raise ValueError(f"Area ids must be unique within {city_id}")
            seen_areas.add(area_id)
            areas.append(
                AreaConfig(
                    area_id=area_id,
                    bbox_wgs84=_bbox(area_raw.get("bbox_wgs84", []), f"{city_id}.{area_id}"),
                )
            )
        if not areas:
            raise ValueError(f"No study areas are configured for {city_id}")
        cities.append(
            CorpusCityConfig(
                city_id=city_id,
                gpkg_path=_resolve(city_raw["gpkg"], config_path),
                split=split,
                areas=tuple(areas),
            )
        )

    if not cities:
        raise ValueError("The corpus config does not contain any cities")

    tile_size = int(tile.get("size_m", 1024))
    stride = int(tile.get("stride_m", tile_size))
    pixels = int(tile.get("pixels", 256))
    if min(tile_size, stride, pixels) <= 0 or stride > tile_size:
        raise ValueError("Invalid tile size, stride or pixel count")

    return CorpusConfig(
        output_root=_resolve(raw.get("output_root", "data/processed/corpus"), config_path),
        manifest_root=_resolve(raw.get("manifest_root", "data/manifests/corpus"), config_path),
        cities=tuple(cities),
        raster=RasterConfig(
            tile_size_m=tile_size,
            pixels=pixels,
            stride_m=stride,
            all_touched=bool(tile.get("all_touched", True)),
            max_height_m=float(tile.get("max_height_m", 180.0)),
            include_partial_tiles=bool(tile.get("include_partial_tiles", False)),
        ),
        quality=QualityConfig(
            minimum_buildings=int(quality.get("minimum_buildings", 0)),
            minimum_road_length_m=float(quality.get("minimum_road_length_m", 0)),
            minimum_nonempty_fraction=float(quality.get("minimum_nonempty_fraction", 0.001)),
            minimum_valid_fraction=float(quality.get("minimum_valid_fraction", 0.95)),
            reject_water_fraction_above=float(quality.get("reject_water_fraction_above", 0.95)),
        ),
        split_ratios={
            name: float(raw.get("split_ratios", {}).get(name, default))
            for name, default in {"train": 0.8, "validation": 0.1, "test": 0.1}.items()
        },
        seed=int(raw.get("seed", 5132)),
        spatial_group_tiles=int(raw.get("spatial_group_tiles", 4)),
        overwrite=bool(raw.get("overwrite", False)),
        save_tile_vectors=bool(raw.get("save_tile_vectors", False)),
        atlas_columns=int(raw.get("atlas", {}).get("columns", 6)),
        atlas_limit=int(raw.get("atlas", {}).get("limit", 160)),
    )


def _check_area_overlap(city: CorpusCityConfig) -> None:
    shapes = [(area, box(*area.bbox_wgs84)) for area in city.areas]
    for index, (left, left_shape) in enumerate(shapes):
        for right, right_shape in shapes[index + 1 :]:
            if left_shape.intersection(right_shape).area > 0:
                raise ValueError(
                    f"Study areas overlap in {city.city_id}: {left.area_id} and {right.area_id}"
                )


def _clip_frame(frame: gpd.GeoDataFrame, geometry) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.copy()
    indexes = list(frame.sindex.query(geometry, predicate="intersects"))
    if not indexes:
        return frame.iloc[0:0].copy()
    result = frame.iloc[indexes].copy()
    result["geometry"] = result.geometry.intersection(geometry)
    return result[result.geometry.notna() & ~result.geometry.is_empty].copy()


def _clip_layers(layers: CityLayers, geometry) -> CityLayers:
    return CityLayers(**{name: _clip_frame(frame, geometry) for name, frame in layers.items()})


def _settings_from_metadata(metadata: dict[str, Any]) -> tuple[RoadConfig, HeightConfig]:
    road_raw = metadata.get("roads", {})
    height_raw = metadata.get("heights", {})
    roads = RoadConfig(
        widths_m={
            name: float(road_raw.get("widths_m", {}).get(name, default))
            for name, default in {"major": 8.0, "secondary": 6.5, "local": 5.0}.items()
        },
        lane_width_m=float(road_raw.get("lane_width_m", 3.25)),
        edge_margin_m=float(road_raw.get("edge_margin_m", 0.8)),
        minimum_width_m=float(road_raw.get("minimum_width_m", 2.5)),
        maximum_width_m=float(road_raw.get("maximum_width_m", 24.0)),
        surface_all_touched=bool(road_raw.get("surface_all_touched", False)),
    )
    heights = HeightConfig(
        floor_height_m=float(height_raw.get("floor_height_m", 3.1)),
        default_height_m=float(height_raw.get("default_height_m", 9.3)),
        default_by_building={
            str(key): float(value) for key, value in height_raw.get("default_by_building", {}).items()
        },
    )
    return roads, heights


def build_corpus(config: CorpusConfig) -> dict[str, Any]:
    if config.output_root.exists() and config.overwrite:
        shutil.rmtree(config.output_root)
    if config.manifest_root.exists() and config.overwrite:
        shutil.rmtree(config.manifest_root)
    config.output_root.mkdir(parents=True, exist_ok=True)

    area_summaries: list[dict[str, Any]] = []
    validation_cities: set[str] = set()
    test_cities: set[str] = set()

    for city in config.cities:
        _check_area_overlap(city)
        layers, metadata = load_city_gpkg(city.gpkg_path)
        metadata_city = str(metadata.get("city_id", city.city_id))
        if metadata_city != city.city_id:
            raise ValueError(
                f"Prepared city id is {metadata_city!r}, but the corpus config uses {city.city_id!r}"
            )
        metric_crs = CRS.from_user_input(metadata.get("metric_crs") or layers.roads.crs)
        roads, heights = _settings_from_metadata(metadata)

        if city.split == "validation":
            validation_cities.add(city.city_id)
        elif city.split == "test":
            test_cities.add(city.city_id)

        prepared_bounds = metadata.get("bbox_wgs84")
        prepared_shape = box(*prepared_bounds) if prepared_bounds else None
        for area in city.areas:
            area_shape = box(*area.bbox_wgs84)
            if prepared_shape is not None and not prepared_shape.covers(area_shape):
                raise ValueError(
                    f"Study area {city.city_id}.{area.area_id} extends outside the prepared city bounds"
                )

            boundary_wgs84 = gpd.GeoDataFrame(
                {"geometry": [area_shape]},
                geometry="geometry",
                crs="EPSG:4326",
            )
            boundary = boundary_wgs84.to_crs(metric_crs)
            clipped = _clip_layers(layers, boundary.geometry.iloc[0])
            output_dir = config.output_root / f"{city.city_id}-{area.area_id}"
            build_config = BuildConfig(
                project=ProjectConfig(
                    city_id=city.city_id,
                    source_name=str(metadata.get("source_name", "OpenStreetMap contributors")),
                    source_license=str(metadata.get("source_license", "ODbL-1.0")),
                    source_snapshot=metadata.get("source_snapshot"),
                ),
                input=InputConfig(
                    pbf_path=city.gpkg_path,
                    bbox_wgs84=area.bbox_wgs84,
                    metric_crs=metric_crs.to_string(),
                ),
                output=OutputConfig(
                    root=output_dir,
                    overwrite=config.overwrite,
                    save_extracted_gpkg=False,
                    save_tile_vectors=config.save_tile_vectors,
                ),
                raster=config.raster,
                roads=roads,
                heights=heights,
                quality=config.quality,
            )
            source_file = {
                "kind": "prepared_gpkg",
                "name": city.gpkg_path.name,
                "original_source": metadata.get("source_file"),
            }
            summary = run_prepared_build(
                build_config,
                clipped,
                boundary,
                metric_crs,
                area_id=area.area_id,
                prepared_city_path=city.gpkg_path,
                source_file=source_file,
            )
            audit_dataset(output_dir)
            create_preview_atlas(
                output_dir,
                output_dir / "atlas.png",
                columns=config.atlas_columns,
                limit=config.atlas_limit,
            )
            area_summaries.append(summary)

    manifest_summary = build_manifests_from_values(
        config.output_root,
        config.manifest_root,
        seed=config.seed,
        split_ratios=config.split_ratios,
        validation_cities=validation_cities,
        test_cities=test_cities,
        spatial_group_tiles=config.spatial_group_tiles,
    )
    combined_audit = audit_dataset(config.output_root, config.output_root / "audit.json")
    create_preview_atlas(
        config.output_root,
        config.output_root / "atlas.png",
        columns=config.atlas_columns,
        limit=config.atlas_limit,
    )

    result = {
        "output_root": str(config.output_root),
        "manifest_root": str(config.manifest_root),
        "cities": len(config.cities),
        "areas": len(area_summaries),
        "tile_candidates": sum(int(item["tile_candidates"]) for item in area_summaries),
        "accepted_tiles": sum(int(item["accepted_tiles"]) for item in area_summaries),
        "rejected_tiles": sum(int(item["rejected_tiles"]) for item in area_summaries),
        "manifests": manifest_summary,
        "audit": combined_audit,
    }
    write_json(config.output_root / "corpus_summary.json", result)
    return result
