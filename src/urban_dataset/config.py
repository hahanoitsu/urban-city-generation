from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _config_root(config_path: Path) -> Path:
    for parent in config_path.parents:
        if parent.name == "configs":
            return parent.parent
    return config_path.parent


@dataclass(frozen=True)
class ProjectConfig:
    city_id: str
    source_name: str = "OpenStreetMap contributors"
    source_license: str = "ODbL-1.0"
    source_snapshot: str | None = None


@dataclass(frozen=True)
class InputConfig:
    pbf_path: Path
    bbox_wgs84: tuple[float, float, float, float]
    metric_crs: str | int = "auto"
    extraction_backend: str = "auto"
    pyrosm_engine: str = "out_of_core"
    pyrosm_workers: int | str | None = "auto"
    keep_metadata: bool = False
    complete_relations: bool = True


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    overwrite: bool = False
    save_extracted_gpkg: bool = True
    save_tile_vectors: bool = True
    gpkg_path: Path | None = None


@dataclass(frozen=True)
class RasterConfig:
    tile_size_m: int = 1024
    pixels: int = 256
    stride_m: int = 1024
    all_touched: bool = True
    max_height_m: float = 180.0
    include_partial_tiles: bool = False


@dataclass(frozen=True)
class RoadConfig:
    widths_m: dict[str, float] = field(
        default_factory=lambda: {"major": 8.0, "secondary": 6.5, "local": 5.0}
    )
    lane_width_m: float = 3.25
    edge_margin_m: float = 0.8
    minimum_width_m: float = 2.5
    maximum_width_m: float = 24.0
    surface_all_touched: bool = False


@dataclass(frozen=True)
class HeightConfig:
    floor_height_m: float = 3.1
    default_height_m: float = 9.3
    default_by_building: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VerticalProfileConfig:
    road_default_elevated_m: float = 7.0
    rail_default_elevated_m: float = 8.0
    road_default_tunnel_depth_m: float = 8.0
    rail_default_tunnel_depth_m: float = 14.0
    layer_step_m: float = 5.0
    deck_thickness_m: float = 1.0
    embankment_offset_m: float = 3.0
    cutting_depth_m: float = 3.0
    road_max_grade: float = 0.08
    rail_max_grade: float = 0.035
    sample_step_m: float = 2.0


@dataclass(frozen=True)
class QualityConfig:
    minimum_buildings: int = 3
    minimum_road_length_m: float = 80.0
    minimum_nonempty_fraction: float = 0.002
    minimum_valid_fraction: float = 0.95
    reject_water_fraction_above: float = 0.98


@dataclass(frozen=True)
class BuildConfig:
    project: ProjectConfig
    input: InputConfig
    output: OutputConfig
    raster: RasterConfig
    roads: RoadConfig
    heights: HeightConfig
    quality: QualityConfig
    vertical_profiles: VerticalProfileConfig = field(default_factory=VerticalProfileConfig)


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key '{section}.{key}'")
    return mapping[key]


def _parse_workers(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value < 1:
            raise ConfigError("input.pyrosm_workers must be at least 1")
        return value
    text = str(value).strip().lower()
    if text in {"auto", "none"}:
        return None if text == "none" else "auto"
    try:
        number = int(text)
    except ValueError as exc:
        raise ConfigError("input.pyrosm_workers must be 'auto', null, or a positive integer") from exc
    if number < 1:
        raise ConfigError("input.pyrosm_workers must be at least 1")
    return number


def _positive_float(mapping: dict[str, Any], key: str, default: float, section: str) -> float:
    value = float(mapping.get(key, default))
    if value <= 0:
        raise ConfigError(f"{section}.{key} must be positive")
    return value


def load_build_config(path: str | Path) -> BuildConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    project_raw = raw.get("project", {})
    input_raw = raw.get("input", {})
    output_raw = raw.get("output", {})
    raster_raw = raw.get("raster", {})
    road_raw = raw.get("roads", {})
    height_raw = raw.get("heights", {})
    vertical_raw = raw.get("vertical_profiles", {})
    quality_raw = raw.get("quality", {})

    bbox = tuple(float(value) for value in _required(input_raw, "bbox_wgs84", "input"))
    if len(bbox) != 4:
        raise ConfigError("input.bbox_wgs84 must contain exactly four numbers")
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ConfigError("input.bbox_wgs84 bounds are not ordered correctly")

    tile_size = int(raster_raw.get("tile_size_m", 1024))
    stride = int(raster_raw.get("stride_m", tile_size))
    pixels = int(raster_raw.get("pixels", 256))
    if tile_size <= 0 or stride <= 0 or pixels <= 0:
        raise ConfigError("tile_size_m, stride_m, and pixels must all be positive")
    if stride > tile_size:
        raise ConfigError("raster.stride_m cannot exceed raster.tile_size_m")

    backend = str(input_raw.get("extraction_backend", "auto")).strip().lower()
    if backend not in {"auto", "pyrosm", "gdal"}:
        raise ConfigError("input.extraction_backend must be 'auto', 'pyrosm', or 'gdal'")
    engine = str(input_raw.get("pyrosm_engine", "out_of_core")).strip()
    if engine not in {"in_memory", "out_of_core"}:
        raise ConfigError("input.pyrosm_engine must be 'in_memory' or 'out_of_core'")

    project = ProjectConfig(
        city_id=str(_required(project_raw, "city_id", "project")).strip(),
        source_name=str(project_raw.get("source_name", "OpenStreetMap contributors")),
        source_license=str(project_raw.get("source_license", "ODbL-1.0")),
        source_snapshot=(
            str(project_raw["source_snapshot"]).strip()
            if project_raw.get("source_snapshot") is not None
            else None
        ),
    )
    if not project.city_id:
        raise ConfigError("project.city_id cannot be empty")

    pbf_path = Path(_required(input_raw, "pbf_path", "input")).expanduser()
    if not pbf_path.is_absolute():
        pbf_path = (_config_root(config_path) / pbf_path).resolve()

    output_root = Path(_required(output_raw, "root", "output")).expanduser()
    if not output_root.is_absolute():
        output_root = (_config_root(config_path) / output_root).resolve()
    gpkg_path = None
    if output_raw.get("gpkg_path") is not None:
        gpkg_path = Path(output_raw["gpkg_path"]).expanduser()
        if not gpkg_path.is_absolute():
            gpkg_path = (_config_root(config_path) / gpkg_path).resolve()

    vertical_profiles = VerticalProfileConfig(
        road_default_elevated_m=_positive_float(
            vertical_raw, "road_default_elevated_m", 7.0, "vertical_profiles"
        ),
        rail_default_elevated_m=_positive_float(
            vertical_raw, "rail_default_elevated_m", 8.0, "vertical_profiles"
        ),
        road_default_tunnel_depth_m=_positive_float(
            vertical_raw, "road_default_tunnel_depth_m", 8.0, "vertical_profiles"
        ),
        rail_default_tunnel_depth_m=_positive_float(
            vertical_raw, "rail_default_tunnel_depth_m", 14.0, "vertical_profiles"
        ),
        layer_step_m=_positive_float(vertical_raw, "layer_step_m", 5.0, "vertical_profiles"),
        deck_thickness_m=_positive_float(
            vertical_raw, "deck_thickness_m", 1.0, "vertical_profiles"
        ),
        embankment_offset_m=_positive_float(
            vertical_raw, "embankment_offset_m", 3.0, "vertical_profiles"
        ),
        cutting_depth_m=_positive_float(
            vertical_raw, "cutting_depth_m", 3.0, "vertical_profiles"
        ),
        road_max_grade=_positive_float(vertical_raw, "road_max_grade", 0.08, "vertical_profiles"),
        rail_max_grade=_positive_float(
            vertical_raw, "rail_max_grade", 0.035, "vertical_profiles"
        ),
        sample_step_m=_positive_float(vertical_raw, "sample_step_m", 2.0, "vertical_profiles"),
    )
    if vertical_profiles.road_max_grade > 0.25 or vertical_profiles.rail_max_grade > 0.15:
        raise ConfigError("Configured transport grades are implausibly large")

    return BuildConfig(
        project=project,
        input=InputConfig(
            pbf_path=pbf_path,
            bbox_wgs84=bbox,
            metric_crs=input_raw.get("metric_crs", "auto"),
            extraction_backend=backend,
            pyrosm_engine=engine,
            pyrosm_workers=_parse_workers(input_raw.get("pyrosm_workers", "auto")),
            keep_metadata=bool(input_raw.get("keep_metadata", False)),
            complete_relations=bool(input_raw.get("complete_relations", True)),
        ),
        output=OutputConfig(
            root=output_root,
            overwrite=bool(output_raw.get("overwrite", False)),
            save_extracted_gpkg=bool(output_raw.get("save_extracted_gpkg", True)),
            save_tile_vectors=bool(output_raw.get("save_tile_vectors", True)),
            gpkg_path=gpkg_path,
        ),
        raster=RasterConfig(
            tile_size_m=tile_size,
            pixels=pixels,
            stride_m=stride,
            all_touched=bool(raster_raw.get("all_touched", True)),
            max_height_m=float(raster_raw.get("max_height_m", 180.0)),
            include_partial_tiles=bool(raster_raw.get("include_partial_tiles", False)),
        ),
        roads=RoadConfig(
            widths_m={
                "major": float(road_raw.get("widths_m", {}).get("major", 8.0)),
                "secondary": float(road_raw.get("widths_m", {}).get("secondary", 6.5)),
                "local": float(road_raw.get("widths_m", {}).get("local", 5.0)),
            },
            lane_width_m=float(road_raw.get("lane_width_m", 3.25)),
            edge_margin_m=float(road_raw.get("edge_margin_m", 0.8)),
            minimum_width_m=float(road_raw.get("minimum_width_m", 2.5)),
            maximum_width_m=float(road_raw.get("maximum_width_m", 24.0)),
            surface_all_touched=bool(road_raw.get("surface_all_touched", False)),
        ),
        heights=HeightConfig(
            floor_height_m=float(height_raw.get("floor_height_m", 3.1)),
            default_height_m=float(height_raw.get("default_height_m", 9.3)),
            default_by_building={
                str(k): float(v) for k, v in height_raw.get("default_by_building", {}).items()
            },
        ),
        vertical_profiles=vertical_profiles,
        quality=QualityConfig(
            minimum_buildings=int(quality_raw.get("minimum_buildings", 3)),
            minimum_road_length_m=float(quality_raw.get("minimum_road_length_m", 80.0)),
            minimum_nonempty_fraction=float(
                quality_raw.get("minimum_nonempty_fraction", 0.002)
            ),
            minimum_valid_fraction=float(quality_raw.get("minimum_valid_fraction", 0.95)),
            reject_water_fraction_above=float(
                quality_raw.get("reject_water_fraction_above", 0.98)
            ),
        ),
    )
