from __future__ import annotations

import csv
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from tqdm import tqdm

from .config import BuildConfig
from .enrich import enrich_layers
from .extract import CityLayers, extract_from_pbf
from .prepared import save_city_gpkg
from .preview import save_preview
from .project import choose_metric_crs, project_and_clip_layers
from .raster import rasterize_tile
from .schema import CHANNEL_NAMES, DATASET_VERSION, VERTICAL_MODE_NAMES
from .tile import clip_layers, iter_tile_specs
from .utils import file_sha256, write_json
from .validate import validate_tile
from .vectors import tile_vector_payload
from .vertical_profile import PROFILE_CONFIDENCE_NAMES, PROFILE_MODE_NAMES


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _prepare_output(config: BuildConfig) -> None:
    root = config.output.root
    if root.exists() and any(root.iterdir()):
        if not config.output.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {root}. Set output.overwrite: true to replace it."
            )
        shutil.rmtree(root)
    (root / "tiles").mkdir(parents=True, exist_ok=True)
    (root / "rejected").mkdir(parents=True, exist_ok=True)


def _write_tiles(
    config: BuildConfig,
    layers: CityLayers,
    boundary: gpd.GeoDataFrame,
    metric_crs: CRS,
    *,
    source_file: dict | None,
    area_id: str | None = None,
    prepared_city: str | None = None,
    prepare_output: bool = True,
) -> dict:
    if prepare_output:
        _prepare_output(config)
    study_geometry = boundary.geometry.iloc[0]
    to_wgs84 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

    tile_specs = list(
        iter_tile_specs(
            config.project.city_id,
            tuple(boundary.total_bounds),
            config.raster.tile_size_m,
            config.raster.stride_m,
            include_partial_tiles=config.raster.include_partial_tiles,
        )
    )
    if not tile_specs:
        raise RuntimeError(
            "The study area is smaller than one complete tile. Enlarge the coordinate box or "
            "enable partial tiles for exploratory output."
        )

    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    label = f"{config.project.city_id}:{area_id}" if area_id else config.project.city_id

    for tile in tqdm(tile_specs, desc=f"Building {label} tiles"):
        tile_layers = clip_layers(layers, tile)
        valid_geometry = study_geometry.intersection(tile.geometry)
        raster = rasterize_tile(tile, tile_layers, config, valid_geometry=valid_geometry)
        validation = validate_tile(tile_layers, raster, config.quality)

        row = {
            "tile_id": tile.tile_id,
            "city_id": config.project.city_id,
            "area_id": area_id or "",
            "column": tile.column,
            "row": tile.row,
            "minx": tile.minx,
            "miny": tile.miny,
            "maxx": tile.maxx,
            "maxy": tile.maxy,
            **validation.metrics,
            "rejection_reasons": ";".join(validation.reasons),
        }

        if not validation.accepted:
            rejected_rows.append(row)
            write_json(config.output.root / "rejected" / f"{tile.tile_id}.json", row)
            continue

        tile_dir = config.output.root / "tiles" / tile.tile_id
        tile_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            tile_dir / "layers.npz",
            layers=raster.layers.astype(np.float32),
            height_confidence=raster.height_confidence.astype(np.uint8),
            height_known_mask=raster.height_known_mask.astype(np.uint8),
            landuse_known_mask=raster.landuse_known_mask.astype(np.uint8),
            road_centerlines=raster.road_centerlines.astype(np.uint8),
            valid_data_mask=raster.valid_data_mask.astype(np.uint8),
            road_vertical_masks=raster.road_vertical_masks.astype(np.uint8),
            rail_vertical_masks=raster.rail_vertical_masks.astype(np.uint8),
            road_vertical_profiles_m=raster.road_vertical_profiles_m.astype(np.float32),
            rail_vertical_profiles_m=raster.rail_vertical_profiles_m.astype(np.float32),
            road_vertical_profile_confidence=(
                raster.road_vertical_profile_confidence.astype(np.uint8)
            ),
            rail_vertical_profile_confidence=(
                raster.rail_vertical_profile_confidence.astype(np.uint8)
            ),
            surface_transport_reservation=raster.surface_transport_reservation.astype(np.uint8),
            buildable_surface_mask=raster.buildable_surface_mask.astype(np.uint8),
            buildability_known_mask=raster.buildability_known_mask.astype(np.uint8),
            channel_names=np.asarray(CHANNEL_NAMES),
            vertical_mode_names=np.asarray(VERTICAL_MODE_NAMES),
            profile_mode_names=np.asarray(PROFILE_MODE_NAMES),
            profile_confidence_names=np.asarray(PROFILE_CONFIDENCE_NAMES),
            affine=np.asarray(raster.transform, dtype=np.float64),
        )
        save_preview(raster.layers, tile_dir / "preview.png")

        source = {
            "name": config.project.source_name,
            "license": config.project.source_license,
            "attribution": "© OpenStreetMap contributors",
        }
        if config.project.source_snapshot:
            source["snapshot"] = config.project.source_snapshot
        west, south = to_wgs84.transform(tile.minx, tile.miny)
        east, north = to_wgs84.transform(tile.maxx, tile.maxy)
        metadata = {
            "dataset_version": DATASET_VERSION,
            "tile_id": tile.tile_id,
            "city_id": config.project.city_id,
            "area_id": area_id,
            "grid": {"column": tile.column, "row": tile.row},
            "source": source,
            "metric_crs": metric_crs.to_string(),
            "projected_bounds": [tile.minx, tile.miny, tile.maxx, tile.maxy],
            "wgs84_bounds": [west, south, east, north],
            "tile_size_m": config.raster.tile_size_m,
            "pixels": config.raster.pixels,
            "metres_per_pixel": config.raster.tile_size_m / config.raster.pixels,
            "channels": list(CHANNEL_NAMES),
            "vertical_modes": list(VERTICAL_MODE_NAMES),
            "vertical_profile_modes": list(PROFILE_MODE_NAMES),
            "vertical_profile_confidence": list(PROFILE_CONFIDENCE_NAMES),
            "vertical_profile_evidence": raster.vertical_profile_evidence,
            "height": {
                "normalization_max_m": config.raster.max_height_m,
                "confidence_levels": {
                    "0": "type_default",
                    "1": "local_context",
                    "2": "levels_derived",
                    "3": "explicit_metres",
                },
            },
            "auxiliary_arrays": {
                "height_confidence": "uint8[H,W]",
                "landuse_known_mask": "uint8[H,W]",
                "road_centerlines": "uint8[3,H,W] surface major/secondary/local",
                "valid_data_mask": "uint8[H,W]",
                "road_vertical_masks": "uint8[4,H,W] surface/underground/elevated/unknown",
                "rail_vertical_masks": "uint8[4,H,W] surface/underground/elevated/unknown",
                "road_vertical_profiles_m": (
                    "float32[3,H,W] signed surface/underground/elevated local z offsets"
                ),
                "rail_vertical_profiles_m": (
                    "float32[3,H,W] signed surface/underground/elevated local z offsets"
                ),
                "road_vertical_profile_confidence": "uint8[3,H,W] 0 missing, 1 inferred, 2 tag-derived, 3 measured",
                "rail_vertical_profile_confidence": "uint8[3,H,W] 0 missing, 1 inferred, 2 tag-derived, 3 measured",
                "surface_transport_reservation": "uint8[H,W]",
                "buildable_surface_mask": "uint8[H,W] excludes water and confirmed surface transport",
                "buildability_known_mask": "uint8[H,W] excludes ambiguous transport vertical mode",
            },
            "quality": validation.metrics,
        }
        if prepared_city is not None:
            metadata["prepared_city"] = prepared_city
        write_json(tile_dir / "metadata.json", metadata)
        if config.output.save_tile_vectors:
            write_json(
                tile_dir / "city.json",
                tile_vector_payload(tile_layers, tile, metric_crs.to_string()),
            )
        accepted_rows.append(row)

    all_rows = [*accepted_rows, *rejected_rows]
    fieldnames = sorted(set().union(*(row.keys() for row in all_rows))) if all_rows else ["tile_id"]
    for name, rows in [("index.csv", accepted_rows), ("rejected.csv", rejected_rows)]:
        with (config.output.root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    raw_config = asdict(config)
    raw_config["input"]["pbf_path"] = _portable_path(config.input.pbf_path)
    raw_config["output"]["root"] = _portable_path(config.output.root)
    if config.output.gpkg_path is not None:
        raw_config["output"]["gpkg_path"] = _portable_path(config.output.gpkg_path)
    summary = {
        "dataset_version": DATASET_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "city_id": config.project.city_id,
        "area_id": area_id,
        "metric_crs": metric_crs.to_string(),
        "tile_candidates": len(tile_specs),
        "accepted_tiles": len(accepted_rows),
        "rejected_tiles": len(rejected_rows),
        "channels": list(CHANNEL_NAMES),
        "vertical_modes": list(VERTICAL_MODE_NAMES),
        "vertical_profile_modes": list(PROFILE_MODE_NAMES),
        "source_file": source_file,
        "prepared_city": prepared_city,
        "config": raw_config,
    }
    write_json(config.output.root / "dataset_summary.json", summary)
    return summary


def run_build(config: BuildConfig, *, extracted_layers: CityLayers | None = None) -> dict:
    source_digest: str | None = None
    if config.input.pbf_path.exists():
        source_digest = file_sha256(config.input.pbf_path)
        snapshot = config.project.source_snapshot or ""
        if snapshot.lower().startswith("sha256:"):
            expected = snapshot.split(":", 1)[1].strip().lower()
            if source_digest.lower() != expected:
                raise ValueError(
                    f"Configured source checksum does not match the PBF: expected {expected}, "
                    f"got {source_digest}"
                )

    layers_wgs84 = extracted_layers or extract_from_pbf(
        config.input.pbf_path,
        config.input.bbox_wgs84,
        backend=config.input.extraction_backend,
        engine=config.input.pyrosm_engine,
        workers=config.input.pyrosm_workers,
        keep_metadata=config.input.keep_metadata,
        complete_relations=config.input.complete_relations,
    )
    metric_crs = choose_metric_crs(config.input.bbox_wgs84, config.input.metric_crs)
    layers, boundary = project_and_clip_layers(
        layers_wgs84,
        config.input.bbox_wgs84,
        metric_crs,
    )
    layers = enrich_layers(layers, config)

    _prepare_output(config)

    if config.output.save_extracted_gpkg:
        metadata = {
            "format": "urban-city-extracted-v1",
            "city_id": config.project.city_id,
            "source_name": config.project.source_name,
            "source_license": config.project.source_license,
            "source_snapshot": config.project.source_snapshot,
            "source_sha256": source_digest,
            "bbox_wgs84": list(config.input.bbox_wgs84),
            "metric_crs": metric_crs.to_string(),
        }
        save_city_gpkg(
            layers,
            config.output.root / "extracted_layers.gpkg",
            metadata,
            overwrite=True,
        )

    source_file = None
    if config.input.pbf_path.exists():
        source_file = {
            "kind": "osm_pbf",
            "name": config.input.pbf_path.name,
            "size_bytes": config.input.pbf_path.stat().st_size,
            "sha256": source_digest,
        }
    return _write_tiles(
        config,
        layers,
        boundary,
        metric_crs,
        source_file=source_file,
        prepare_output=False,
    )


def run_prepared_build(
    config: BuildConfig,
    layers: CityLayers,
    boundary: gpd.GeoDataFrame,
    metric_crs: CRS,
    *,
    area_id: str,
    prepared_city_path: Path,
    source_file: dict | None,
) -> dict:
    return _write_tiles(
        config,
        layers,
        boundary,
        metric_crs,
        source_file=source_file,
        area_id=area_id,
        prepared_city=_portable_path(prepared_city_path),
    )
