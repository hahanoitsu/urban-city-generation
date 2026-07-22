from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm

from .config import BuildConfig
from .extract import CityLayers, extract_from_pbf
from .enrich import enrich_layers
from .preview import save_preview
from .project import choose_metric_crs, project_and_clip_layers
from .raster import rasterize_tile
from .schema import CHANNEL_NAMES, DATASET_VERSION
from .tile import clip_layers, iter_tile_specs
from .utils import file_sha256, write_json
from .validate import validate_tile
from .vectors import tile_vector_payload


def _jsonify_object(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _gpkg_safe_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    safe = frame.copy()
    geometry_name = safe.geometry.name
    for column in safe.columns:
        if column == geometry_name:
            continue
        if pd.api.types.is_object_dtype(safe[column].dtype):
            safe[column] = safe[column].map(_jsonify_object)
    return safe


def _save_gpkg(layers: CityLayers, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    wrote = False
    for name, frame in layers.items():
        if frame.empty:
            continue
        _gpkg_safe_frame(frame).to_file(path, layer=name, driver="GPKG", engine="pyogrio")
        wrote = True
    if not wrote:
        raise RuntimeError("No extracted layers were available to write")


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


def run_build(config: BuildConfig, *, extracted_layers: CityLayers | None = None) -> dict:
    source_digest: str | None = None
    if config.input.pbf_path.exists():
        source_digest = file_sha256(config.input.pbf_path)
        snapshot = config.project.source_snapshot or ""
        if snapshot.lower().startswith("sha256:"):
            expected = snapshot.split(":", 1)[1].strip().lower()
            if source_digest.lower() != expected:
                raise ValueError(
                    "Configured source_snapshot checksum does not match the PBF: "
                    f"expected {expected}, got {source_digest}"
                )
    _prepare_output(config)
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
        layers_wgs84, config.input.bbox_wgs84, metric_crs
    )
    layers = enrich_layers(layers, config)
    study_geometry = boundary.geometry.iloc[0]
    to_wgs84 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

    if config.output.save_extracted_gpkg:
        _save_gpkg(layers, config.output.root / "extracted_layers.gpkg")

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
            "The projected study area is smaller than one complete tile. Enlarge the bounding box "
            "or set raster.include_partial_tiles: true for exploratory output."
        )

    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []

    for tile in tqdm(tile_specs, desc=f"Building {config.project.city_id} tiles"):
        tile_layers = clip_layers(layers, tile)
        valid_geometry = study_geometry.intersection(tile.geometry)
        raster = rasterize_tile(
            tile,
            tile_layers,
            config,
            valid_geometry=valid_geometry,
        )
        validation = validate_tile(tile_layers, raster, config.quality)

        row = {
            "tile_id": tile.tile_id,
            "city_id": config.project.city_id,
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
            channel_names=np.asarray(CHANNEL_NAMES),
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
            "grid": {"column": tile.column, "row": tile.row},
            "source": source,
            "metric_crs": metric_crs.to_string(),
            "projected_bounds": [tile.minx, tile.miny, tile.maxx, tile.maxy],
            "wgs84_bounds": [west, south, east, north],
            "tile_size_m": config.raster.tile_size_m,
            "pixels": config.raster.pixels,
            "metres_per_pixel": config.raster.tile_size_m / config.raster.pixels,
            "channels": list(CHANNEL_NAMES),
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
                "road_centerlines": "uint8[3,H,W] in major/secondary/local order",
                "valid_data_mask": "uint8[H,W]",
            },
            "quality": validation.metrics,
        }
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

    source_file = None
    if config.input.pbf_path.exists():
        source_file = {
            "name": config.input.pbf_path.name,
            "size_bytes": config.input.pbf_path.stat().st_size,
            "sha256": source_digest,
        }
    summary = {
        "dataset_version": DATASET_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "city_id": config.project.city_id,
        "metric_crs": metric_crs.to_string(),
        "tile_candidates": len(tile_specs),
        "accepted_tiles": len(accepted_rows),
        "rejected_tiles": len(rejected_rows),
        "channels": list(CHANNEL_NAMES),
        "source_file": source_file,
        "config": asdict(config),
    }
    summary["config"]["input"]["pbf_path"] = _portable_path(config.input.pbf_path)
    summary["config"]["output"]["root"] = _portable_path(config.output.root)
    write_json(config.output.root / "dataset_summary.json", summary)
    return summary
