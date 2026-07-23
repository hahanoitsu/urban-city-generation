from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyogrio

from .config import BuildConfig
from .enrich import enrich_layers
from .extract import CityLayers, extract_from_pbf
from .project import choose_metric_crs, project_and_clip_layers
from .schema import DATASET_VERSION
from .utils import file_sha256

LAYER_NAMES = (
    "roads",
    "buildings",
    "landuse",
    "landuse_known",
    "water",
    "green",
    "rail",
)


def _jsonify(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _safe_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = frame.copy()
    geometry_name = result.geometry.name
    for column in result.columns:
        if column == geometry_name:
            continue
        if pd.api.types.is_object_dtype(result[column].dtype):
            result[column] = result[column].map(_jsonify)
    return result


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS urban_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute("DELETE FROM urban_metadata")
        connection.executemany(
            "INSERT INTO urban_metadata (key, value) VALUES (?, ?)",
            [(key, json.dumps(value, ensure_ascii=False, sort_keys=True)) for key, value in metadata.items()],
        )
        connection.commit()


def read_city_metadata(path: str | Path) -> dict[str, Any]:
    gpkg_path = Path(path).expanduser().resolve()
    with sqlite3.connect(gpkg_path) as connection:
        rows = connection.execute("SELECT key, value FROM urban_metadata").fetchall()
    return {key: json.loads(value) for key, value in rows}


def save_city_gpkg(
    layers: CityLayers,
    path: str | Path,
    metadata: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    gpkg_path = Path(path).expanduser().resolve()
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    if gpkg_path.exists():
        if not overwrite:
            raise FileExistsError(f"Prepared city already exists: {gpkg_path}")
        gpkg_path.unlink()

    wrote = False
    for name, frame in layers.items():
        if frame.empty:
            continue
        _safe_frame(frame).to_file(
            gpkg_path,
            layer=name,
            driver="GPKG",
            engine="pyogrio",
        )
        wrote = True
    if not wrote:
        raise RuntimeError("No city layers were available to write")

    _write_metadata(gpkg_path, metadata)
    return gpkg_path


def load_city_gpkg(path: str | Path) -> tuple[CityLayers, dict[str, Any]]:
    gpkg_path = Path(path).expanduser().resolve()
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Prepared city does not exist: {gpkg_path}")

    available = {str(name) for name, _geometry_type in pyogrio.list_layers(gpkg_path)}
    frames: dict[str, gpd.GeoDataFrame] = {}
    city_crs = None
    for name in LAYER_NAMES:
        if name in available:
            frame = pyogrio.read_dataframe(gpkg_path, layer=name)
            if frame.crs is not None and city_crs is None:
                city_crs = frame.crs
            frames[name] = frame
        else:
            frames[name] = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=city_crs)

    if city_crs is None:
        raise ValueError(f"Prepared city has no spatial layers: {gpkg_path}")
    for name, frame in frames.items():
        if frame.crs is None:
            frames[name] = frame.set_crs(city_crs)
        elif frame.crs != city_crs:
            frames[name] = frame.to_crs(city_crs)

    return CityLayers(**frames), read_city_metadata(gpkg_path)


def prepare_city(config: BuildConfig) -> dict[str, Any]:
    gpkg_path = config.output.gpkg_path
    if gpkg_path is None:
        raise ValueError("output.gpkg_path is required for prepare-city")
    if not config.input.pbf_path.exists():
        raise FileNotFoundError(f"PBF file does not exist: {config.input.pbf_path}")

    source_digest = file_sha256(config.input.pbf_path)
    snapshot = config.project.source_snapshot or ""
    if snapshot.lower().startswith("sha256:"):
        expected = snapshot.split(":", 1)[1].strip().lower()
        if source_digest.lower() != expected:
            raise ValueError(
                f"Configured source checksum does not match the PBF: expected {expected}, "
                f"got {source_digest}"
            )

    layers_wgs84 = extract_from_pbf(
        config.input.pbf_path,
        config.input.bbox_wgs84,
        backend=config.input.extraction_backend,
        engine=config.input.pyrosm_engine,
        workers=config.input.pyrosm_workers,
        keep_metadata=config.input.keep_metadata,
        complete_relations=config.input.complete_relations,
    )
    metric_crs = choose_metric_crs(config.input.bbox_wgs84, config.input.metric_crs)
    layers, _boundary = project_and_clip_layers(
        layers_wgs84,
        config.input.bbox_wgs84,
        metric_crs,
    )
    layers = enrich_layers(layers, config)

    metadata = {
        "format": "urban-city-prepared-v1",
        "dataset_version": DATASET_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "city_id": config.project.city_id,
        "source_name": config.project.source_name,
        "source_license": config.project.source_license,
        "source_snapshot": config.project.source_snapshot,
        "source_file": {
            "name": config.input.pbf_path.name,
            "size_bytes": config.input.pbf_path.stat().st_size,
            "sha256": source_digest,
        },
        "bbox_wgs84": list(config.input.bbox_wgs84),
        "metric_crs": metric_crs.to_string(),
        "roads": asdict(config.roads),
        "heights": asdict(config.heights),
        "feature_counts": {name: len(frame) for name, frame in layers.items()},
    }
    save_city_gpkg(layers, gpkg_path, metadata, overwrite=config.output.overwrite)
    return {
        "city_id": config.project.city_id,
        "gpkg": str(gpkg_path),
        "metric_crs": metric_crs.to_string(),
        "source_sha256": source_digest,
        "feature_counts": metadata["feature_counts"],
    }
