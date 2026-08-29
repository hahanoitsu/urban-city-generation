from __future__ import annotations

import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from urban_dataset.extract import CityLayers
from urban_dataset.prepared import load_city_gpkg, save_city_gpkg

from .corpus_correction import extract_monorail, prune_corpus_to_boundary

URA_PLANNING_AREA_DATASET = "d_4765db0e87b9c86336792efe8a1f7a66"


def download_data_gov_geojson(dataset_id: str, output: str | Path) -> Path:
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    api = (
        "https://api-open.data.gov.sg/v1/public/api/datasets/"
        f"{dataset_id}/poll-download"
    )
    request = urllib.request.Request(api, headers={"User-Agent": "urban-city-generation/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("code") != 0:
        raise RuntimeError(payload.get("errMsg") or f"data.gov.sg returned {payload!r}")

    url = payload.get("data", {}).get("url")
    if not url:
        raise RuntimeError("data.gov.sg did not return a download URL")

    request = urllib.request.Request(url, headers={"User-Agent": "urban-city-generation/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()

    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith((".geojson", ".json"))]
            if not names:
                raise RuntimeError("Downloaded archive contains no GeoJSON")
            content = archive.read(names[0])

    try:
        parsed = json.loads(content)
    except Exception as exc:
        raise RuntimeError("Downloaded boundary is not GeoJSON") from exc
    if parsed.get("type") != "FeatureCollection" or not parsed.get("features"):
        raise RuntimeError("Downloaded boundary has no GeoJSON features")

    target.write_bytes(content)
    return target


def prepare_monorail_city(
    prepared_city: str | Path,
    pbf_path: str | Path,
    output_city: str | Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(prepared_city).expanduser().resolve()
    output = Path(output_city).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(output)

    layers, metadata = load_city_gpkg(source)
    metric_crs = CRS.from_user_input(metadata.get("metric_crs") or layers.roads.crs)
    monorail = extract_monorail(pbf_path, bbox_wgs84).to_crs(metric_crs)

    columns = sorted(set(layers.rail.columns) | set(monorail.columns))
    existing = layers.rail.copy()
    extra = monorail.copy()
    for column in columns:
        if column not in existing.columns:
            existing[column] = None
        if column not in extra.columns:
            extra[column] = None

    rail = gpd.GeoDataFrame(
        pd.concat([existing[columns], extra[columns]], ignore_index=True),
        geometry="geometry",
        crs=metric_crs,
    )
    corrected = CityLayers(
        roads=layers.roads,
        buildings=layers.buildings,
        landuse=layers.landuse,
        landuse_known=layers.landuse_known,
        water=layers.water,
        green=layers.green,
        rail=rail,
    )

    corrected_metadata = dict(metadata)
    corrected_metadata["format"] = "urban-city-prepared-v2-corrected"
    corrected_metadata["source_corrections"] = {
        "added_railway_types": ["monorail"],
        "country_mask": "applied after corpus build from official URA planning-area polygons",
    }
    corrected_metadata["feature_counts"] = {
        name: len(frame) for name, frame in corrected.items()
    }
    corrected_metadata["monorail_features_added"] = int(len(monorail))
    corrected_metadata["monorail_length_km_added"] = float(monorail.length.sum() / 1000.0)

    save_city_gpkg(corrected, output, corrected_metadata, overwrite=overwrite)
    return {
        "prepared_city": str(source),
        "corrected_city": str(output),
        "monorail_features_added": int(len(monorail)),
        "monorail_length_km_added": float(monorail.length.sum() / 1000.0),
        "feature_counts": corrected_metadata["feature_counts"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the corrected Singapore corpus inputs")
    sub = parser.add_subparsers(dest="command", required=True)

    boundary = sub.add_parser("download-boundary")
    boundary.add_argument("--output", required=True, type=Path)
    boundary.add_argument("--dataset-id", default=URA_PLANNING_AREA_DATASET)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--prepared", required=True, type=Path)
    prepare.add_argument("--pbf", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--bbox", nargs=4, type=float, required=True)
    prepare.add_argument("--overwrite", action="store_true")

    prune = sub.add_parser("prune")
    prune.add_argument("--config", required=True, type=Path)
    prune.add_argument("--boundary", required=True, type=Path)
    prune.add_argument("--minimum-coverage", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "download-boundary":
            path = download_data_gov_geojson(args.dataset_id, args.output)
            frame = gpd.read_file(path)
            result = {
                "boundary": str(path),
                "features": len(frame),
                "crs": str(frame.crs),
                "bounds": [float(value) for value in frame.total_bounds],
            }
        elif args.command == "prepare":
            result = prepare_monorail_city(
                args.prepared,
                args.pbf,
                args.output,
                bbox_wgs84=tuple(args.bbox),
                overwrite=args.overwrite,
            )
        else:
            result = prune_corpus_to_boundary(
                args.config,
                args.boundary,
                minimum_coverage=args.minimum_coverage,
            )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
