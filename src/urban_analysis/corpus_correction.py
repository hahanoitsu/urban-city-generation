from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyogrio
from pyproj import CRS
from shapely.geometry import box

from urban_dataset.audit import audit_dataset
from urban_dataset.classify import clean_tag
from urban_dataset.corpus import load_corpus_config
from urban_dataset.extract import RAIL_TAGS, CityLayers, _expand_other_tags
from urban_dataset.manifests import build_manifests_from_values
from urban_dataset.prepared import load_city_gpkg, save_city_gpkg


def _normalise(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    return "" if text in {"nan", "none"} else text


def _repair(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.copy()
    result = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    invalid = ~result.geometry.is_valid
    if invalid.any():
        result.loc[invalid, result.geometry.name] = result.loc[invalid].geometry.make_valid()
    return result[result.geometry.notna() & ~result.geometry.is_empty].copy()


def extract_admin_boundary(
    pbf_path: str | Path,
    bbox_wgs84: tuple[float, float, float, float],
    *,
    name: str,
    admin_level: str = "2",
) -> gpd.GeoDataFrame:
    path = Path(pbf_path).expanduser().resolve()
    frame = pyogrio.read_dataframe(path, layer="multipolygons", bbox=bbox_wgs84)
    frame = _expand_other_tags(frame, ["boundary", "admin_level", "name"])
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    elif frame.crs.to_epsg() != 4326:
        frame = frame.to_crs("EPSG:4326")

    boundary = frame.get("boundary", pd.Series(index=frame.index, dtype=object)).map(clean_tag)
    levels = frame.get("admin_level", pd.Series(index=frame.index, dtype=object)).map(_normalise)
    names = frame.get("name", pd.Series(index=frame.index, dtype=object)).map(_normalise)
    selected = frame[
        boundary.eq("administrative")
        & levels.eq(_normalise(admin_level))
        & names.eq(_normalise(name))
    ].copy()
    selected = _repair(selected)
    if selected.empty:
        raise ValueError(f"Could not find admin_level={admin_level} boundary named {name!r}")

    geometry = selected.geometry.union_all()
    return gpd.GeoDataFrame(
        {
            "name": [name],
            "admin_level": [str(admin_level)],
            "geometry": [geometry],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def extract_monorail(
    pbf_path: str | Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    path = Path(pbf_path).expanduser().resolve()
    lines = pyogrio.read_dataframe(path, layer="lines", bbox=bbox_wgs84)
    lines = _expand_other_tags(
        lines,
        ["railway", "service", "usage", *RAIL_TAGS],
    )
    if lines.crs is None:
        lines = lines.set_crs("EPSG:4326")
    elif lines.crs.to_epsg() != 4326:
        lines = lines.to_crs("EPSG:4326")

    railway = lines.get("railway", pd.Series(index=lines.index, dtype=object)).map(clean_tag)
    result = lines[railway.eq("monorail")].copy()
    result = _repair(result)
    if result.empty:
        raise ValueError("No railway=monorail features were found in the source PBF")
    return result


def _clip_frame(frame: gpd.GeoDataFrame, geometry) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.copy()
    indexes = list(frame.sindex.query(geometry, predicate="intersects"))
    if not indexes:
        return frame.iloc[0:0].copy()
    result = frame.iloc[indexes].copy()
    result["geometry"] = result.geometry.intersection(geometry)
    return _repair(result)


def prepare_corrected_city(
    prepared_city: str | Path,
    pbf_path: str | Path,
    output_city: str | Path,
    boundary_output: str | Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    boundary_name: str,
    admin_level: str = "2",
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(prepared_city).expanduser().resolve()
    output_path = Path(output_city).expanduser().resolve()
    boundary_path = Path(boundary_output).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    if boundary_path.exists() and not overwrite:
        raise FileExistsError(boundary_path)

    layers, metadata = load_city_gpkg(source_path)
    metric_crs = CRS.from_user_input(metadata.get("metric_crs") or layers.roads.crs)

    boundary_wgs84 = extract_admin_boundary(
        pbf_path,
        bbox_wgs84,
        name=boundary_name,
        admin_level=admin_level,
    )
    boundary = boundary_wgs84.to_crs(metric_crs)
    region = boundary.geometry.iloc[0]

    monorail = extract_monorail(pbf_path, bbox_wgs84).to_crs(metric_crs)
    existing_columns = set(layers.rail.columns)
    all_columns = sorted(existing_columns | set(monorail.columns))
    existing = layers.rail.copy()
    extra = monorail.copy()
    for column in all_columns:
        if column not in existing.columns:
            existing[column] = None
        if column not in extra.columns:
            extra[column] = None
    rail = gpd.GeoDataFrame(
        pd.concat([existing[all_columns], extra[all_columns]], ignore_index=True),
        geometry="geometry",
        crs=metric_crs,
    )

    corrected = CityLayers(
        **{
            name: _clip_frame(rail if name == "rail" else frame, region)
            for name, frame in layers.items()
        }
    )

    corrected_metadata = dict(metadata)
    corrected_metadata["format"] = "urban-city-prepared-v2-corrected"
    corrected_metadata["source_corrections"] = {
        "added_railway_types": ["monorail"],
        "admin_boundary_name": boundary_name,
        "admin_level": str(admin_level),
        "boundary_source": Path(pbf_path).name,
    }
    corrected_metadata["feature_counts"] = {
        name: len(frame) for name, frame in corrected.items()
    }
    corrected_metadata["monorail_features_added"] = int(len(monorail))
    corrected_metadata["monorail_length_km_added"] = float(monorail.length.sum() / 1000.0)

    save_city_gpkg(corrected, output_path, corrected_metadata, overwrite=overwrite)
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    if boundary_path.exists():
        boundary_path.unlink()
    boundary_wgs84.to_file(boundary_path, driver="GeoJSON")

    return {
        "prepared_city": str(source_path),
        "corrected_city": str(output_path),
        "boundary": str(boundary_path),
        "monorail_features_added": int(len(monorail)),
        "monorail_length_km_added": float(monorail.length.sum() / 1000.0),
        "feature_counts": corrected_metadata["feature_counts"],
    }


def _rewrite_index(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prune_corpus_to_boundary(
    corpus_config: str | Path,
    boundary_path: str | Path,
    *,
    minimum_coverage: float = 0.995,
) -> dict[str, Any]:
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    config = load_corpus_config(corpus_config)
    boundary = gpd.read_file(Path(boundary_path).expanduser().resolve())
    if boundary.empty:
        raise ValueError("Boundary file is empty")

    removed: list[dict[str, Any]] = []
    kept = 0
    for index_path in sorted(config.output_root.glob("*/index.csv")):
        summary_path = index_path.parent / "dataset_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metric_crs = CRS.from_user_input(summary["metric_crs"])
        region = boundary.to_crs(metric_crs).geometry.union_all()

        with index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        accepted: list[dict[str, str]] = []
        for row in rows:
            tile = box(
                float(row["minx"]),
                float(row["miny"]),
                float(row["maxx"]),
                float(row["maxy"]),
            )
            coverage = float(tile.intersection(region).area / tile.area)
            if coverage + 1e-12 >= minimum_coverage:
                accepted.append(row)
                kept += 1
                continue

            tile_dir = index_path.parent / "tiles" / row["tile_id"]
            if tile_dir.exists():
                shutil.rmtree(tile_dir)
            removed.append(
                {
                    "dataset": index_path.parent.name,
                    "tile_id": row["tile_id"],
                    "boundary_coverage": coverage,
                }
            )

        _rewrite_index(index_path, accepted, fieldnames)

    if config.manifest_root.exists():
        shutil.rmtree(config.manifest_root)
    manifest_summary = build_manifests_from_values(
        config.output_root,
        config.manifest_root,
        seed=config.seed,
        split_ratios=config.split_ratios,
        validation_cities={city.city_id for city in config.cities if city.split == "validation"},
        test_cities={city.city_id for city in config.cities if city.split == "test"},
        spatial_group_tiles=config.spatial_group_tiles,
    )
    audit = audit_dataset(config.output_root, config.output_root / "audit.json")

    removed_path = config.output_root / "country-mask-removed.csv"
    with removed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "tile_id", "boundary_coverage"],
        )
        writer.writeheader()
        writer.writerows(removed)

    result = {
        "minimum_boundary_coverage": minimum_coverage,
        "kept_tiles": kept,
        "removed_tiles": len(removed),
        "removed": removed,
        "manifests": manifest_summary,
        "audit": audit,
    }
    (config.output_root / "country-mask-prune.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Correct source omissions and country masking")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--prepared", required=True, type=Path)
    prepare.add_argument("--pbf", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--boundary-output", required=True, type=Path)
    prepare.add_argument("--boundary-name", required=True)
    prepare.add_argument("--admin-level", default="2")
    prepare.add_argument("--bbox", nargs=4, type=float, required=True)
    prepare.add_argument("--overwrite", action="store_true")

    prune = sub.add_parser("prune")
    prune.add_argument("--config", required=True, type=Path)
    prune.add_argument("--boundary", required=True, type=Path)
    prune.add_argument("--minimum-coverage", type=float, default=0.995)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_corrected_city(
                args.prepared,
                args.pbf,
                args.output,
                args.boundary_output,
                bbox_wgs84=tuple(args.bbox),
                boundary_name=args.boundary_name,
                admin_level=args.admin_level,
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
