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
from shapely.geometry import GeometryCollection, MultiLineString, MultiPolygon, Polygon, box
from shapely.ops import polygonize, unary_union

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


def _relation_id(frame: gpd.GeoDataFrame) -> pd.Series:
    for name in ("osm_id", "id"):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(float("nan"), index=frame.index)


def _polygonal_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = [part for part in geometry.geoms if isinstance(part, (Polygon, MultiPolygon))]
        if polygons:
            return unary_union(polygons)
        lines = [part for part in geometry.geoms if isinstance(part, MultiLineString)]
        if lines:
            values = list(polygonize(unary_union(lines)))
            return unary_union(values) if values else None
    if isinstance(geometry, MultiLineString):
        values = list(polygonize(geometry))
        return unary_union(values) if values else None
    return None


def _boundary_candidates(
    pbf_path: Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> list[tuple[str, gpd.GeoDataFrame]]:
    wanted = [
        "boundary",
        "admin_level",
        "name",
        "name:en",
        "official_name",
        "short_name",
        "ISO3166-1",
        "ISO3166-1:alpha2",
        "ISO3166-1:alpha3",
        "type",
        "place",
    ]
    result: list[tuple[str, gpd.GeoDataFrame]] = []
    for layer in ("multipolygons", "other_relations", "multilinestrings"):
        try:
            frame = pyogrio.read_dataframe(pbf_path, layer=layer, bbox=bbox_wgs84)
        except Exception:
            continue
        if frame.empty:
            continue
        frame = _expand_other_tags(frame, wanted)
        if frame.crs is None:
            frame = frame.set_crs("EPSG:4326")
        elif frame.crs.to_epsg() != 4326:
            frame = frame.to_crs("EPSG:4326")
        result.append((layer, frame))
    return result


def extract_admin_boundary(
    pbf_path: str | Path,
    bbox_wgs84: tuple[float, float, float, float],
    *,
    name: str,
    admin_level: str = "2",
    relation_id: int | None = None,
    country_code: str | None = None,
) -> gpd.GeoDataFrame:
    path = Path(pbf_path).expanduser().resolve()
    candidates = _boundary_candidates(path, bbox_wgs84)
    matches: list[tuple[int, str, Any, dict[str, Any]]] = []
    expected_name = _normalise(name)
    expected_level = _normalise(admin_level)
    expected_code = _normalise(country_code)

    for layer, frame in candidates:
        ids = _relation_id(frame)
        boundary = frame.get("boundary", pd.Series(index=frame.index, dtype=object)).map(clean_tag)
        levels = frame.get("admin_level", pd.Series(index=frame.index, dtype=object)).map(_normalise)
        names = frame.get("name", pd.Series(index=frame.index, dtype=object)).map(_normalise)
        names_en = frame.get("name:en", pd.Series(index=frame.index, dtype=object)).map(_normalise)
        official = frame.get("official_name", pd.Series(index=frame.index, dtype=object)).map(_normalise)
        short = frame.get("short_name", pd.Series(index=frame.index, dtype=object)).map(_normalise)
        iso = frame.get("ISO3166-1", pd.Series(index=frame.index, dtype=object)).map(_normalise)
        alpha2 = frame.get("ISO3166-1:alpha2", pd.Series(index=frame.index, dtype=object)).map(_normalise)

        for index, row in frame.iterrows():
            score = 0
            rid = ids.loc[index]
            if relation_id is not None and pd.notna(rid) and int(rid) == int(relation_id):
                score += 100
            if expected_code and (iso.loc[index] == expected_code or alpha2.loc[index] == expected_code):
                score += 60
            row_names = {names.loc[index], names_en.loc[index], official.loc[index], short.loc[index]}
            if expected_name in row_names:
                score += 40
            elif expected_name and any(expected_name in value for value in row_names if value):
                score += 20
            if boundary.loc[index] == "administrative":
                score += 10
            if levels.loc[index] == expected_level:
                score += 10
            if score < 40:
                continue

            geometry = _polygonal_geometry(row.geometry)
            if geometry is None or geometry.is_empty:
                continue
            matches.append(
                (
                    score,
                    layer,
                    geometry,
                    {
                        "relation_id": None if pd.isna(rid) else int(rid),
                        "name": str(row.get("name") or ""),
                        "name_en": str(row.get("name:en") or ""),
                        "official_name": str(row.get("official_name") or ""),
                        "admin_level": str(row.get("admin_level") or ""),
                        "boundary": str(row.get("boundary") or ""),
                        "iso": str(row.get("ISO3166-1") or row.get("ISO3166-1:alpha2") or ""),
                    },
                )
            )

    if not matches:
        diagnostics: list[str] = []
        for layer, frame in candidates:
            ids = _relation_id(frame)
            boundary = frame.get("boundary", pd.Series(index=frame.index, dtype=object)).map(clean_tag)
            levels = frame.get("admin_level", pd.Series(index=frame.index, dtype=object)).map(_normalise)
            names = frame.get("name", pd.Series(index=frame.index, dtype=object)).map(_normalise)
            mask = boundary.eq("administrative") | levels.eq(expected_level) | names.str.contains(expected_name, regex=False)
            for index in frame.index[mask][:12]:
                rid = ids.loc[index]
                diagnostics.append(
                    f"{layer}: id={None if pd.isna(rid) else int(rid)} "
                    f"name={frame.loc[index].get('name')!r} "
                    f"admin_level={frame.loc[index].get('admin_level')!r} "
                    f"boundary={frame.loc[index].get('boundary')!r}"
                )
        detail = "\n".join(diagnostics[:20]) or "no administrative candidates exposed by GDAL"
        raise ValueError(
            f"Could not find boundary for {name!r} "
            f"(relation_id={relation_id}, country_code={country_code}).\nCandidates:\n{detail}"
        )

    matches.sort(key=lambda item: item[0], reverse=True)
    best_score = matches[0][0]
    best = [item for item in matches if item[0] == best_score]
    geometry = unary_union([item[2] for item in best])
    metadata = best[0][3]
    return gpd.GeoDataFrame(
        {
            "name": [name],
            "admin_level": [str(admin_level)],
            "relation_id": [metadata["relation_id"]],
            "country_code": [country_code or metadata["iso"]],
            "source_layer": [best[0][1]],
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
    lines = _expand_other_tags(lines, ["railway", "service", "usage", *RAIL_TAGS])
    if lines.crs is None:
        lines = lines.set_crs("EPSG:4326")
    elif lines.crs.to_epsg() != 4326:
        lines = lines.to_crs("EPSG:4326")

    railway = lines.get("railway", pd.Series(index=lines.index, dtype=object)).map(clean_tag)
    result = _repair(lines[railway.eq("monorail")].copy())
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
    boundary_relation_id: int | None = None,
    country_code: str | None = None,
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
        relation_id=boundary_relation_id,
        country_code=country_code,
    )
    boundary = boundary_wgs84.to_crs(metric_crs)
    region = boundary.geometry.iloc[0]

    monorail = extract_monorail(pbf_path, bbox_wgs84).to_crs(metric_crs)
    all_columns = sorted(set(layers.rail.columns) | set(monorail.columns))
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
            layer_name: _clip_frame(rail if layer_name == "rail" else frame, region)
            for layer_name, frame in layers.items()
        }
    )

    corrected_metadata = dict(metadata)
    corrected_metadata["format"] = "urban-city-prepared-v2-corrected"
    corrected_metadata["source_corrections"] = {
        "added_railway_types": ["monorail"],
        "admin_boundary_name": boundary_name,
        "admin_level": str(admin_level),
        "boundary_relation_id": boundary_relation_id,
        "country_code": country_code,
        "boundary_source": Path(pbf_path).name,
    }
    corrected_metadata["feature_counts"] = {
        layer_name: len(frame) for layer_name, frame in corrected.items()
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
        "boundary_relation_id": boundary_relation_id,
        "country_code": country_code,
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
    prepare.add_argument("--boundary-relation-id", type=int)
    prepare.add_argument("--country-code")
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
                boundary_relation_id=args.boundary_relation_id,
                country_code=args.country_code,
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
