from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

from urban_dataset.corpus import build_corpus, load_corpus_config
from urban_dataset.utils import write_json


SPLITS = ("train", "validation", "test")


def key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("city_id", "")),
        str(row.get("area_id", "")),
        str(row["tile_id"]),
    )


def read_canonical(manifest_root: Path) -> dict[tuple[str, str, str], tuple[str, dict]]:
    result = {}
    for split in SPLITS:
        path = manifest_root / f"{split}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                item_key = key(row)
                if item_key in result:
                    raise ValueError(f"Duplicate canonical tile: {item_key}")
                result[item_key] = (split, row)
    return result


def highres_rows(dataset_root: Path, manifest_root: Path) -> dict[tuple[str, str, str], dict]:
    result = {}
    for index_path in sorted(dataset_root.glob("*/index.csv")):
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                tile_dir = index_path.parent / "tiles" / row["tile_id"]
                copy = dict(row)
                copy["dataset_dir"] = Path(
                    os.path.relpath(index_path.parent, manifest_root)
                ).as_posix()
                copy["sample_path"] = Path(
                    os.path.relpath(tile_dir / "layers.npz", manifest_root)
                ).as_posix()
                copy["metadata_path"] = Path(
                    os.path.relpath(tile_dir / "metadata.json", manifest_root)
                ).as_posix()
                item_key = key(copy)
                if item_key in result:
                    raise ValueError(f"Duplicate high-res tile: {item_key}")
                result[item_key] = copy
    return result


def filter_indexes(dataset_root: Path, wanted: set[tuple[str, str, str]]) -> None:
    for index_path in sorted(dataset_root.glob("*/index.csv")):
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = list(reader.fieldnames or [])

        kept = []
        for row in rows:
            if key(row) in wanted:
                kept.append(row)
            else:
                tile_dir = index_path.parent / "tiles" / row["tile_id"]
                if tile_dir.exists():
                    shutil.rmtree(tile_dir)

        with index_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)


def sync_manifests(
    dataset_root: Path,
    manifest_root: Path,
    canonical_root: Path,
) -> dict:
    canonical = read_canonical(canonical_root)
    wanted = set(canonical)

    filter_indexes(dataset_root, wanted)
    rows = highres_rows(dataset_root, manifest_root)

    missing = sorted(wanted - set(rows))
    extra = sorted(set(rows) - wanted)
    if missing:
        raise RuntimeError(
            f"High-res corpus is missing {len(missing)} canonical tiles; first: {missing[:5]}"
        )
    if extra:
        raise RuntimeError(f"Unexpected extra tiles remain after filtering: {extra[:5]}")

    counts = {}
    manifest_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        output = []
        for item_key, (canonical_split, old) in canonical.items():
            if canonical_split != split:
                continue
            row = dict(rows[item_key])
            row["split"] = split
            if "spatial_group" in old:
                row["spatial_group"] = old["spatial_group"]
            output.append(row)

        output.sort(key=lambda value: value["tile_id"])
        path = manifest_root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in output:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(output)

    counts["total"] = sum(counts.values())
    counts["canonical_manifest_root"] = str(canonical_root)
    counts["same_tile_population_and_splits"] = True
    write_json(manifest_root / "manifest_summary.json", counts)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--pixels", type=int, default=1024)
    args = parser.parse_args()

    root = args.source_root.expanduser().resolve()
    base = load_corpus_config(root / "configs/corpus.yaml")

    output_root = root / "data/processed/corpus-v2-1024"
    manifest_root = root / "data/manifests/corpus-v2-1024"
    canonical_root = root / "data/manifests/corpus-v2"
    corrected_city = root / "data/cities/singapore-v2.gpkg"

    if not corrected_city.exists():
        raise FileNotFoundError(corrected_city)
    if not canonical_root.exists():
        raise FileNotFoundError(canonical_root)

    cities = tuple(
        replace(city, gpkg_path=corrected_city)
        for city in base.cities
    )
    raster = replace(base.raster, pixels=args.pixels)
    quality = replace(
        base.quality,
        minimum_buildings=0,
        minimum_road_length_m=0.0,
        minimum_nonempty_fraction=0.0,
        minimum_valid_fraction=0.0,
        reject_water_fraction_above=1.0,
    )
    config = replace(
        base,
        output_root=output_root,
        manifest_root=manifest_root,
        cities=cities,
        raster=raster,
        quality=quality,
        overwrite=True,
        save_tile_vectors=False,
        atlas_limit=80,
    )

    print(f"building native {args.pixels}x{args.pixels} corpus", flush=True)
    print(f"source: {corrected_city}", flush=True)
    build_result = build_corpus(config)

    print("syncing to canonical corpus-v2 tile population and splits", flush=True)
    sync = sync_manifests(output_root, manifest_root, canonical_root)

    result = {
        "pixels": args.pixels,
        "metres_per_pixel": float(base.raster.tile_size_m / args.pixels),
        "output_root": str(output_root),
        "manifest_root": str(manifest_root),
        "build": build_result,
        "canonical_sync": sync,
    }
    write_json(output_root / "highres_summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
