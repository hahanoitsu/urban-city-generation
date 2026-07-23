from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import audit_dataset, create_preview_atlas
from .bundle import compile_city_state_bundle
from .config import load_build_config
from .corpus import build_corpus, load_corpus_config
from .demo import run_demo
from .manifests import build_manifests
from .network_scene import build_network_scene
from .obj_export import export_city_state_obj
from .pipeline import run_build
from .prepared import prepare_city


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urban-dataset",
        description="Prepare layered urban data and compile deterministic city scenes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build one study area directly from a PBF")
    build.add_argument("--config", required=True, type=Path)

    prepare = subparsers.add_parser(
        "prepare-city", help="Extract and enrich one city into a reusable GeoPackage"
    )
    prepare.add_argument("--config", required=True, type=Path)

    corpus = subparsers.add_parser(
        "build-corpus", help="Build several study areas from prepared city GeoPackages"
    )
    corpus.add_argument("--config", required=True, type=Path)

    demo = subparsers.add_parser("demo", help="Run the full path on built-in synthetic data")
    demo.add_argument("--output", default="data/demo", type=Path)
    demo.add_argument("--overwrite", action="store_true")

    manifests = subparsers.add_parser(
        "manifests", help="Create leakage-resistant train/validation/test JSONL manifests"
    )
    manifests.add_argument("--config", required=True, type=Path)

    audit = subparsers.add_parser("audit", help="Check tensor integrity and channel coverage")
    audit.add_argument("--dataset", required=True, type=Path)
    audit.add_argument("--output", type=Path)

    atlas = subparsers.add_parser("atlas", help="Create a contact sheet of tile previews")
    atlas.add_argument("--dataset", required=True, type=Path)
    atlas.add_argument("--output", required=True, type=Path)
    atlas.add_argument("--columns", type=int, default=6)
    atlas.add_argument("--limit", type=int, default=120)
    atlas.add_argument("--thumbnail-size", type=int, default=192)

    bundle = subparsers.add_parser(
        "compile-city-state",
        help="Index graph and building tile states for a multi-city 3D/PCG importer",
    )
    bundle.add_argument("--dataset", required=True, type=Path)
    bundle.add_argument("--output", required=True, type=Path)

    obj = subparsers.add_parser(
        "export-obj",
        help="Compile one tile city state into a visible OBJ prototype",
    )
    obj.add_argument("--state", required=True, type=Path)
    obj.add_argument("--output", required=True, type=Path)

    scene = subparsers.add_parser(
        "build-scene",
        help="Stitch tile graphs, derive blocks and parcels, and export a scene",
    )
    scene.add_argument("--dataset", required=True, type=Path)
    scene.add_argument("--output", required=True, type=Path)
    scene.add_argument("--city")
    scene.add_argument("--area")
    scene.add_argument("--max-tiles", type=int)
    scene.add_argument("--stitch-tolerance", type=float, default=0.10)
    scene.add_argument("--minimum-block-area", type=float, default=400.0)
    scene.add_argument("--target-parcel-area", type=float, default=900.0)
    scene.add_argument("--minimum-parcel-area", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            result = run_build(load_build_config(args.config))
        elif args.command == "prepare-city":
            result = prepare_city(load_build_config(args.config))
        elif args.command == "build-corpus":
            result = build_corpus(load_corpus_config(args.config))
        elif args.command == "demo":
            result = run_demo(args.output, overwrite=args.overwrite)
        elif args.command == "manifests":
            result = build_manifests(args.config)
        elif args.command == "audit":
            result = audit_dataset(args.dataset, args.output)
        elif args.command == "atlas":
            path = create_preview_atlas(
                args.dataset,
                args.output,
                columns=args.columns,
                limit=args.limit,
                thumbnail_size=args.thumbnail_size,
            )
            result = {"atlas": str(path)}
        elif args.command == "compile-city-state":
            result = compile_city_state_bundle(args.dataset, args.output)
        elif args.command == "export-obj":
            result = export_city_state_obj(args.state, args.output)
        elif args.command == "build-scene":
            result = build_network_scene(
                args.dataset,
                args.output,
                city_id=args.city,
                area_id=args.area,
                max_tiles=args.max_tiles,
                stitch_tolerance_m=args.stitch_tolerance,
                minimum_block_area_m2=args.minimum_block_area,
                target_parcel_area_m2=args.target_parcel_area,
                minimum_parcel_area_m2=args.minimum_parcel_area,
            )
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
