from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import audit_dataset, create_preview_atlas
from .config import load_build_config
from .demo import run_demo
from .manifests import build_manifests
from .pipeline import run_build


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urban-dataset",
        description="Build machine-learning-ready urban morphology tiles from OpenStreetMap PBF data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build one city dataset from a YAML config")
    build.add_argument("--config", required=True, type=Path)

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            result = run_build(load_build_config(args.config))
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
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
