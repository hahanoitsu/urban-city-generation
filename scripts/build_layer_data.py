from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_layers.data import TILE, build_city, build_trunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=Path, required=True, help="prepared city gpkg from urban_dataset prepare-city")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=float, default=TILE / 2, help="street tile stride in metres")
    parser.add_argument("--corridors-from", type=Path, help="only build the corridor tiles again, with the road classes of --roads, and copy the plan from this folder of built layer data")
    parser.add_argument("--roads", nargs="+", default=["major", "secondary"], help="road classes the corridor tiles hold when --corridors-from is given")
    args = parser.parse_args()
    if args.corridors_from is not None:
        summary = build_trunk(args.city, args.name, args.corridors_from, args.output, tuple(args.roads))
    else:
        summary = build_city(args.city, args.name, args.output, args.stride)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
