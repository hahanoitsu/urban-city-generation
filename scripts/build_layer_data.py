from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_layers.data import TILE, build_city


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=Path, required=True, help="prepared city gpkg from urban_dataset prepare-city")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=float, default=TILE / 2, help="street tile stride in metres")
    args = parser.parse_args()
    summary = build_city(args.city, args.name, args.output, args.stride)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
