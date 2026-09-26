from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_model.structured_city_data import SceneTensorConfig, StructuredCityDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=448)
    parser.add_argument("--edges", type=int, default=512)
    parser.add_argument("--buildings", type=int, default=512)
    parser.add_argument("--areas", type=int, default=160)
    parser.add_argument("--ports", type=int, default=96)
    args = parser.parse_args()

    config = SceneTensorConfig(
        node_slots=args.nodes,
        edge_slots=args.edges,
        building_slots=args.buildings,
        area_slots=args.areas,
        maximum_ports=args.ports,
    )
    dataset = StructuredCityDataset(
        args.data,
        config=config,
        cache_dir=args.cache_dir,
    )
    summary = {
        "samples": len(dataset),
        "source_rows": dataset.total_rows,
        "accepted": dataset.accepted_before_limit,
        "rejected": dataset.rejected,
        "cache_hits": dataset.cache_hits,
        "cache_built": dataset.cache_built,
        "cache_dir": str(dataset.cache_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
