from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_dataset.spatial_world import SpatialWorldConfig, build_spatial_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-size", type=float, default=5120.0)
    parser.add_argument("--local-vector-size", type=float, default=2560.0)
    parser.add_argument("--target-size", type=float, default=1024.0)
    parser.add_argument("--stride", type=float, default=1024.0)
    parser.add_argument("--context-cell", type=float, default=512.0)
    parser.add_argument("--minimum-transport", type=float, default=100.0)
    parser.add_argument("--maximum-samples", type=int)
    args = parser.parse_args()

    config = SpatialWorldConfig(
        context_size_m=args.context_size,
        local_vector_size_m=args.local_vector_size,
        target_size_m=args.target_size,
        target_stride_m=args.stride,
        context_cell_m=args.context_cell,
        minimum_transport_length_m=args.minimum_transport,
    )
    summary = build_spatial_world(
        args.city,
        args.output,
        config=config,
        show_progress=True,
        maximum_samples=args.maximum_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
