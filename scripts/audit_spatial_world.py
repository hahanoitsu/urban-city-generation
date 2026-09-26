from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (args.data / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = (
        "ports",
        "visible_roads",
        "visible_rail",
        "nodes",
        "edges",
        "raw_nodes",
        "raw_edges",
        "buildings",
        "green",
        "water",
        "landuse",
    )
    summary = {
        "samples": len(rows),
        "counts": {
            name: stats([int(row[name]) for row in rows])
            for name in names
        },
        "compression": {
            "node_ratio_mean": float(
                np.mean(
                    [
                        row["nodes"] / max(row["raw_nodes"], 1)
                        for row in rows
                    ]
                )
            ),
            "edge_ratio_mean": float(
                np.mean(
                    [
                        row["edges"] / max(row["raw_edges"], 1)
                        for row in rows
                    ]
                )
            ),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
