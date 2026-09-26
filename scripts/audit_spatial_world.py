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
    limits = {
        "context_lines": 768,
        "ports": 128,
        "nodes": 384,
        "edges": 512,
    }
    fitting = [
        row
        for row in rows
        if row["ports"] <= limits["ports"]
        and row["nodes"] <= limits["nodes"]
        and row["edges"] <= limits["edges"]
    ]
    context_counts = [
        int(row["visible_roads"]) + int(row["visible_rail"])
        for row in rows
    ]
    retained = [
        min(value, limits["context_lines"]) / max(value, 1)
        for value in context_counts
    ]
    summary = {
        "samples": len(rows),
        "counts": {
            name: stats([int(row[name]) for row in rows])
            for name in names
        },
        "limits": limits,
        "samples_fitting_graph_limits": len(fitting),
        "samples_rejected_by_graph_limits": len(rows) - len(fitting),
        "overflow_samples": {
            "ports": sum(row["ports"] > limits["ports"] for row in rows),
            "nodes": sum(row["nodes"] > limits["nodes"] for row in rows),
            "edges": sum(row["edges"] > limits["edges"] for row in rows),
        },
        "context_line_retained_fraction": {
            "mean": float(np.mean(retained)),
            "p10": float(np.percentile(retained, 10)),
            "p50": float(np.percentile(retained, 50)),
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
