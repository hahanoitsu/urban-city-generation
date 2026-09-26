from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def node_cell(node, grid, size):
    x, y = [float(value) for value in node["position_local_m"][:2]]
    column = min(grid - 1, max(0, int(x / size * grid)))
    row = min(grid - 1, max(0, int(y / size * grid)))
    return row, column


def assign_slots(nodes, grid, slots, size):
    groups = defaultdict(list)
    for node in nodes:
        groups[node_cell(node, grid, size)].append(node)

    lookup = {}
    overflow = 0
    max_occupancy = 0
    for (row, column), values in groups.items():
        values.sort(
            key=lambda node: (
                float(node["position_local_m"][1]),
                float(node["position_local_m"][0]),
                str(node["id"]),
            )
        )
        max_occupancy = max(max_occupancy, len(values))
        overflow += max(0, len(values) - slots)
        for subslot, node in enumerate(values[:slots]):
            lookup[str(node["id"])] = (
                (row * grid + column) * slots + subslot
            )
    return lookup, overflow, max_occupancy


def evaluate(rows, root, grid, slots, size):
    sample_overflow = 0
    node_overflow = 0
    occupancies = []
    degrees = []
    forward_degrees = []
    duplicate_edge_fractions = []
    edge_lengths = []

    for row in rows:
        with gzip.open(root / row["sample_path"], "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        graph = payload["target"]["transport_graph"]
        lookup, overflow, max_occupancy = assign_slots(
            graph["nodes"],
            grid,
            slots,
            size,
        )
        occupancies.append(max_occupancy)
        node_overflow += overflow
        if overflow:
            sample_overflow += 1

        adjacency = defaultdict(set)
        forward = Counter()
        pairs = []
        for edge in graph["edges"]:
            left = lookup.get(str(edge["from_node"]))
            right = lookup.get(str(edge["to_node"]))
            if left is None or right is None or left == right:
                continue
            pair = tuple(sorted((left, right)))
            pairs.append(pair)
            adjacency[left].add(right)
            adjacency[right].add(left)
            forward[pair[0]] += 1
            edge_lengths.append(float(edge.get("length_m", 0.0)))

        if pairs:
            duplicate_edge_fractions.append(
                (len(pairs) - len(set(pairs))) / len(pairs)
            )
        else:
            duplicate_edge_fractions.append(0.0)

        degrees.extend(len(values) for values in adjacency.values())
        forward_degrees.extend(forward.values())

    return {
        "grid": grid,
        "cell_size_m": size / grid,
        "slots_per_cell": slots,
        "candidate_node_slots": grid * grid * slots,
        "samples_with_node_overflow": sample_overflow,
        "sample_fit_fraction": 1.0 - sample_overflow / max(len(rows), 1),
        "overflow_nodes": node_overflow,
        "max_cell_occupancy": summary(occupancies),
        "degree": summary(degrees) if degrees else {},
        "forward_degree": summary(forward_degrees) if forward_degrees else {},
        "parallel_or_duplicate_edge_fraction": float(
            np.mean(duplicate_edge_fractions)
        ),
        "edge_length_m": summary(edge_lengths) if edge_lengths else {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target-size", type=float, default=1024.0)
    parser.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=[16, 20, 24, 28, 32],
    )
    parser.add_argument(
        "--slots",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (args.data / "samples.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    results = []
    for grid in args.grids:
        for slots in args.slots:
            results.append(
                evaluate(
                    rows,
                    args.data,
                    grid,
                    slots,
                    args.target_size,
                )
            )

    print(
        json.dumps(
            {
                "samples": len(rows),
                "target_size_m": args.target_size,
                "candidates": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
