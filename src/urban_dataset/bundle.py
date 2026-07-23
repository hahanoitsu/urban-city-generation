from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .city_state import CITY_STATE_VERSION
from .utils import write_json


def compile_city_state_bundle(
    dataset_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    state_paths = sorted(dataset_root.glob("**/tiles/*/city.json"))
    if not state_paths:
        raise FileNotFoundError(
            f"No tile city.json files found below {dataset_root}. "
            "Build the corpus with save_tile_vectors: true."
        )

    tiles: list[dict[str, Any]] = []
    city_counts: dict[str, int] = defaultdict(int)
    area_counts: dict[str, int] = defaultdict(int)
    missing_graphs: list[str] = []

    for state_path in state_paths:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        metadata_path = state_path.with_name("metadata.json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        graph = payload.get("transport_graph")
        if not isinstance(graph, dict):
            missing_graphs.append(str(state_path.relative_to(dataset_root)))
            continue

        tile_data = payload.get("tile", {})
        coordinate = payload.get("coordinate_system", {})
        city_id = str(tile_data.get("city_id") or metadata.get("city_id") or "unknown")
        area_id = str(tile_data.get("area_id") or metadata.get("area_id") or "")
        tile_id = str(tile_data.get("tile_id") or metadata.get("tile_id") or state_path.parent.name)
        relative_state = state_path.relative_to(dataset_root).as_posix()

        tiles.append(
            {
                "tile_id": tile_id,
                "city_id": city_id,
                "area_id": area_id or None,
                "state_path": relative_state,
                "origin_projected_m": coordinate.get("origin_projected"),
                "source_projected_crs": coordinate.get("source_projected_crs"),
                "local_bounds_m": coordinate.get("local_bounds"),
                "transport_nodes": int(graph.get("statistics", {}).get("nodes", 0)),
                "transport_edges": int(graph.get("statistics", {}).get("edges", 0)),
                "buildings": len(payload.get("building_solids", [])),
            }
        )
        city_counts[city_id] += 1
        area_counts[f"{city_id}:{area_id or 'unknown'}"] += 1

    if not tiles:
        raise ValueError(
            "Tile vectors were found, but none contain the city-state transport graph. "
            "Rebuild them with the city-state-graph branch."
        )

    result = {
        "format": "urban-city-state-bundle",
        "version": CITY_STATE_VERSION,
        "dataset_root": dataset_root.name,
        "axis_convention": "x-east, y-north, z-up",
        "tiles": tiles,
        "summary": {
            "tiles": len(tiles),
            "cities": dict(sorted(city_counts.items())),
            "areas": dict(sorted(area_counts.items())),
            "transport_nodes": sum(tile["transport_nodes"] for tile in tiles),
            "transport_edges": sum(tile["transport_edges"] for tile in tiles),
            "buildings": sum(tile["buildings"] for tile in tiles),
            "missing_graph_tiles": missing_graphs,
        },
        "notes": {
            "streaming": "Each tile state is independent and can map to an Unreal World Partition cell.",
            "vertical_values": (
                "Underground and elevated z values are procedural defaults until metric elevation "
                "or depth data is supplied."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    result["output"] = str(output)
    return result
