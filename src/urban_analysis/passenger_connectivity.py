from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from .connectivity import (
    _add_transitions,
    _city_json,
    _metrics,
    _namespace,
    _read_rows,
    _stitch,
    _tile_graph,
    _transition_candidates,
)

_TAG_PAIR = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')
_RAIL_YES = {"yes", "true", "1"}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return "" if text in {"nan", "none"} else text


def _other_tags(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    result: dict[str, str] = {}
    for key, raw in _TAG_PAIR.findall(str(value)):
        result[key.replace(r'\"', '"')] = raw.replace(r'\"', '"')
    return result


def _value(row: Any, key: str) -> str:
    direct = row.get(key)
    if direct is not None and _clean(direct):
        return _clean(direct)
    return _clean(_other_tags(row.get("other_tags")).get(key))


def _display_value(row: Any, key: str) -> str:
    direct = row.get(key)
    if direct is not None and _clean(direct):
        return str(direct).strip()
    value = _other_tags(row.get("other_tags")).get(key)
    return "" if not _clean(value) else str(value).strip()


def _rail_station_kind(row: Any) -> str | None:
    railway = _value(row, "railway")
    public_transport = _value(row, "public_transport")
    station = _value(row, "station")
    rail_flag = any(
        _value(row, key) in _RAIL_YES
        for key in ("train", "subway", "light_rail", "tram")
    )

    if railway in {"station", "halt"}:
        return "station"
    if public_transport == "station" and (rail_flag or station in {"subway", "light_rail"}):
        return "station"
    if railway in {"stop", "tram_stop"}:
        return "stop_position"
    if public_transport == "stop_position" and rail_flag:
        return "stop_position"
    return None


def _normalise_name(value: str) -> str:
    return " ".join("".join(char if char.isalnum() else " " for char in value.casefold()).split())


def extract_station_features(
    pbf_path: str | Path,
    *,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    try:
        import pyogrio
    except ImportError as exc:
        raise RuntimeError("pyogrio is required for station extraction") from exc

    path = Path(pbf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    frames: list[gpd.GeoDataFrame] = []
    for layer in ("points", "multipolygons"):
        try:
            frame = pyogrio.read_dataframe(path, layer=layer, bbox=bbox_wgs84)
        except Exception:
            continue
        if frame.empty:
            continue
        if frame.crs is None:
            frame = frame.set_crs("EPSG:4326")
        elif frame.crs.to_epsg() != 4326:
            frame = frame.to_crs("EPSG:4326")
        frames.append(frame)

    if not frames:
        raise RuntimeError("No OSM point or multipolygon station data could be read from the PBF")

    rows: list[dict[str, Any]] = []
    for frame in frames:
        for index, row in frame.iterrows():
            kind = _rail_station_kind(row)
            if kind is None or row.geometry is None or row.geometry.is_empty:
                continue
            name = _display_value(row, "name")
            source_id = row.get("osm_id")
            if not _clean(source_id):
                source_id = row.get("osm_way_id")
            if not _clean(source_id):
                source_id = index
            rows.append(
                {
                    "source_index": str(source_id),
                    "kind": kind,
                    "name": name,
                    "name_key": _normalise_name(name),
                    "railway": _value(row, "railway"),
                    "public_transport": _value(row, "public_transport"),
                    "station": _value(row, "station"),
                    "geometry": row.geometry.representative_point(),
                }
            )

    if not rows:
        raise RuntimeError("No passenger rail station or stop-position features were found in the PBF")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def group_station_features(
    frame: gpd.GeoDataFrame,
    *,
    named_distance_m: float = 300.0,
    unnamed_distance_m: float = 80.0,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for index, row in frame.sort_values(["name_key", "kind", "source_index"]).iterrows():
        point = row.geometry
        name_key = str(row.get("name_key") or "")
        tolerance = named_distance_m if name_key else unnamed_distance_m
        match = None
        best = float("inf")
        for group in groups:
            if name_key and group["name_key"] != name_key:
                continue
            if not name_key and group["name_key"]:
                continue
            distance = float(point.distance(group["geometry"]))
            if distance <= tolerance and distance < best:
                match = group
                best = distance
        member = {
            "source_index": str(row.get("source_index", index)),
            "kind": str(row.get("kind", "")),
            "name": str(row.get("name", "")),
            "point": point,
        }
        if match is None:
            groups.append(
                {
                    "group_id": f"station_{len(groups):04d}",
                    "name": str(row.get("name", "")),
                    "name_key": name_key,
                    "geometry": point,
                    "members": [member],
                }
            )
        else:
            match["members"].append(member)
            points = [value["point"] for value in match["members"]]
            match["geometry"] = Point(
                float(np.mean([value.x for value in points])),
                float(np.mean([value.y for value in points])),
            )
            if not match["name"] and member["name"]:
                match["name"] = member["name"]
    return groups


def _load_states(manifests: Iterable[Path]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for row, manifest in _read_rows(manifests):
        path = _city_json(row, manifest)
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if "transport_graph" not in state:
            raise ValueError(f"Tile {row.get('tile_id')} has no transport_graph")
        states.append(state)
    return states


def _global_rail_graphs(states: list[dict[str, Any]], transition_tolerance_m: float):
    strict = nx.Graph()
    for state in states:
        strict = nx.compose(strict, _tile_graph(state, "rail"))
    stitched, open_nodes, _, _ = _stitch(strict)
    candidates = _transition_candidates(stitched, transition_tolerance_m)
    assisted = _add_transitions(stitched, candidates)
    return strict, stitched, assisted, open_nodes, candidates


def _rail_edge_records(states: list[dict[str, Any]], graph: nx.Graph) -> list[dict[str, Any]]:
    components = list(nx.connected_components(graph))
    component_of = {node: index for index, values in enumerate(components) for node in values}
    records: list[dict[str, Any]] = []

    for state in states:
        tile_id = str(state["tile"]["tile_id"])
        origin = state["coordinate_system"]["origin_projected"]
        ox, oy = float(origin[0]), float(origin[1])
        for edge in state["transport_graph"].get("edges", []):
            if edge.get("transport_mode") != "rail":
                continue
            coords = edge.get("geometry_local_m", [])
            if len(coords) < 2:
                continue
            left = _namespace(tile_id, str(edge["from_node"]))
            right = _namespace(tile_id, str(edge["to_node"]))
            if left not in graph or right not in graph:
                continue
            line = LineString([(ox + float(p[0]), oy + float(p[1])) for p in coords])
            records.append(
                {
                    "tile_id": tile_id,
                    "left": left,
                    "right": right,
                    "component": component_of[left],
                    "vertical_mode": edge.get("vertical_mode"),
                    "length_m": float(edge.get("length_m", line.length)),
                    "geometry": line,
                }
            )
    return records


def _tile_regions(states: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    tiles: list[tuple[str, Any]] = []
    for state in states:
        tile_id = str(state["tile"]["tile_id"])
        origin = state["coordinate_system"]["origin_projected"]
        bounds = state["coordinate_system"]["local_bounds"]
        tiles.append(
            (
                tile_id,
                box(
                    float(origin[0]) + float(bounds[0]),
                    float(origin[1]) + float(bounds[1]),
                    float(origin[0]) + float(bounds[2]),
                    float(origin[1]) + float(bounds[3]),
                ),
            )
        )

    adjacency = nx.Graph()
    adjacency.add_nodes_from(tile_id for tile_id, _ in tiles)
    for i, (left_id, left) in enumerate(tiles):
        for right_id, right in tiles[i + 1 :]:
            if left.distance(right) <= 0.1:
                adjacency.add_edge(left_id, right_id)

    components = sorted(
        nx.connected_components(adjacency),
        key=lambda values: (-len(values), sorted(values)[0]),
    )
    mapping: dict[str, int] = {}
    regions: list[dict[str, Any]] = []
    shapes = dict(tiles)
    for index, values in enumerate(components, start=1):
        for tile_id in values:
            mapping[tile_id] = index
        regions.append(
            {
                "region": index,
                "tiles": len(values),
                "geometry": unary_union([shapes[tile_id] for tile_id in values]),
            }
        )
    return mapping, regions


def _filter_station_groups(groups: list[dict[str, Any]], study_geometry, radius_m: float):
    return [group for group in groups if group["geometry"].distance(study_geometry) <= radius_m]


def attach_station_groups(
    groups: list[dict[str, Any]],
    edge_records: list[dict[str, Any]],
    graph: nx.Graph,
    *,
    radius_m: float,
) -> list[dict[str, Any]]:
    if not edge_records:
        return []
    lines = [record["geometry"] for record in edge_records]
    tree = STRtree(lines)
    rows: list[dict[str, Any]] = []

    for group in groups:
        by_component: dict[int, dict[str, Any]] = {}
        member_points = [member["point"] for member in group["members"]]
        for point in member_points:
            for raw_index in tree.query(point.buffer(radius_m)):
                index = int(raw_index)
                record = edge_records[index]
                distance = float(point.distance(record["geometry"]))
                if distance > radius_m:
                    continue
                component = int(record["component"])
                current = by_component.get(component)
                if current is None or distance < current["distance_m"]:
                    nearest_node = min(
                        (record["left"], record["right"]),
                        key=lambda node: Point(
                            graph.nodes[node]["position_projected_m"][:2]
                        ).distance(point),
                    )
                    by_component[component] = {
                        "component": component,
                        "distance_m": distance,
                        "node": nearest_node,
                        "tile_id": record["tile_id"],
                        "vertical_mode": record["vertical_mode"],
                    }

        rows.append(
            {
                "group_id": group["group_id"],
                "name": group["name"],
                "member_count": len(group["members"]),
                "station_members": sum(member["kind"] == "station" for member in group["members"]),
                "stop_position_members": sum(
                    member["kind"] == "stop_position" for member in group["members"]
                ),
                "attachments": sorted(by_component.values(), key=lambda value: value["component"]),
            }
        )
    return rows


def _add_station_transfers(
    graph: nx.Graph,
    attachments: list[dict[str, Any]],
    region_of_tile: dict[str, int],
) -> tuple[nx.Graph, int]:
    result = graph.copy()
    transfers = 0
    for row in attachments:
        values = row["attachments"]
        if not values:
            continue
        station = f"passenger:{row['group_id']}"
        regions = {region_of_tile.get(str(value["tile_id"])) for value in values}
        regions.discard(None)
        result.add_node(
            station,
            node_type="passenger_station",
            station_name=row["name"],
            region=next(iter(regions)) if len(regions) == 1 else None,
        )
        seen_nodes: set[str] = set()
        for value in values:
            node = str(value["node"])
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            result.add_edge(station, node, kind="station_access", length_m=0.0)
        if len({value["component"] for value in values}) >= 2:
            transfers += 1
    return result, transfers


def _region_subgraph(graph: nx.Graph, region: int, region_of_tile: dict[str, int]) -> nx.Graph:
    nodes: list[str] = []
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "passenger_station":
            if data.get("region") == region:
                nodes.append(node)
            continue
        tile_id = data.get("tile_id")
        if tile_id is not None and region_of_tile.get(str(tile_id)) == region:
            nodes.append(node)
    return graph.subgraph(nodes).copy()


def audit_passenger_connectivity(
    manifests: Iterable[str | Path],
    pbf_path: str | Path,
    output: str | Path,
    *,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    attachment_radii_m: tuple[float, ...] = (20.0, 40.0, 80.0, 120.0),
    transition_tolerance_m: float = 0.75,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_paths = [Path(value).expanduser().resolve() for value in manifests]
    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_path}")
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    states = _load_states(manifest_paths)
    if not states:
        raise ValueError("No city states loaded")
    crs_values = {state["coordinate_system"]["source_projected_crs"] for state in states}
    if len(crs_values) != 1:
        raise ValueError(f"Passenger audit requires one projected CRS, found {sorted(crs_values)}")
    projected_crs = next(iter(crs_values))

    strict, stitched, assisted, open_nodes, transition_candidates = _global_rail_graphs(
        states, transition_tolerance_m
    )
    edge_records = _rail_edge_records(states, assisted)
    region_of_tile, regions = _tile_regions(states)
    study_geometry = unary_union([region["geometry"] for region in regions])

    station_frame = extract_station_features(pbf_path, bbox_wgs84=bbox_wgs84).to_crs(projected_crs)
    groups = group_station_features(station_frame)
    groups = _filter_station_groups(groups, study_geometry, max(attachment_radii_m))

    station_rows = []
    for group in groups:
        station_rows.append(
            {
                "group_id": group["group_id"],
                "name": group["name"],
                "members": len(group["members"]),
                "station_members": sum(value["kind"] == "station" for value in group["members"]),
                "stop_position_members": sum(
                    value["kind"] == "stop_position" for value in group["members"]
                ),
                "x": float(group["geometry"].x),
                "y": float(group["geometry"].y),
            }
        )

    summary: dict[str, Any] = {
        "analysis_version": 1,
        "tiles": len(states),
        "projected_crs": projected_crs,
        "pbf": str(Path(pbf_path).expanduser().resolve()),
        "bbox_wgs84": list(bbox_wgs84) if bbox_wgs84 else None,
        "station_features": int(len(station_frame)),
        "station_groups_in_study_area": len(groups),
        "track": {
            "strict": _metrics(strict),
            "stitched": _metrics(stitched, open_nodes),
            "transition_assisted": _metrics(assisted, open_nodes),
            "vertical_transition_candidates": len(transition_candidates),
        },
        "regions": [
            {
                "region": region["region"],
                "tiles": region["tiles"],
                "strict": _metrics(_region_subgraph(strict, region["region"], region_of_tile)),
                "stitched": _metrics(_region_subgraph(stitched, region["region"], region_of_tile)),
                "transition_assisted": _metrics(
                    _region_subgraph(assisted, region["region"], region_of_tile)
                ),
            }
            for region in regions
        ],
        "passenger": {},
    }

    all_attachment_rows: list[dict[str, Any]] = []
    for radius in attachment_radii_m:
        attachments = attach_station_groups(groups, edge_records, assisted, radius_m=float(radius))
        passenger_graph, transfer_groups = _add_station_transfers(
            assisted, attachments, region_of_tile
        )
        attached = [row for row in attachments if row["attachments"]]
        distances = [
            value["distance_m"]
            for row in attachments
            for value in row["attachments"]
        ]
        key = f"{float(radius):g}m"
        summary["passenger"][key] = {
            "attachment_radius_m": float(radius),
            "station_groups_attached": len(attached),
            "station_groups_unattached": len(attachments) - len(attached),
            "transfer_station_groups": transfer_groups,
            "mean_attachment_distance_m": float(np.mean(distances)) if distances else None,
            "p90_attachment_distance_m": float(np.quantile(distances, 0.9)) if distances else None,
            "network": _metrics(passenger_graph, open_nodes),
            "regions": [
                {
                    "region": region["region"],
                    **_metrics(_region_subgraph(passenger_graph, region["region"], region_of_tile)),
                }
                for region in regions
            ],
        }
        for row in attachments:
            for value in row["attachments"]:
                all_attachment_rows.append(
                    {
                        "radius_m": float(radius),
                        "group_id": row["group_id"],
                        "name": row["name"],
                        "member_count": row["member_count"],
                        "component": value["component"],
                        "distance_m": value["distance_m"],
                        "tile_id": value["tile_id"],
                        "vertical_mode": value["vertical_mode"],
                        "is_transfer_group": int(len(row["attachments"]) >= 2),
                    }
                )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_path / "station_groups.csv", station_rows)
    write_csv(output_path / "station_attachments.csv", all_attachment_rows)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit passenger rail connectivity from OSM stations")
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--pbf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--attachment-radius", action="append", type=float)
    parser.add_argument("--transition-tolerance-m", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    radii = tuple(args.attachment_radius or [20.0, 40.0, 80.0, 120.0])
    try:
        result = audit_passenger_connectivity(
            args.manifest,
            args.pbf,
            args.output,
            bbox_wgs84=tuple(args.bbox) if args.bbox else None,
            attachment_radii_m=radii,
            transition_tolerance_m=args.transition_tolerance_m,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
