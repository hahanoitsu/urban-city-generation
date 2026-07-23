from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .schema import (
    VERTICAL_Z_M,
    ProgramConfig,
    dequantise,
    dequantise_layer,
    dequantise_width,
    normalise_class,
    normalise_mode,
    normalise_vertical,
)


def _validate_edge_attributes(
    command: dict[str, Any],
    command_index: int,
    config: ProgramConfig,
) -> tuple[str, str, int]:
    mode = normalise_mode(command.get("transport_mode"))
    if mode != command.get("transport_mode"):
        raise ValueError(f"Command {command_index}: invalid transport mode")
    edge_class = normalise_class(command.get("class"), mode)
    if edge_class != command.get("class"):
        raise ValueError(f"Command {command_index}: invalid class for transport mode")
    vertical = normalise_vertical(command.get("vertical_mode"))
    if vertical != command.get("vertical_mode"):
        raise ValueError(f"Command {command_index}: invalid vertical mode")
    width_bin = int(command.get("width_bin", 0))
    maximum_width_bin = max(1, int(round(config.maximum_width_m / config.width_quantum_m)))
    if not 1 <= width_bin <= maximum_width_bin:
        raise ValueError(f"Command {command_index}: width_bin outside configured range")
    layer_bin = int(command.get("layer_bin", -1))
    layer_count = config.layer_max - config.layer_min + 1
    if not 0 <= layer_bin < layer_count:
        raise ValueError(f"Command {command_index}: layer_bin outside configured range")
    return mode, vertical, layer_bin


def validate_program(program: dict[str, Any], config: ProgramConfig | None = None) -> None:
    config = config or ProgramConfig(**program.get("program_config", {}))
    if program.get("format") != "urban-graph-program":
        raise ValueError("Unsupported graph program format")
    node_components: list[int] = []
    component_signatures: list[tuple[str, str, int]] = []

    for command_index, command in enumerate(program.get("commands", [])):
        op = command.get("op")
        if op in {"root", "add"}:
            if int(command.get("node", -1)) != len(node_components):
                raise ValueError(f"Command {command_index}: node index must be sequential")
            coordinate = (int(command.get("x_bin", -1)), int(command.get("y_bin", -1)))
            if not all(0 <= value < config.coordinate_bins for value in coordinate):
                raise ValueError(f"Command {command_index}: coordinate outside configured range")

        if op == "root":
            mode = normalise_mode(command.get("transport_mode"))
            vertical = normalise_vertical(command.get("vertical_mode"))
            layer_bin = int(command.get("layer_bin", -1))
            if mode != command.get("transport_mode"):
                raise ValueError(f"Command {command_index}: invalid root transport mode")
            if vertical != command.get("vertical_mode"):
                raise ValueError(f"Command {command_index}: invalid root vertical mode")
            if not 0 <= layer_bin < config.layer_max - config.layer_min + 1:
                raise ValueError(f"Command {command_index}: invalid root layer")
            component_id = len(component_signatures)
            component_signatures.append((mode, vertical, layer_bin))
            node_components.append(component_id)
            continue

        if op == "add":
            parent = int(command.get("parent", -1))
            if not 0 <= parent < len(node_components):
                raise ValueError(f"Command {command_index}: parent must reference an existing node")
            signature = _validate_edge_attributes(command, command_index, config)
            component_id = node_components[parent]
            if signature != component_signatures[component_id]:
                raise ValueError(
                    f"Command {command_index}: edge mode/level does not match its component"
                )
            node_components.append(component_id)
        elif op == "connect":
            left = int(command.get("from", -1))
            right = int(command.get("to", -1))
            if not (0 <= left < len(node_components) and 0 <= right < len(node_components)):
                raise ValueError(
                    f"Command {command_index}: connection must reference existing nodes"
                )
            if left == right:
                raise ValueError(f"Command {command_index}: self connections are not allowed")
            if node_components[left] != node_components[right]:
                raise ValueError(f"Command {command_index}: cannot connect separate components")
            signature = _validate_edge_attributes(command, command_index, config)
            if signature != component_signatures[node_components[left]]:
                raise ValueError(
                    f"Command {command_index}: edge mode/level does not match its component"
                )
        else:
            raise ValueError(f"Command {command_index}: unsupported operation {op!r}")


def program_to_city_state(
    program: dict[str, Any],
    config: ProgramConfig | None = None,
) -> dict[str, Any]:
    config = config or ProgramConfig(**program.get("program_config", {}))
    validate_program(program, config)
    bounds = [float(value) for value in program.get("bounds_m", [0, 0, 1024, 1024])]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_components: list[int] = []
    component_signatures: list[tuple[str, str, int]] = []

    def position(command: dict[str, Any]) -> tuple[float, float]:
        return (
            dequantise(int(command["x_bin"]), bounds[0], bounds[2], config.coordinate_bins),
            dequantise(int(command["y_bin"]), bounds[1], bounds[3], config.coordinate_bins),
        )

    for command_index, command in enumerate(program.get("commands", [])):
        op = command["op"]
        if op == "root":
            component_id = len(component_signatures)
            signature = (
                str(command["transport_mode"]),
                str(command["vertical_mode"]),
                int(command["layer_bin"]),
            )
            component_signatures.append(signature)
            node_components.append(component_id)
            x, y = position(command)
            nodes.append(
                {
                    "id": f"node_{len(nodes):04d}",
                    "position_local_m": [x, y, VERTICAL_Z_M[signature[1]]],
                    "transport_mode": signature[0],
                    "vertical_mode": signature[1],
                    "layer_order": dequantise_layer(signature[2], config),
                    "generated_by_command": command_index,
                    "requires_vertical_review": signature[1] == "unknown",
                }
            )
            continue

        if op == "add":
            left = int(command["parent"])
            right = len(nodes)
            component_id = node_components[left]
            node_components.append(component_id)
            x, y = position(command)
            signature = component_signatures[component_id]
            nodes.append(
                {
                    "id": f"node_{right:04d}",
                    "position_local_m": [x, y, VERTICAL_Z_M[signature[1]]],
                    "transport_mode": signature[0],
                    "vertical_mode": signature[1],
                    "layer_order": dequantise_layer(signature[2], config),
                    "generated_by_command": command_index,
                    "requires_vertical_review": signature[1] == "unknown",
                }
            )
        else:
            left = int(command["from"])
            right = int(command["to"])

        vertical = str(command["vertical_mode"])
        z = VERTICAL_Z_M[vertical]
        left_position = nodes[left]["position_local_m"]
        right_position = nodes[right]["position_local_m"]
        length = math.hypot(
            float(right_position[0]) - float(left_position[0]),
            float(right_position[1]) - float(left_position[1]),
        )
        edges.append(
            {
                "id": f"edge_{len(edges):04d}",
                "from_node": nodes[left]["id"],
                "to_node": nodes[right]["id"],
                "transport_mode": command["transport_mode"],
                "class": command["class"],
                "vertical_mode": vertical,
                "layer_order": dequantise_layer(int(command["layer_bin"]), config),
                "width_m": dequantise_width(int(command["width_bin"]), config),
                "length_m": length,
                "geometry_local_m": [
                    [float(left_position[0]), float(left_position[1]), z],
                    [float(right_position[0]), float(right_position[1]), z],
                ],
                "generated_by_command": command_index,
                "requires_vertical_review": vertical == "unknown",
            }
        )

    degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        degree[edge["from_node"]] += 1
        degree[edge["to_node"]] += 1
    for node in nodes:
        node["degree"] = degree[node["id"]]
        node["node_type"] = (
            "endpoint" if node["degree"] <= 1 else "intersection" if node["degree"] >= 3 else "continuation"
        )

    return {
        "format": "urban-generated-city",
        "version": "0.1.0",
        "coordinate_system": {
            "units": "metres",
            "axis_convention": "x-east, y-north, z-up",
            "local_bounds": bounds,
        },
        "generation": {
            "kind": "unconditional_graph_program",
            "seed": program.get("source", {}).get("seed"),
            "style": program.get("style", {}),
        },
        "source_program": {
            "format": program.get("format"),
            "version": program.get("version"),
        },
        "transport_graph": {"nodes": nodes, "edges": edges},
        "statistics": {
            "nodes": len(nodes),
            "edges": len(edges),
            "components": len(component_signatures),
            "surface_edges": sum(edge["vertical_mode"] == "surface" for edge in edges),
            "underground_edges": sum(edge["vertical_mode"] == "underground" for edge in edges),
            "elevated_edges": sum(edge["vertical_mode"] == "elevated" for edge in edges),
        },
    }
