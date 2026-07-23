from __future__ import annotations

from typing import Any

import torch

from .codec import (
    FIELDS,
    OP_ADD,
    OP_BOS,
    OP_CONNECT,
    OP_EOS,
    OP_ROOT,
    CommandCodecConfig,
    class_name,
    classes_for_mode,
    empty_encoded_command,
    mode_index,
    mode_name,
    vertical_index,
    vertical_name,
)
from .model import GraphProgramTransformer
from .schema import GRAPH_PROGRAM_VERSION


def _sample(
    logits: torch.Tensor,
    allowed: list[int] | tuple[int, ...] | None,
    *,
    temperature: float,
    generator: torch.Generator,
) -> int:
    values = logits.float()
    if allowed is not None:
        if not allowed:
            raise ValueError("Cannot sample from an empty allowed set")
        allowed_tensor = torch.tensor(list(allowed), dtype=torch.long, device=values.device)
        values = values[allowed_tensor]
    else:
        allowed_tensor = None
    if temperature <= 0:
        selected = int(torch.argmax(values).item())
    else:
        probabilities = torch.softmax(values / temperature, dim=-1)
        selected = int(torch.multinomial(probabilities, 1, generator=generator).item())
    return int(allowed_tensor[selected].item()) if allowed_tensor is not None else selected


def _stack(commands: list[dict[str, int]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        field: torch.tensor(
            [[command[field] for command in commands]], dtype=torch.long, device=device
        )
        for field in FIELDS
    }


def _free_coordinate(
    x_logits: torch.Tensor,
    y_logits: torch.Tensor,
    occupied: set[tuple[int, int]],
    *,
    coordinate_bins: int,
    temperature: float,
    generator: torch.Generator,
) -> tuple[int, int]:
    for _ in range(24):
        x = _sample(x_logits, None, temperature=temperature, generator=generator)
        y = _sample(y_logits, None, temperature=temperature, generator=generator)
        if (x, y) not in occupied:
            return x, y
    start_x = int(torch.argmax(x_logits).item())
    start_y = int(torch.argmax(y_logits).item())
    for offset in range(coordinate_bins * coordinate_bins):
        x = (start_x + offset) % coordinate_bins
        y = (start_y + offset // coordinate_bins) % coordinate_bins
        if (x, y) not in occupied:
            return x, y
    raise RuntimeError("No free quantised coordinate remains")


def _connect_candidates(
    node_components: list[int],
    edge_pairs: set[tuple[int, int]],
) -> dict[int, list[int]]:
    candidates: dict[int, list[int]] = {}
    for left, component in enumerate(node_components):
        rights = [
            right
            for right, right_component in enumerate(node_components)
            if right != left
            and right_component == component
            and (min(left, right), max(left, right)) not in edge_pairs
        ]
        if rights:
            candidates[left] = rights
    return candidates


@torch.no_grad()
def generate_program(
    model: GraphProgramTransformer,
    style: torch.Tensor,
    *,
    bounds_m: list[float],
    raw_style: dict[str, float] | None = None,
    minimum_nodes: int = 32,
    minimum_edges: int | None = None,
    minimum_surface_road_edges: int | None = None,
    maximum_components: int = 8,
    maximum_commands: int | None = None,
    temperature: float = 0.9,
    seed: int = 5132,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    codec: CommandCodecConfig = model.config.codec
    maximum_commands = min(
        int(maximum_commands or model.config.maximum_sequence_length),
        model.config.maximum_sequence_length,
    )
    if minimum_nodes < 2 or minimum_nodes > codec.maximum_nodes:
        raise ValueError("minimum_nodes must be between 2 and codec.maximum_nodes")
    maximum_components = max(1, min(int(maximum_components), minimum_nodes // 2))
    minimum_edges = int(
        minimum_edges if minimum_edges is not None else max(1, minimum_nodes - maximum_components)
    )
    minimum_surface_road_edges = int(
        minimum_surface_road_edges
        if minimum_surface_road_edges is not None
        else max(1, minimum_nodes // 4)
    )
    if maximum_commands < minimum_nodes + 2:
        raise ValueError("maximum_commands is too small for minimum_nodes and EOS")

    encoded = [empty_encoded_command(OP_BOS)]
    commands: list[dict[str, Any]] = []
    node_components: list[int] = []
    component_signatures: list[tuple[str, str, int]] = []
    component_sizes: list[int] = []
    occupied: set[tuple[int, int]] = set()
    edge_pairs: set[tuple[int, int]] = set()
    model.eval()
    style = style.to(device=device, dtype=torch.float32).reshape(1, -1)

    while len(encoded) < maximum_commands:
        logits = model(_stack(encoded, device), style)
        last = {field: value[0, -1] for field, value in logits.items()}
        node_count = len(node_components)
        component_count = len(component_signatures)
        candidates = _connect_candidates(node_components, edge_pairs)

        edge_count = len(edge_pairs)
        surface_road_edges = sum(
            1
            for left, right in edge_pairs
            if component_signatures[node_components[left]][0] == "road"
            and component_signatures[node_components[left]][1] == "surface"
        )
        complete_components = all(size >= 2 for size in component_sizes)
        can_finish = (
            node_count >= minimum_nodes
            and edge_count >= minimum_edges
            and surface_road_edges >= minimum_surface_road_edges
            and complete_components
        )
        if len(encoded) == maximum_commands - 1:
            if not can_finish:
                raise RuntimeError("Generation exhausted its command budget before the graph was complete")
            op = OP_EOS
        else:
            allowed_ops: list[int] = []
            if can_finish:
                allowed_ops.append(OP_EOS)
            if (
                component_count < maximum_components
                and node_count < codec.maximum_nodes
                and complete_components
                and (component_count == 0 or surface_road_edges >= minimum_surface_road_edges)
            ):
                allowed_ops.append(OP_ROOT)
            if node_count and node_count < codec.maximum_nodes:
                allowed_ops.append(OP_ADD)
            if candidates:
                allowed_ops.append(OP_CONNECT)
            op = _sample(last["op"], allowed_ops, temperature=temperature, generator=generator)

        if op == OP_EOS:
            encoded.append(empty_encoded_command(OP_EOS))
            break

        record = empty_encoded_command(op)
        if op == OP_ROOT:
            x, y = _free_coordinate(
                last["x"],
                last["y"],
                occupied,
                coordinate_bins=codec.program.coordinate_bins,
                temperature=temperature,
                generator=generator,
            )
            if not component_signatures:
                mode = mode_index("road")
                vertical = vertical_index("surface")
                layer = -codec.program.layer_min
            else:
                mode = _sample(last["mode"], None, temperature=temperature, generator=generator)
                vertical = _sample(
                    last["vertical"], None, temperature=temperature, generator=generator
                )
                layer = _sample(last["layer"], None, temperature=temperature, generator=generator)
            signature = (mode_name(mode), vertical_name(vertical), layer)
            component_id = len(component_signatures)
            component_signatures.append(signature)
            component_sizes.append(1)
            node_components.append(component_id)
            occupied.add((x, y))
            record.update(
                {
                    "x": x + 1,
                    "y": y + 1,
                    "mode": mode + 1,
                    "vertical": vertical + 1,
                    "layer": layer + 1,
                }
            )
            commands.append(
                {
                    "op": "root",
                    "node": len(node_components) - 1,
                    "x_bin": x,
                    "y_bin": y,
                    "transport_mode": signature[0],
                    "vertical_mode": signature[1],
                    "layer_bin": signature[2],
                }
            )
        elif op == OP_ADD:
            parent = _sample(
                last["id1"],
                list(range(len(node_components))),
                temperature=temperature,
                generator=generator,
            )
            component_id = node_components[parent]
            mode_name_value, vertical_name_value, layer = component_signatures[component_id]
            edge_class = _sample(
                last["class"],
                classes_for_mode(mode_name_value),
                temperature=temperature,
                generator=generator,
            )
            width = _sample(last["width"], None, temperature=temperature, generator=generator)
            x, y = _free_coordinate(
                last["x"],
                last["y"],
                occupied,
                coordinate_bins=codec.program.coordinate_bins,
                temperature=temperature,
                generator=generator,
            )
            new_node = len(node_components)
            node_components.append(component_id)
            component_sizes[component_id] += 1
            occupied.add((x, y))
            edge_pairs.add((min(parent, new_node), max(parent, new_node)))
            mode = mode_index(mode_name_value)
            vertical = vertical_index(vertical_name_value)
            record.update(
                {
                    "x": x + 1,
                    "y": y + 1,
                    "id1": parent + 1,
                    "mode": mode + 1,
                    "class": edge_class + 1,
                    "width": width + 1,
                    "vertical": vertical + 1,
                    "layer": layer + 1,
                }
            )
            commands.append(
                {
                    "op": "add",
                    "node": new_node,
                    "parent": parent,
                    "x_bin": x,
                    "y_bin": y,
                    "transport_mode": mode_name_value,
                    "class": class_name(edge_class),
                    "width_bin": width + 1,
                    "vertical_mode": vertical_name_value,
                    "layer_bin": layer,
                }
            )
        elif op == OP_CONNECT:
            candidates = _connect_candidates(node_components, edge_pairs)
            left = _sample(
                last["id1"],
                list(candidates),
                temperature=temperature,
                generator=generator,
            )
            right = _sample(
                last["id2"],
                candidates[left],
                temperature=temperature,
                generator=generator,
            )
            component_id = node_components[left]
            mode_name_value, vertical_name_value, layer = component_signatures[component_id]
            edge_class = _sample(
                last["class"],
                classes_for_mode(mode_name_value),
                temperature=temperature,
                generator=generator,
            )
            width = _sample(last["width"], None, temperature=temperature, generator=generator)
            edge_pairs.add((min(left, right), max(left, right)))
            mode = mode_index(mode_name_value)
            vertical = vertical_index(vertical_name_value)
            record.update(
                {
                    "id1": left + 1,
                    "id2": right + 1,
                    "mode": mode + 1,
                    "class": edge_class + 1,
                    "width": width + 1,
                    "vertical": vertical + 1,
                    "layer": layer + 1,
                }
            )
            commands.append(
                {
                    "op": "connect",
                    "from": left,
                    "to": right,
                    "transport_mode": mode_name_value,
                    "class": class_name(edge_class),
                    "width_bin": width + 1,
                    "vertical_mode": vertical_name_value,
                    "layer_bin": layer,
                }
            )
        else:
            raise AssertionError(f"Unhandled operation: {op}")
        encoded.append(record)
    else:
        raise RuntimeError("Generation reached maximum_commands before EOS")

    return {
        "format": "urban-graph-program",
        "version": GRAPH_PROGRAM_VERSION,
        "bounds_m": [float(value) for value in bounds_m],
        "source": {"kind": "generated", "seed": int(seed)},
        "program_config": codec.program.to_dict(),
        "style": dict(raw_style or {}),
        "commands": commands,
    }
