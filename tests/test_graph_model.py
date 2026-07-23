from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from urban_ai.codec import CommandCodecConfig, FIELDS, encode_program
from urban_ai.generate import generate_program
from urban_ai.loss import graph_program_loss
from urban_ai.model import GraphProgramTransformer, GraphTransformerConfig
from urban_ai.schema import ProgramConfig
from urban_ai.validation import validate_program


def small_program() -> dict:
    return {
        "format": "urban-graph-program",
        "version": "0.2.0",
        "bounds_m": [0.0, 0.0, 100.0, 100.0],
        "program_config": ProgramConfig(coordinate_bins=16).to_dict(),
        "style": {},
        "commands": [
            {
                "op": "root",
                "node": 0,
                "x_bin": 2,
                "y_bin": 2,
                "transport_mode": "road",
                "vertical_mode": "surface",
                "layer_bin": 5,
            },
            {
                "op": "add",
                "node": 1,
                "parent": 0,
                "x_bin": 8,
                "y_bin": 2,
                "transport_mode": "road",
                "class": "major",
                "width_bin": 10,
                "vertical_mode": "surface",
                "layer_bin": 5,
            },
        ],
    }


def test_model_forward_and_loss() -> None:
    codec = CommandCodecConfig(program=ProgramConfig(coordinate_bins=16), maximum_nodes=16)
    config = GraphTransformerConfig(
        codec=codec,
        style_dimensions=14,
        maximum_sequence_length=16,
        model_dimensions=32,
        attention_heads=4,
        layers=1,
        feedforward_dimensions=64,
        dropout=0.0,
    )
    model = GraphProgramTransformer(config)
    encoded = encode_program(small_program(), codec)
    batch = {field: torch.tensor([encoded[field]], dtype=torch.long) for field in FIELDS}
    inputs = {field: value[:, :-1] for field, value in batch.items()}
    targets = {field: value[:, 1:] for field, value in batch.items()}
    logits = model(inputs, torch.zeros((1, 14)))
    loss, fields = graph_program_loss(logits, targets)
    assert torch.isfinite(loss)
    assert fields["op"] > 0


def test_random_generation_obeys_program_grammar() -> None:
    codec = CommandCodecConfig(program=ProgramConfig(coordinate_bins=16), maximum_nodes=12)
    config = GraphTransformerConfig(
        codec=codec,
        style_dimensions=14,
        maximum_sequence_length=24,
        model_dimensions=32,
        attention_heads=4,
        layers=1,
        feedforward_dimensions=64,
        dropout=0.0,
    )
    model = GraphProgramTransformer(config)
    program = generate_program(
        model,
        torch.zeros(14),
        bounds_m=[0.0, 0.0, 100.0, 100.0],
        minimum_nodes=4,
        maximum_components=2,
        maximum_commands=20,
        temperature=0.8,
        seed=7,
    )
    validate_program(program)
    nodes = [command for command in program["commands"] if command["op"] in {"root", "add"}]
    assert len(nodes) >= 4
    assert len({(command["x_bin"], command["y_bin"]) for command in nodes}) == len(nodes)
