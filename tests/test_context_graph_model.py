import json
from pathlib import Path

import torch

from urban_ai.codec import FIELDS, CommandCodecConfig
from urban_ai.schema import ProgramConfig
from urban_model.context_graph import ContextGraphModelConfig, ContextGraphProgramModel


def test_context_graph_model_forward():
    codec = CommandCodecConfig(program=ProgramConfig(), maximum_nodes=64)
    config = ContextGraphModelConfig(
        codec=codec,
        context_dimensions=18,
        port_dimensions=21,
        relation_count=3,
        model_dimensions=64,
        attention_heads=4,
        context_layers=2,
        decoder_layers=2,
        feedforward_dimensions=128,
        maximum_sequence_length=32,
    )
    model = ContextGraphProgramModel(config)
    commands = {
        field: torch.zeros((2, 16), dtype=torch.long)
        for field in FIELDS
    }
    commands["op"][:, 0] = 1
    context = torch.randn(2, 12, 18)
    relations = torch.eye(12).reshape(1, 1, 12, 12).expand(2, 3, -1, -1)
    ports = torch.randn(2, 8, 21)
    padding = torch.zeros(2, 8, dtype=torch.bool)
    output = model(commands, context, relations, ports, padding)
    assert output["op"].shape == (2, 16, 6)
    assert output["x"].shape == (2, 16, codec.program.coordinate_bins)
