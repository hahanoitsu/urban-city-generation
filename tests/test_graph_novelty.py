from __future__ import annotations

from urban_ai.novelty import edge_jaccard, edge_signature
from urban_ai.schema import ProgramConfig


def sample_program() -> dict:
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


def test_edge_signature_detects_exact_program_and_changes() -> None:
    program = sample_program()
    signature = edge_signature(program)
    assert edge_jaccard(signature, signature) == 1.0
    changed = sample_program()
    changed["commands"][1]["class"] = "local"
    assert edge_jaccard(signature, edge_signature(changed)) < 1.0
