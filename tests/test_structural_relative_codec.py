from __future__ import annotations

import pytest

from urban_ai.codec import CommandCodecConfig, encode_program
from urban_ai.schema import ProgramConfig


def _program():
    return {
        "format": "urban-graph-program",
        "version": "0.2.0",
        "bounds_m": [0.0, 0.0, 1024.0, 1024.0],
        "program_config": {
            "coordinate_bins": 256,
            "relative_add_coordinates": True,
            "maximum_segment_length_m": 200.0,
        },
        "commands": [
            {
                "op": "root",
                "node": 0,
                "x_bin": 100,
                "y_bin": 120,
                "transport_mode": "road",
                "vertical_mode": "surface",
                "layer_bin": 5,
            },
            {
                "op": "add",
                "node": 1,
                "parent": 0,
                "x_bin": 112,
                "y_bin": 113,
                "transport_mode": "road",
                "class": "secondary",
                "width_bin": 12,
                "vertical_mode": "surface",
                "layer_bin": 5,
            },
        ],
    }


def test_relative_add_coordinates_encode_parent_displacement():
    program = _program()
    config = CommandCodecConfig(
        program=ProgramConfig(
            coordinate_bins=256,
            relative_add_coordinates=True,
            maximum_segment_length_m=200.0,
        )
    )

    encoded = encode_program(program, config)
    center = (256 - 1) // 2

    # BOS, root, add, EOS. Root remains absolute; ADD becomes dx/dy.
    assert encoded["x"][1] == 101
    assert encoded["y"][1] == 121
    assert encoded["x"][2] == center + 12 + 1
    assert encoded["y"][2] == center - 7 + 1


def test_relative_add_coordinate_range_is_checked():
    program = _program()
    program["commands"][1]["x_bin"] = 255
    program["commands"][1]["parent"] = 0
    config = CommandCodecConfig(
        program=ProgramConfig(coordinate_bins=256, relative_add_coordinates=True)
    )

    with pytest.raises(ValueError, match="Relative coordinate delta"):
        encode_program(program, config)
