import numpy as np
import torch
from shapely.geometry import mapping, box

from urban_ai.object3d import (
    CityObject3DDataset,
    Object3DConfig,
    Object3DDenoiser,
    TOKEN_DIM,
    state_tokens,
)
from urban_ai.object3d_scene import decode_tokens


def sample_state():
    return {
        "coordinate_system": {
            "local_bounds": [0.0, 0.0, 1024.0, 1024.0],
        },
        "transport_graph": {
            "edges": [
                {
                    "transport_mode": "road",
                    "class": "major",
                    "vertical_mode": "surface",
                    "width_m": 8.0,
                    "geometry_local_m": [
                        [100.0, 200.0, 0.0],
                        [700.0, 400.0, 0.0],
                    ],
                },
                {
                    "transport_mode": "rail",
                    "class": "subway",
                    "vertical_mode": "underground",
                    "width_m": 6.0,
                    "geometry_local_m": [
                        [200.0, 800.0, -12.0],
                        [900.0, 700.0, -12.0],
                    ],
                },
            ],
        },
        "building_solids": [
            {
                "footprint_local_m": mapping(box(300.0, 300.0, 360.0, 340.0)),
                "base_z_m": 0.0,
                "height_m": 24.0,
            }
        ],
    }


def test_state_tokens_keep_3d_objects():
    values, count, total_count = state_tokens(sample_state(), 16)

    assert values.shape == (16, TOKEN_DIM)
    assert count == 3
    assert total_count == 3

    decoded = decode_tokens(values)
    assert sum(item["type"] == "road" for item in decoded) == 1
    assert sum(item["type"] == "rail" for item in decoded) == 1
    assert sum(item["type"] == "building" for item in decoded) == 1

    rail = next(item for item in decoded if item["type"] == "rail")
    assert rail["vertical_mode"] == "underground"
    assert np.isclose(rail["center_m"][2], -12.0, atol=0.5)


def test_object_model_is_set_shaped():
    config = Object3DConfig(
        maximum_tokens=16,
        model_dimensions=64,
        attention_heads=4,
        layers=2,
        feedforward_dimensions=128,
    )
    model = Object3DDenoiser(config)
    tokens = torch.randn(2, 16, TOKEN_DIM)
    timestep = torch.tensor([0.2, 0.9])
    style = torch.randn(2, 14)

    result = model(tokens, timestep, style)
    assert result.shape == tokens.shape
