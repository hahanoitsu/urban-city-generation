import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sample_spatial_anchor.py"
SPEC = importlib.util.spec_from_file_location("sample_spatial_anchor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
render = MODULE.render


def test_anchor_preview_accepts_tensor_coordinates():
    sample = {
        "context_line_padding": torch.tensor([False]),
        "context_line_points": torch.tensor(
            [[[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]]],
            dtype=torch.float32,
        ),
        "context_line_mode": torch.tensor([0]),
        "port_padding": torch.tensor([False]),
        "ports": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]),
    }
    graph = {
        "nodes": [
            {
                "position_local_m": [256.0, 256.0],
            },
            {
                "position_local_m": [768.0, 768.0],
            },
        ],
        "edges": [
            {
                "mode": "road",
                "geometry_local_m": [
                    [256.0, 256.0],
                    [512.0, 512.0],
                    [768.0, 768.0],
                ],
            }
        ],
    }
    config = SimpleNamespace(
        local_vector_size_m=2560.0,
        target_size_m=1024.0,
    )
    image = render(graph, sample, config, size=128)
    assert image.size == (128, 128)
