from pathlib import Path

import numpy as np
import pandas as pd
import torch

from urban_model.config import LayeredDiffusionConfig
from urban_model.morphology_control import (
    CONTROLS,
    ControlDataset,
    build_model,
    condition_planes,
    control_stats,
    measure,
    normalise,
)


def test_normalise_uses_training_stats():
    frame = pd.DataFrame(
        {
            "tile_id": ["a", "b", "c"],
            **{
                name: [0.0, 1.0, 2.0]
                for name in CONTROLS
            },
        }
    )
    stats = control_stats(frame)
    values = frame[list(CONTROLS)].to_numpy(dtype=np.float32)
    scaled = normalise(values, stats)

    assert scaled.shape == values.shape
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-6)


class _TinyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "tile_id": "tile-a:0:0",
            "x0": torch.zeros(19, 8, 8),
            "valid_mask": torch.ones(19, 8, 8),
        }


def test_control_dataset_matches_base_tile_id():
    frame = pd.DataFrame(
        {
            "tile_id": ["tile-a"],
            **{name: [1.0] for name in CONTROLS},
        }
    )
    stats = control_stats(frame)
    dataset = ControlDataset(_TinyDataset(), frame, stats)
    item = dataset[0]

    assert item["controls"].shape == (len(CONTROLS),)
    assert torch.isfinite(item["controls"]).all()


def test_model_accepts_control_planes():
    config = LayeredDiffusionConfig(
        train_manifest=Path("train.jsonl"),
        validation_manifest=Path("validation.jsonl"),
        output_dir=Path("runs/test"),
        resolution=(64, 64),
        block_out_channels=(32, 64, 64),
        attention_levels=(False, False, True),
        batch_size=1,
        num_workers=0,
    )
    model = build_model(config)
    noisy = torch.randn(1, 8, 64, 64)
    xy = torch.randn(1, 2, 64, 64)
    controls = condition_planes(torch.randn(1, len(CONTROLS)), 64, 64)
    output = model(
        torch.cat([noisy, xy, controls], dim=1),
        torch.tensor([999]),
    ).sample

    assert output.shape == (1, 8, 64, 64)


def test_measure_reads_simple_surface_map():
    classes = np.zeros((64, 64), dtype=np.uint8)
    classes[:, :16] = 1
    classes[:, 16:32] = 2
    classes[32, :] = 3
    classes[48:, :] = 7

    result = measure(classes)

    assert result["green_coverage"] > 0.20
    assert result["building_coverage"] > 0.20
    assert result["water_coverage"] > 0.20
    assert result["road_length_km_per_km2"] > 0
    assert result["road_major_share"] > 0.99
