import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from urban_model.semantic_diffusion import (
    SEMANTIC_NAMES,
    SemanticBlockDataset,
    SemanticDiffusionConfig,
    SemanticOutpaintingDataset,
    build_semantic_model,
    check_semantic_data,
    diffusion_loss,
    diffusion_model_input,
    layers_to_semantic,
    model_space_to_semantic,
    sample_semantic,
    semantic_input_channels,
    semantic_to_model_space,
)


def _archive(path: Path, *, west: bool = False, east: bool = False) -> None:
    pixels = 16
    layers = np.zeros((12, pixels, pixels), dtype=np.float32)
    layers[7] = 1
    layers[8, 3:13, 3:13] = 1
    centerlines = np.zeros((3, pixels, pixels), dtype=np.uint8)
    if east:
        layers[3, 8, 5:] = 1
        centerlines[2, 8, 5:] = 1
    if west:
        layers[3, 8, :11] = 1
        centerlines[2, 8, :11] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        layers=layers,
        height_known_mask=(layers[8] > 0).astype(np.uint8),
        height_confidence=np.where(layers[8] > 0, 3, 0).astype(np.uint8),
        landuse_known_mask=np.ones((pixels, pixels), dtype=np.uint8),
        road_centerlines=centerlines,
        valid_data_mask=np.ones((pixels, pixels), dtype=np.uint8),
    )


def _manifest(tmp_path: Path, count: int = 2) -> Path:
    manifest = tmp_path / "manifests" / "train.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for column in range(count):
        archive = tmp_path / "tiles" / f"tile-{column}" / "layers.npz"
        _archive(archive, east=column < count - 1, west=column > 0)
        rows.append(
            {
                "tile_id": f"tile-{column}",
                "city_id": "demo",
                "area_id": "centre",
                "column": column,
                "row": 0,
                "sample_path": os.path.relpath(archive, manifest.parent),
            }
        )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


def _config(tmp_path: Path, manifest: Path, task: str = "block") -> SemanticDiffusionConfig:
    return SemanticDiffusionConfig(
        task=task,
        train_manifest=manifest,
        validation_manifest=manifest,
        output_dir=tmp_path / "runs",
        resolution=(16, 16) if task == "block" else (16, 32),
        crop_size_pixels=8,
        crop_stride_pixels=4,
        directions=("east",),
        boundary_width=2,
        guide_length=4,
        block_out_channels=(8, 16),
        layers_per_block=1,
        attention_levels=(False, False),
        norm_num_groups=4,
        diffusion_steps=8,
        inference_steps=4,
        epochs=1,
        batch_size=1,
        max_train_steps_per_epoch=1,
        max_validation_steps=1,
    )


def test_semantic_priority_and_round_trip():
    layers = torch.zeros(12, 8, 8)
    layers[7, 2:6, 2:6] = 1
    layers[0, 0:2, :] = 1
    layers[8, 3:5, 3:5] = 1
    layers[3, 4, :] = 1
    layers[11, 4, 4] = 1
    classes = layers_to_semantic(layers)
    assert classes[2, 2] == SEMANTIC_NAMES.index("vegetation")
    assert classes[3, 3] == SEMANTIC_NAMES.index("building")
    assert classes[4, 3] == SEMANTIC_NAMES.index("road")
    assert classes[4, 4] == SEMANTIC_NAMES.index("rail")
    assert classes[0, 0] == SEMANTIC_NAMES.index("water")
    encoded = semantic_to_model_space(classes)
    assert encoded.shape == (len(SEMANTIC_NAMES), 8, 8)
    assert torch.equal(model_space_to_semantic(encoded), classes)


def test_block_dataset_creates_dense_crops(tmp_path):
    manifest = _manifest(tmp_path, count=1)
    dataset = SemanticBlockDataset(_config(tmp_path, manifest), manifest, augment=False)
    assert len(dataset) == 9
    sample = dataset[0]
    assert sample["x0"].shape == (len(SEMANTIC_NAMES), 16, 16)


def test_outpainting_dataset_builds_mask_and_guide(tmp_path):
    manifest = _manifest(tmp_path)
    config = _config(tmp_path, manifest, task="outpaint")
    dataset = SemanticOutpaintingDataset(config, manifest, augment=False)
    sample = dataset[0]
    assert sample["x0"].shape == (len(SEMANTIC_NAMES), 16, 32)
    assert sample["known_mask"][:, :, :16].min() == 1
    assert sample["known_mask"][:, :, 16:].max() == 0
    assert sample["road_guide"].shape == (3, 16, 32)


def test_masked_diffusion_loss_ignores_known_region():
    prediction = torch.zeros(1, len(SEMANTIC_NAMES), 4, 8)
    target = torch.zeros_like(prediction)
    mask = torch.zeros(1, 1, 4, 8)
    mask[:, :, :, :4] = 1
    prediction[:, :, :, :4] = 100
    assert diffusion_loss(prediction, target, known_mask=mask) == 0
    prediction[:, :, :, 4:] = 1
    assert torch.isclose(
        diffusion_loss(prediction, target, known_mask=mask),
        torch.tensor(1.0),
    )


def test_input_channel_contract():
    assert semantic_input_channels("block") == len(SEMANTIC_NAMES)
    assert semantic_input_channels("outpaint") == len(SEMANTIC_NAMES) + 4
    noisy = torch.zeros(1, len(SEMANTIC_NAMES), 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    guide = torch.zeros(1, 3, 8, 8)
    assert diffusion_model_input(noisy, "block").shape == noisy.shape
    assert diffusion_model_input(
        noisy,
        "outpaint",
        known_mask=mask,
        road_guide=guide,
    ).shape[1] == len(SEMANTIC_NAMES) + 4


def test_check_semantic_data_reports_counts(tmp_path):
    manifest = _manifest(tmp_path, count=1)
    result = check_semantic_data(_config(tmp_path, manifest), samples_per_split=2)
    assert result["task"] == "block"
    assert result["train"]["samples"] == 9
    assert result["train"]["sample_shapes"][0] == [len(SEMANTIC_NAMES), 16, 16]


def test_diffusers_model_shapes_and_known_region_sampling(tmp_path):
    pytest.importorskip("diffusers")
    block_config = _config(tmp_path, _manifest(tmp_path, count=1))
    block_model = build_semantic_model(block_config)
    noisy = torch.randn(1, len(SEMANTIC_NAMES), 16, 16)
    output = block_model(noisy, torch.tensor([1])).sample
    assert output.shape == noisy.shape

    outpaint_config = _config(tmp_path, _manifest(tmp_path), task="outpaint")
    outpaint_model = build_semantic_model(outpaint_config)
    known = torch.randn(1, len(SEMANTIC_NAMES), 16, 32)
    mask = torch.zeros(1, 1, 16, 32)
    mask[:, :, :, :16] = 1
    guide = torch.zeros(1, 3, 16, 32)
    generated = sample_semantic(
        outpaint_model,
        outpaint_config,
        batch_size=1,
        device=torch.device("cpu"),
        seed=2,
        known_x0=known,
        known_mask=mask,
        road_guide=guide,
    )
    assert torch.equal(generated[:, :, :, :16], known[:, :, :, :16])
