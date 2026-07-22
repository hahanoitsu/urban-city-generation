import json
import os
from pathlib import Path

import numpy as np
import torch

from urban_model.config import load_training_config
from urban_model.data import ExtensionDataset
from urban_model.extend import extend_tile
from urban_model.losses import ReconstructionLoss
from urban_model.metrics import MetricAccumulator
from urban_model.model import ReconstructionAutoencoder
from urban_model.train import train


def _archive(path: Path, *, road_at_east: bool = False, road_at_west: bool = False) -> None:
    pixels = 16
    layers = np.zeros((12, pixels, pixels), dtype=np.float32)
    layers[4] = 1
    layers[8, 4:12, 4:12] = 1
    layers[9, 4:12, 4:12] = 0.2
    centerlines = np.zeros((3, pixels, pixels), dtype=np.uint8)
    if road_at_east:
        layers[3, 8, 4:] = 1
        centerlines[2, 8, 4:] = 1
    if road_at_west:
        layers[3, 8, :12] = 1
        centerlines[2, 8, :12] = 1
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


def _manifest(tmp_path: Path, count: int = 3) -> Path:
    manifest = tmp_path / "manifests" / "train.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = []
    for column in range(count):
        archive = tmp_path / "tiles" / f"tile-{column}" / "layers.npz"
        _archive(
            archive,
            road_at_east=column < count - 1,
            road_at_west=column > 0,
        )
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


def test_extension_dataset_builds_canonical_pair(tmp_path):
    manifest = _manifest(tmp_path, count=2)
    dataset = ExtensionDataset(manifest, directions=["east"], boundary_width=2, guide_length=4)

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["input"].shape == (16, 16, 32)
    assert sample["known_mask"][:, :, :16].min() == 1
    assert sample["known_mask"][:, :, 16:].max() == 0
    assert sample["valid_mask"][:, :16].max() == 0
    assert sample["valid_mask"][:, 16:].min() == 1
    assert sample["boundary_guide"][2, 8, 16:20].min() == 1


def test_extension_loss_ignores_seed_half(tmp_path):
    manifest = _manifest(tmp_path, count=2)
    sample = ExtensionDataset(manifest, directions=["east"])[0]
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }
    model = ReconstructionAutoencoder(
        load_training_config(_tiny_config(tmp_path, manifest)).model
    )
    criterion = ReconstructionLoss(load_training_config(_tiny_config(tmp_path, manifest)).loss)
    outputs = model(batch["input"])
    original = criterion(outputs, batch)["total"]
    batch["binary_target"] = batch["binary_target"].clone()
    batch["binary_target"][:, :, :, :16] = 1 - batch["binary_target"][:, :, :, :16]
    changed = criterion(outputs, batch)["total"]
    assert torch.allclose(original, changed)


def test_boundary_metric_reports_matched_crossing(tmp_path):
    manifest = _manifest(tmp_path, count=2)
    sample = ExtensionDataset(manifest, directions=["east"], guide_length=4)[0]
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }
    height, width = sample["valid_mask"].shape
    outputs = {
        "road_logits": torch.zeros(1, 4, height, width),
        "landuse_logits": torch.zeros(1, 6, height, width),
        "binary_logits": torch.zeros(1, 3, height, width),
        "height_logits": torch.zeros(1, 1, height, width),
        "centerline_logits": torch.full((1, 3, height, width), -10.0),
    }
    outputs["centerline_logits"][0, 2, 8, 16:20] = 10
    accumulator = MetricAccumulator()
    accumulator.update(outputs, batch, {"total": torch.tensor(0.0)})
    metrics = accumulator.compute()
    assert metrics["boundary_road_crossings"] == 1
    assert metrics["boundary_road_recall"] == 1.0


def _tiny_config(tmp_path: Path, manifest: Path) -> Path:
    path = tmp_path / "configs" / "extension.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    relative = manifest.relative_to(tmp_path)
    path.write_text(
        "data:\n"
        "  task: extension\n"
        f"  train_manifest: {relative}\n"
        f"  validation_manifest: {relative}\n"
        "  directions: [east]\n"
        "  boundary_width: 2\n"
        "  guide_length: 4\n"
        "model:\n"
        "  input_channels: 16\n"
        "  base_channels: 4\n"
        "  channel_multipliers: [1, 2]\n"
        "run:\n"
        "  output_dir: runs/test-extension\n"
        "  epochs: 1\n"
        "  batch_size: 1\n"
        "  preview_every: 1\n"
        "  checkpoint_every: 1\n"
        "  max_train_steps_per_epoch: 1\n"
        "  max_validation_steps: 1\n"
        "  device: cpu\n"
    )
    return path


def test_extension_training_smoke(tmp_path):
    manifest = _manifest(tmp_path, count=3)
    config = load_training_config(_tiny_config(tmp_path, manifest))
    result = train(config)

    assert result["epochs_completed"] == 1
    assert result["train_samples"] == 2
    checkpoint = tmp_path / "runs/test-extension/best.pt"
    assert checkpoint.exists()
    assert (tmp_path / "runs/test-extension/previews/epoch-0001.png").exists()

    extension = extend_tile(
        config,
        checkpoint,
        tmp_path / "tiles/tile-0/layers.npz",
        tmp_path / "generated",
        direction="east",
        device_name="cpu",
    )
    with np.load(extension["archive"], allow_pickle=False) as archive:
        assert archive["layers"].shape == (12, 16, 16)
        assert archive["combined_layers"].shape == (12, 16, 32)
    assert Path(extension["preview"]).exists()
