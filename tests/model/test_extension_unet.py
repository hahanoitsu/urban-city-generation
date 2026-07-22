import json
import os
from pathlib import Path

import numpy as np
import torch

from urban_model.config import LossConfig, ModelConfig, load_training_config
from urban_model.data import ExtensionDataset
from urban_model.extend import extend_tile
from urban_model.losses import ReconstructionLoss
from urban_model.metrics import MetricAccumulator
from urban_model.model import ExtensionUNet, build_model
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


def _tiny_config(tmp_path: Path, manifest: Path, *, pair_limit: int | None = None) -> Path:
    path = tmp_path / "configs" / "extension.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    relative = manifest.relative_to(tmp_path)
    pair_line = f"  pair_limit: {pair_limit}\n" if pair_limit is not None else ""
    path.write_text(
        "data:\n"
        "  task: extension\n"
        f"  train_manifest: {relative}\n"
        f"  validation_manifest: {relative}\n"
        "  directions: [east]\n"
        "  boundary_width: 2\n"
        "  guide_length: 4\n"
        f"{pair_line}"
        "model:\n"
        "  architecture: extension_unet\n"
        "  input_channels: 16\n"
        "  base_channels: 4\n"
        "  channel_multipliers: [1, 2]\n"
        "loss:\n"
        "  centerline: 2.0\n"
        "  boundary_centerline: 2.0\n"
        "  centerline_tversky: 0.5\n"
        "run:\n"
        "  output_dir: runs/test-extension-unet\n"
        "  epochs: 1\n"
        "  batch_size: 1\n"
        "  preview_every: 1\n"
        "  checkpoint_every: 1\n"
        "  max_train_steps_per_epoch: 1\n"
        "  max_validation_steps: 1\n"
        "  device: cpu\n"
    )
    return path


def test_extension_unet_preserves_spatial_shape():
    model = ExtensionUNet(
        ModelConfig(
            architecture="extension_unet",
            input_channels=16,
            base_channels=4,
            channel_multipliers=(1, 2, 4),
        )
    )
    outputs = model(torch.randn(2, 16, 16, 32))
    assert outputs["road_logits"].shape == (2, 4, 16, 32)
    assert outputs["centerline_logits"].shape == (2, 3, 16, 32)


def test_model_factory_keeps_autoencoder_available():
    autoencoder = build_model(ModelConfig(architecture="autoencoder", input_channels=12))
    extension = build_model(ModelConfig(architecture="extension_unet", input_channels=16))
    assert autoencoder.__class__.__name__ == "ReconstructionAutoencoder"
    assert extension.__class__.__name__ == "ExtensionUNet"


def test_pair_limit_creates_tiny_overfit_dataset(tmp_path):
    manifest = _manifest(tmp_path, count=4)
    dataset = ExtensionDataset(manifest, directions=["east"], pair_limit=2)
    assert len(dataset) == 2


def test_extension_loss_ignores_seed_half(tmp_path):
    manifest = _manifest(tmp_path, count=2)
    sample = ExtensionDataset(manifest, directions=["east"])[0]
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }
    config = load_training_config(_tiny_config(tmp_path, manifest))
    model = build_model(config.model)
    criterion = ReconstructionLoss(config.loss)
    outputs = model(batch["input"])
    original = criterion(outputs, batch)["total"]
    batch["binary_target"] = batch["binary_target"].clone()
    batch["binary_target"][:, :, :, :16] = 1 - batch["binary_target"][:, :, :, :16]
    changed = criterion(outputs, batch)["total"]
    assert torch.allclose(original, changed)


def test_boundary_loss_rewards_required_continuation(tmp_path):
    manifest = _manifest(tmp_path, count=2)
    sample = ExtensionDataset(manifest, directions=["east"], guide_length=4)[0]
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }
    shape = batch["centerline_target"].shape
    base_outputs = {
        "road_logits": torch.zeros(1, 4, shape[-2], shape[-1]),
        "landuse_logits": torch.zeros(1, 6, shape[-2], shape[-1]),
        "binary_logits": torch.zeros(1, 3, shape[-2], shape[-1]),
        "height_logits": torch.zeros(1, 1, shape[-2], shape[-1]),
        "centerline_logits": torch.full(shape, -8.0),
    }
    criterion = ReconstructionLoss(LossConfig(boundary_centerline=4.0))
    missing = criterion(base_outputs, batch)["total"]
    followed = {key: value.clone() for key, value in base_outputs.items()}
    followed["centerline_logits"][batch["boundary_guide"] > 0.5] = 8.0
    matched = criterion(followed, batch)["total"]
    assert matched < missing


def test_boundary_metric_measures_continuation_depth(tmp_path):
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
    assert metrics["boundary_road_recall"] == 1.0
    assert metrics["boundary_mean_continuation_pixels"] == 4.0


def test_extension_unet_training_and_inference_smoke(tmp_path):
    manifest = _manifest(tmp_path, count=3)
    config = load_training_config(_tiny_config(tmp_path, manifest, pair_limit=2))
    result = train(config)

    assert result["architecture"] == "extension_unet"
    checkpoint = tmp_path / "runs/test-extension-unet/best.pt"
    assert checkpoint.exists()

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
