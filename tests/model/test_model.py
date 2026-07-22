import torch

from urban_model.config import LossConfig, ModelConfig
from urban_model.losses import ReconstructionLoss
from urban_model.model import ReconstructionAutoencoder


def _batch(batch_size=1, pixels=16):
    road = torch.zeros(batch_size, pixels, pixels, dtype=torch.long)
    road[:, 2:14, 7:9] = 3
    landuse = torch.zeros(batch_size, pixels, pixels, dtype=torch.long)
    landuse[:, 2:14, 2:14] = 1
    binary = torch.zeros(batch_size, 3, pixels, pixels)
    binary[:, 1, 5:11, 5:11] = 1
    height = torch.zeros(batch_size, 1, pixels, pixels)
    height[:, :, 5:11, 5:11] = 0.2
    centerline = torch.zeros(batch_size, 3, pixels, pixels)
    centerline[:, 2, 2:14, 8] = 1
    return {
        "input": torch.rand(batch_size, 12, pixels, pixels),
        "road_target": road,
        "landuse_target": landuse,
        "binary_target": binary,
        "height_target": height,
        "centerline_target": centerline,
        "height_confidence": torch.full(
            (batch_size, pixels, pixels), 3, dtype=torch.long
        ),
        "landuse_known_mask": torch.ones(batch_size, pixels, pixels),
        "valid_mask": torch.ones(batch_size, pixels, pixels),
        "tile_id": [f"tile-{index}" for index in range(batch_size)],
    }


def test_model_output_shapes():
    model = ReconstructionAutoencoder(
        ModelConfig(base_channels=4, channel_multipliers=(1, 2))
    )
    batch = _batch()
    latent = model.encode(batch["input"])
    assert latent.shape == (1, 8, 8, 8)
    outputs = model.decode(latent)
    assert outputs["road_logits"].shape == (1, 4, 16, 16)
    assert outputs["landuse_logits"].shape == (1, 6, 16, 16)
    assert outputs["binary_logits"].shape == (1, 3, 16, 16)
    assert outputs["height_logits"].shape == (1, 1, 16, 16)
    assert outputs["centerline_logits"].shape == (1, 3, 16, 16)


def test_loss_backpropagates():
    model = ReconstructionAutoencoder(
        ModelConfig(base_channels=4, channel_multipliers=(1, 2))
    )
    criterion = ReconstructionLoss(LossConfig())
    batch = _batch()
    losses = criterion(model(batch["input"]), batch)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_default_encoder_uses_32_pixel_latent_map():
    model = ReconstructionAutoencoder(
        ModelConfig(base_channels=4, channel_multipliers=(1, 2, 4, 8))
    )
    latent = model.encode(torch.rand(1, 12, 256, 256))
    assert latent.shape == (1, 32, 32, 32)
