from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .config import TrainingConfig
from .data import ReconstructionDataset
from .losses import ReconstructionLoss
from .metrics import MetricAccumulator
from .model import ReconstructionAutoencoder
from .runtime import move_batch, select_device, worker_count, write_json


def evaluate(
    config: TrainingConfig,
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    device_name: str | None = None,
) -> dict[str, object]:
    manifests = {
        "train": config.data.train_manifest,
        "validation": config.data.validation_manifest,
        "test": config.data.test_manifest,
    }
    manifest = manifests.get(split)
    if manifest is None:
        raise ValueError(f"No manifest configured for split '{split}'")

    device = select_device(device_name or config.run.device)
    dataset = ReconstructionDataset(manifest, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=config.run.batch_size,
        shuffle=False,
        num_workers=worker_count(config.run.num_workers),
    )
    model = ReconstructionAutoencoder(config.model).to(device)
    load_checkpoint(checkpoint_path, model=model, device=device)
    criterion = ReconstructionLoss(config.loss).to(device)
    accumulator = MetricAccumulator(height_scale_m=config.data.height_scale_m)

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(batch["input"])
            losses = criterion(outputs, batch)
            accumulator.update(outputs, batch, losses)

    result = {
        "split": split,
        "samples": len(dataset),
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "metrics": accumulator.compute(),
    }
    write_json(config.run.output_dir / f"evaluation-{split}.json", result)
    return result
