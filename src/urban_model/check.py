from __future__ import annotations

from pathlib import Path

from .config import TrainingConfig
from .data import dataset_for_task


def check_data(config: TrainingConfig, *, samples_per_split: int = 4) -> dict[str, object]:
    manifests = {
        "train": config.data.train_manifest,
        "validation": config.data.validation_manifest,
        "test": config.data.test_manifest,
    }
    result: dict[str, object] = {}
    for split, manifest in manifests.items():
        if manifest is None:
            continue
        path = Path(manifest)
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} manifest: {path}")
        dataset = dataset_for_task(
            config.data.task,
            path,
            augment=False,
            directions=config.data.directions,
            boundary_width=config.data.boundary_width,
            guide_length=config.data.guide_length,
            pair_limit=config.data.pair_limit,
        )
        shapes: set[tuple[int, ...]] = set()
        for index in range(min(len(dataset), samples_per_split)):
            sample = dataset[index]
            shapes.add(tuple(sample["input"].shape))
            if not sample["tile_id"]:
                raise ValueError(f"Empty tile ID in {split} manifest")
        result[split] = {
            "manifest": str(path),
            "samples": len(dataset),
            "sample_shapes": sorted(shapes),
            "task": config.data.task,
        }
    return result
