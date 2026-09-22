from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from urban_model.config import load_layered_diffusion_config
from urban_model.morphology_control import train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-descriptors", required=True, type=Path)
    parser.add_argument("--validation-descriptors", required=True, type=Path)
    parser.add_argument("--max-hours", type=float, default=22.0)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--preview-every", type=int, default=25)
    parser.add_argument("--sweep-steps", type=int, default=250)
    args = parser.parse_args()

    root = args.source_root.expanduser().resolve()
    config = load_layered_diffusion_config(
        root / "configs/layered-corpus-v2-1km.yaml"
    )
    config = replace(
        config,
        train_manifest=root / "data/manifests/corpus-v2-1024/train.jsonl",
        validation_manifest=root / "data/manifests/corpus-v2-1024/validation.jsonl",
        output_dir=args.output.expanduser().resolve(),
        resolution=(1024, 1024),
        crop_size_pixels=1024,
        crop_stride_pixels=1024,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        precision="bf16",
        device="cuda",
        vertical_crop_repeat=1,
    )

    train(
        config,
        args.train_descriptors.expanduser().resolve(),
        args.validation_descriptors.expanduser().resolve(),
        args.output.expanduser().resolve(),
        max_hours=args.max_hours,
        max_epochs=args.max_epochs,
        device_name="cuda",
        preview_every=args.preview_every,
        sweep_steps=args.sweep_steps,
        overwrite=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
