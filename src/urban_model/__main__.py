from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_layered_diffusion_config
from .data import check_layered_data
from .train import sample_layered_checkpoint, train_layered_diffusion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urban-model",
        description="Train and sample multilayer 2D urban diffusion models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Inspect multilayer training tensors")
    check.add_argument("--config", default="configs/layered.yaml", type=Path)
    check.add_argument("--samples", type=int, default=4)

    train = commands.add_parser("train", help="Train the multilayer city diffusion model")
    train.add_argument("--config", default="configs/layered.yaml", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--device")
    train.add_argument("--overwrite", action="store_true")

    sample = commands.add_parser("sample", help="Generate new multilayer city layouts")
    sample.add_argument("--config", default="configs/layered.yaml", type=Path)
    sample.add_argument("--checkpoint", required=True, type=Path)
    sample.add_argument("--output", default="runs/layered-samples", type=Path)
    sample.add_argument("--count", type=int, default=8)
    sample.add_argument("--seed", type=int)
    sample.add_argument("--device")
    sample.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_layered_diffusion_config(args.config)
        if args.command == "check":
            result = check_layered_data(config, samples_per_split=args.samples)
        elif args.command == "train":
            result = train_layered_diffusion(
                config,
                device_name=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                resume=args.resume,
                overwrite=args.overwrite,
            )
        elif args.command == "sample":
            result = sample_layered_checkpoint(
                config,
                args.checkpoint,
                args.output,
                count=args.count,
                seed=args.seed,
                device_name=args.device,
                overwrite=args.overwrite,
            )
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
