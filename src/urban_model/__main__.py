from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .check import check_data
from .config import load_training_config
from .evaluate import evaluate
from .extend import extend_tile
from .train import train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="urban-train")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-data", help="Check manifests and sample tensors")
    check.add_argument("--config", required=True, type=Path)
    check.add_argument("--samples-per-split", type=int, default=4)

    training = commands.add_parser("train", help="Train the reconstruction model")
    training.add_argument("--config", required=True, type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--epochs", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--device")

    evaluation = commands.add_parser("evaluate", help="Evaluate a checkpoint")
    evaluation.add_argument("--config", required=True, type=Path)
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )
    evaluation.add_argument("--device")

    extension = commands.add_parser("extend", help="Extend one saved tile in a chosen direction")
    extension.add_argument("--config", required=True, type=Path)
    extension.add_argument("--checkpoint", required=True, type=Path)
    extension.add_argument("--seed", required=True, type=Path)
    extension.add_argument("--output", required=True, type=Path)
    extension.add_argument(
        "--direction", choices=["east", "west", "north", "south"], default="east"
    )
    extension.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_training_config(args.config)
        if args.command == "check-data":
            result = check_data(config, samples_per_split=args.samples_per_split)
        elif args.command == "train":
            result = train(
                config,
                resume=args.resume,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device_name=args.device,
            )
        elif args.command == "evaluate":
            result = evaluate(
                config,
                args.checkpoint,
                split=args.split,
                device_name=args.device,
            )
        elif args.command == "extend":
            result = extend_tile(
                config,
                args.checkpoint,
                args.seed,
                args.output,
                direction=args.direction,
                device_name=args.device,
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
