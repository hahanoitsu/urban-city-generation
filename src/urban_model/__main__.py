from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .check import check_data
from .config import load_training_config
from .evaluate import evaluate
from .extend import extend_tile
from .semantic_diffusion import (
    check_semantic_data,
    load_semantic_diffusion_config,
    sample_blocks_from_checkpoint,
    sample_outpainting_from_checkpoint,
    train_semantic_diffusion,
)
from .semantic_topology import evaluate_semantic_topology
from .train import train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="urban-train")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-data", help="Check manifests and sample tensors")
    check.add_argument("--config", required=True, type=Path)
    check.add_argument("--samples-per-split", type=int, default=4)

    training = commands.add_parser("train", help="Train a reconstruction or extension baseline")
    training.add_argument("--config", required=True, type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--epochs", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--device")

    evaluation = commands.add_parser("evaluate", help="Evaluate a baseline checkpoint")
    evaluation.add_argument("--config", required=True, type=Path)
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )
    evaluation.add_argument("--device")

    extension = commands.add_parser("extend", help="Extend one tile with the old baseline")
    extension.add_argument("--config", required=True, type=Path)
    extension.add_argument("--checkpoint", required=True, type=Path)
    extension.add_argument("--seed", required=True, type=Path)
    extension.add_argument("--output", required=True, type=Path)
    extension.add_argument(
        "--direction", choices=["east", "west", "north", "south"], default="east"
    )
    extension.add_argument("--device")

    semantic_check = commands.add_parser(
        "check-semantic-data",
        help="Check semantic block or outpainting samples",
    )
    semantic_check.add_argument("--config", required=True, type=Path)
    semantic_check.add_argument("--samples-per-split", type=int, default=4)

    semantic_topology = commands.add_parser(
        "evaluate-semantic-topology",
        help="Compare real semantic crops with generated layouts",
    )
    semantic_topology.add_argument("--config", required=True, type=Path)
    semantic_topology.add_argument(
        "--generated",
        required=True,
        action="append",
        type=Path,
        help="Generated semantic-blocks.npy file or its containing directory",
    )
    semantic_topology.add_argument("--output", required=True, type=Path)
    semantic_topology.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
    )
    semantic_topology.add_argument("--max-real-samples", type=int, default=1000)

    semantic_train = commands.add_parser(
        "train-semantic",
        help="Train CityGen-style semantic diffusion",
    )
    semantic_train.add_argument("--config", required=True, type=Path)
    semantic_train.add_argument("--resume", type=Path)
    semantic_train.add_argument("--epochs", type=int)
    semantic_train.add_argument("--batch-size", type=int)
    semantic_train.add_argument("--device")

    block_sample = commands.add_parser(
        "sample-semantic-blocks",
        help="Sample complete semantic city blocks",
    )
    block_sample.add_argument("--config", required=True, type=Path)
    block_sample.add_argument("--checkpoint", required=True, type=Path)
    block_sample.add_argument("--output", required=True, type=Path)
    block_sample.add_argument("--count", type=int, default=4)
    block_sample.add_argument("--sample-seed", type=int)
    block_sample.add_argument("--device")

    outpaint_sample = commands.add_parser(
        "sample-semantic-outpainting",
        help="Outpaint a saved OSM-derived tile",
    )
    outpaint_sample.add_argument("--config", required=True, type=Path)
    outpaint_sample.add_argument("--checkpoint", required=True, type=Path)
    outpaint_sample.add_argument("--seed", required=True, type=Path)
    outpaint_sample.add_argument("--output", required=True, type=Path)
    outpaint_sample.add_argument(
        "--direction", choices=["east", "west", "north", "south"], default="east"
    )
    outpaint_sample.add_argument("--sample-seed", type=int)
    outpaint_sample.add_argument("--device")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {
            "check-semantic-data",
            "evaluate-semantic-topology",
            "train-semantic",
            "sample-semantic-blocks",
            "sample-semantic-outpainting",
        }:
            config = load_semantic_diffusion_config(args.config)
            if args.command == "check-semantic-data":
                result = check_semantic_data(
                    config,
                    samples_per_split=args.samples_per_split,
                )
            elif args.command == "evaluate-semantic-topology":
                result = evaluate_semantic_topology(
                    config,
                    args.generated,
                    args.output,
                    split=args.split,
                    max_real_samples=args.max_real_samples,
                )
            elif args.command == "train-semantic":
                result = train_semantic_diffusion(
                    config,
                    resume=args.resume,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device_name=args.device,
                )
            elif args.command == "sample-semantic-blocks":
                result = sample_blocks_from_checkpoint(
                    config,
                    args.checkpoint,
                    args.output,
                    count=args.count,
                    device_name=args.device,
                    seed=args.sample_seed,
                )
            else:
                result = sample_outpainting_from_checkpoint(
                    config,
                    args.checkpoint,
                    args.seed,
                    args.output,
                    direction=args.direction,
                    device_name=args.device,
                    seed=args.sample_seed,
                )
        else:
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
