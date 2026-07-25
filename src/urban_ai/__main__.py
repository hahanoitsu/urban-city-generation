from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .conversion import city_state_to_program
from .prepare import check_program_dataset, prepare_from_config, write_json
from .scene import compile_generated_city, export_generated_city_obj, render_generated_city
from .schema import ProgramConfig
from .validation import program_to_city_state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="urban-ai",
        description="Train and sample JSON-first generative urban graph models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="Convert city.json states into complete graph-program examples"
    )
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--overwrite", action="store_true")

    check = commands.add_parser("check", help="Inspect the prepared graph-program corpus")
    check.add_argument("--config", required=True, type=Path)

    roundtrip = commands.add_parser(
        "roundtrip", help="Convert one real city state through the graph-program representation"
    )
    roundtrip.add_argument("--state", required=True, type=Path)
    roundtrip.add_argument("--output", required=True, type=Path)

    train = commands.add_parser("train", help="Train the complete-graph Transformer")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--device")
    train.add_argument("--resume", type=Path)
    train.add_argument("--overwrite", action="store_true")

    sample = commands.add_parser(
        "sample", help="Generate complete fictional city graphs from style controls"
    )
    sample.add_argument("--config", required=True, type=Path)
    sample.add_argument("--checkpoint", required=True, type=Path)
    sample.add_argument("--output", required=True, type=Path)
    sample.add_argument("--count", type=int)
    sample.add_argument("--mix", action="append", default=[])
    sample.add_argument("--set", dest="overrides", action="append", default=[])
    sample.add_argument("--seed", type=int)
    sample.add_argument("--temperature", type=float)
    sample.add_argument("--device")
    sample.add_argument("--overwrite", action="store_true")
    return parser


def _roundtrip(state_path: Path, output: Path) -> dict:
    payload = json.loads(state_path.expanduser().read_text(encoding="utf-8"))
    program = city_state_to_program(payload, ProgramConfig())
    city = compile_generated_city(program_to_city_state(program), seed=5132)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "program.json", program)
    write_json(output / "city.json", city)
    preview = render_generated_city(city, output / "preview.png")
    obj = export_generated_city_obj(city, output / "city.obj")
    return {
        "program": str(output / "program.json"),
        "city": str(output / "city.json"),
        "preview": preview["preview"],
        "obj": obj["obj"],
        "statistics": city.get("statistics", {}),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_from_config(args.config, overwrite=args.overwrite)
        elif args.command == "check":
            result = check_program_dataset(args.config)
        elif args.command == "roundtrip":
            result = _roundtrip(args.state, args.output)
        elif args.command == "train":
            from .training import train_from_config

            result = train_from_config(
                args.config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device_name=args.device,
                resume=args.resume,
                overwrite=args.overwrite,
            )
        elif args.command == "sample":
            from .sampling import parse_assignments, sample_from_config

            result = sample_from_config(
                args.config,
                args.checkpoint,
                args.output,
                count=args.count,
                mix=parse_assignments(args.mix),
                overrides=parse_assignments(args.overrides),
                seed=args.seed,
                temperature=args.temperature,
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
