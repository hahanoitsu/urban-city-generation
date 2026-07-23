from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .codec import CommandCodecConfig, command_sequence_length
from .config import load_config, resolve_path
from .conversion import city_state_to_program
from .schema import ProgramConfig, STYLE_FIELDS, style_vector

SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def prepare_program_dataset(
    dataset_root: str | Path,
    manifest_root: str | Path,
    output_root: str | Path,
    *,
    program_config: ProgramConfig | None = None,
    maximum_nodes: int = 512,
    maximum_commands: int = 1024,
    minimum_nodes: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    manifest_root = Path(manifest_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    program_config = program_config or ProgramConfig()
    codec_config = CommandCodecConfig(program=program_config, maximum_nodes=maximum_nodes)

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Program dataset output is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "format": "urban-graph-program-corpus",
        "version": "0.2.0",
        "dataset_root": str(dataset_root),
        "manifest_root": str(manifest_root),
        "output_root": str(output_root),
        "codec": codec_config.to_dict(),
        "maximum_commands": int(maximum_commands),
        "minimum_nodes": int(minimum_nodes),
        "splits": {},
    }
    train_styles: list[list[float]] = []
    train_styles_by_city: dict[str, list[list[float]]] = {}

    for split in SPLITS:
        manifest_path = manifest_root / f"{split}.jsonl"
        if not manifest_path.exists():
            continue
        accepted_rows: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        command_counts: list[int] = []
        node_counts: list[int] = []
        edge_counts: list[int] = []

        for row in read_jsonl(manifest_path):
            sample_path = (manifest_root / str(row["sample_path"])).resolve()
            state_path = sample_path.parent / "city.json"
            tile_id = str(row.get("tile_id") or sample_path.parent.name)
            if not state_path.exists():
                rejections.append({"tile_id": tile_id, "reason": "missing_city_state"})
                continue
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                program = city_state_to_program(payload, program_config)
            except Exception as exc:
                rejections.append(
                    {
                        "tile_id": tile_id,
                        "reason": "conversion_error",
                        "detail": str(exc),
                    }
                )
                continue

            node_count = sum(command["op"] in {"root", "add"} for command in program["commands"])
            edge_count = sum(command["op"] in {"add", "connect"} for command in program["commands"])
            sequence_length = command_sequence_length(program)
            reasons: list[str] = []
            if node_count < minimum_nodes:
                reasons.append("too_few_nodes")
            if node_count > maximum_nodes:
                reasons.append("too_many_nodes")
            if sequence_length > maximum_commands:
                reasons.append("too_many_commands")
            if reasons:
                rejections.append(
                    {
                        "tile_id": tile_id,
                        "reason": ";".join(reasons),
                        "nodes": node_count,
                        "edges": edge_count,
                        "commands": sequence_length,
                    }
                )
                continue

            program_path = output_root / "programs" / split / f"{tile_id}.json"
            write_json(program_path, program)
            style = style_vector(program["style"])
            record = {
                "tile_id": tile_id,
                "city_id": program.get("source", {}).get("city_id"),
                "area_id": program.get("source", {}).get("area_id"),
                "split": split,
                "program_path": program_path.relative_to(output_root).as_posix(),
                "source_state_path": state_path.relative_to(dataset_root).as_posix(),
                "nodes": node_count,
                "edges": edge_count,
                "commands": sequence_length,
                "style": style,
            }
            accepted_rows.append(record)
            command_counts.append(sequence_length)
            node_counts.append(node_count)
            edge_counts.append(edge_count)
            if split == "train":
                train_styles.append(style)
                city_key = str(record.get("city_id") or "unknown")
                train_styles_by_city.setdefault(city_key, []).append(style)

        with (output_root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in accepted_rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(output_root / f"{split}-rejected.json", rejections)
        reason_names = sorted(
            {
                reason
                for item in rejections
                for reason in str(item.get("reason", "")).split(";")
                if reason
            }
        )
        summary["splits"][split] = {
            "manifest": str(manifest_path),
            "accepted": len(accepted_rows),
            "rejected": len(rejections),
            "command_count": _percentiles(command_counts),
            "node_count": _percentiles(node_counts),
            "edge_count": _percentiles(edge_counts),
            "rejection_reasons": {
                reason: sum(reason in str(item.get("reason", "")).split(";") for item in rejections)
                for reason in reason_names
            },
        }

    if not train_styles:
        raise RuntimeError("No training graph programs were accepted")
    style_array = np.asarray(train_styles, dtype=np.float64)
    style_mean = style_array.mean(axis=0)
    style_std = style_array.std(axis=0)
    style_std[style_std < 1e-8] = 1.0
    write_json(
        output_root / "style-stats.json",
        {"fields": list(STYLE_FIELDS), "mean": style_mean.tolist(), "std": style_std.tolist()},
    )
    city_styles = {
        city_id: {
            "samples": len(values),
            "style": np.asarray(values, dtype=np.float64).mean(axis=0).tolist(),
        }
        for city_id, values in sorted(train_styles_by_city.items())
    }
    write_json(
        output_root / "city-styles.json",
        {"fields": list(STYLE_FIELDS), "cities": city_styles},
    )
    write_json(output_root / "codec.json", codec_config.to_dict())
    write_json(output_root / "summary.json", summary)
    return summary


def prepare_from_config(config_file: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    config_path, config = load_config(config_file)
    data = config.get("data", {})
    program = config.get("program", {})
    return prepare_program_dataset(
        resolve_path(config_path, data.get("dataset_root", "data/processed/corpus")),
        resolve_path(config_path, data.get("manifest_root", "data/manifests/corpus")),
        resolve_path(config_path, data.get("program_root", "data/programs")),
        program_config=ProgramConfig(
            coordinate_bins=int(program.get("coordinate_bins", 256)),
            width_quantum_m=float(program.get("width_quantum_m", 1.0)),
            maximum_width_m=float(program.get("maximum_width_m", 32.0)),
            simplify_tolerance_m=float(program.get("simplify_tolerance_m", 6.0)),
            layer_min=int(program.get("layer_min", -5)),
            layer_max=int(program.get("layer_max", 5)),
        ),
        maximum_nodes=int(program.get("maximum_nodes", 512)),
        maximum_commands=int(config.get("model", {}).get("maximum_sequence_length", 1024)),
        minimum_nodes=int(program.get("minimum_nodes", 4)),
        overwrite=overwrite,
    )


def check_program_dataset(config_file: str | Path) -> dict[str, Any]:
    config_path, config = load_config(config_file)
    data = config.get("data", {})
    program_root = resolve_path(config_path, data.get("program_root", "data/programs"))
    codec = json.loads((program_root / "codec.json").read_text(encoding="utf-8"))
    model_limit = int(config.get("model", {}).get("maximum_sequence_length", 1024))
    result: dict[str, Any] = {
        "program_root": str(program_root),
        "maximum_nodes": int(codec["maximum_nodes"]),
        "model_sequence_length": model_limit,
        "splits": {},
    }
    for split in SPLITS:
        index_path = program_root / f"{split}.jsonl"
        if not index_path.exists():
            continue
        rows = read_jsonl(index_path)
        maximum_commands = max((int(row["commands"]) for row in rows), default=0)
        result["splits"][split] = {
            "samples": len(rows),
            "maximum_commands": maximum_commands,
            "maximum_nodes": max((int(row["nodes"]) for row in rows), default=0),
            "cities": sorted({str(row.get("city_id")) for row in rows}),
            "fits_model_sequence_length": maximum_commands <= model_limit,
        }
    return result
