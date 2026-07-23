from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_config, resolve_path
from .generate import generate_program
from .model import GraphProgramTransformer, GraphTransformerConfig
from .novelty import nearest_training_overlap
from .prepare import write_json
from .scene import compile_generated_city, export_generated_city_obj, render_generated_city
from .schema import STYLE_FIELDS, style_from_vector
from .validation import program_to_city_state, validate_program


def parse_assignments(values: list[str] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected name=value, got {value!r}")
        name, raw = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing name in assignment: {value!r}")
        result[name] = float(raw)
    return result


def _raw_style(
    program_root: Path,
    *,
    mix: dict[str, float] | None,
    overrides: dict[str, float] | None,
) -> tuple[dict[str, float], list[float], list[float]]:
    style_stats = json.loads((program_root / "style-stats.json").read_text(encoding="utf-8"))
    mean = [float(value) for value in style_stats["mean"]]
    std = [float(value) for value in style_stats["std"]]
    raw = np.asarray(mean, dtype=np.float64)

    if mix:
        city_payload = json.loads(
            (program_root / "city-styles.json").read_text(encoding="utf-8")
        )
        cities = city_payload.get("cities", {})
        total = sum(max(0.0, float(weight)) for weight in mix.values())
        if total <= 0:
            raise ValueError("Style mixture weights must contain a positive value")
        raw = np.zeros(len(STYLE_FIELDS), dtype=np.float64)
        for city, weight in mix.items():
            if city not in cities:
                available = ", ".join(sorted(cities)) or "none"
                raise ValueError(f"Unknown city style {city!r}; available: {available}")
            raw += np.asarray(cities[city]["style"], dtype=np.float64) * max(0.0, weight) / total

    style = style_from_vector(raw.tolist())
    for name, value in (overrides or {}).items():
        if name not in STYLE_FIELDS:
            raise ValueError(f"Unknown style field {name!r}")
        style[name] = float(value)
    raw_values = [style[name] for name in STYLE_FIELDS]
    normalized = [
        (value - mean_value) / std_value
        for value, mean_value, std_value in zip(raw_values, mean, std, strict=True)
    ]
    return style, normalized, mean


def sample_from_config(
    config_file: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    *,
    count: int | None = None,
    mix: dict[str, float] | None = None,
    overrides: dict[str, float] | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    device_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_path, config = load_config(config_file)
    data = config.get("data", {})
    sampling = config.get("sampling", {})
    program_root = resolve_path(config_path, data.get("program_root", "data/programs"))
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Sample output is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if device_name is None or device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = GraphTransformerConfig.from_dict(checkpoint["model_config"])
    model = GraphProgramTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    raw_style, normalized_style, _mean = _raw_style(
        program_root, mix=mix, overrides=overrides
    )
    style_tensor = torch.tensor(normalized_style, dtype=torch.float32, device=device)
    count = int(count or sampling.get("count", 4))
    seed = int(seed if seed is not None else sampling.get("seed", 5132))
    temperature = float(
        temperature if temperature is not None else sampling.get("temperature", 0.9)
    )
    bounds = [float(value) for value in sampling.get("bounds_m", [0, 0, 1024, 1024])]
    results: list[dict[str, Any]] = []

    for index in range(count):
        sample_seed = seed + index
        sample_dir = output_root / f"sample-{index + 1:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        program = generate_program(
            model,
            style_tensor,
            bounds_m=bounds,
            raw_style=raw_style,
            minimum_nodes=int(sampling.get("minimum_nodes", 32)),
            maximum_components=int(sampling.get("maximum_components", 8)),
            maximum_commands=int(
                sampling.get("maximum_commands", model_config.maximum_sequence_length)
            ),
            temperature=temperature,
            seed=sample_seed,
        )
        validate_program(program)
        city = program_to_city_state(program)
        city = compile_generated_city(
            city,
            seed=sample_seed,
            minimum_block_area_m2=float(sampling.get("minimum_block_area_m2", 400.0)),
            target_parcel_area_m2=float(sampling.get("target_parcel_area_m2", 1200.0)),
            minimum_parcel_area_m2=float(sampling.get("minimum_parcel_area_m2", 160.0)),
        )
        write_json(sample_dir / "program.json", program)
        write_json(sample_dir / "city.json", city)
        preview = render_generated_city(city, sample_dir / "preview.png")
        obj = export_generated_city_obj(city, sample_dir / "city.obj")
        novelty = nearest_training_overlap(program, program_root)
        result = {
            "index": index,
            "seed": sample_seed,
            "program": str(sample_dir / "program.json"),
            "city": str(sample_dir / "city.json"),
            "preview": preview["preview"],
            "obj": obj["obj"],
            "statistics": city.get("statistics", {}),
            "nearest_training_overlap": novelty,
        }
        results.append(result)

    summary = {
        "checkpoint": str(checkpoint_path),
        "output_root": str(output_root),
        "device": str(device),
        "style": raw_style,
        "mix": mix or {},
        "temperature": temperature,
        "samples": results,
    }
    write_json(output_root / "summary.json", summary)
    return summary
