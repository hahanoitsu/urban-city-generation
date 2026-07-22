from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from urban_dataset.preview import render_preview

from .checkpoint import load_checkpoint
from .config import TrainingConfig
from .data import make_boundary_guide
from .model import build_model
from .reconstruct import reconstruct_layers
from .runtime import select_device


def _turns_to_east(direction: str) -> int:
    try:
        return {"east": 0, "west": 2, "north": 3, "south": 1}[direction]
    except KeyError as exc:
        raise ValueError("direction must be east, west, north or south") from exc


def _combined(seed: torch.Tensor, generated: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "east":
        return torch.cat([seed, generated], dim=-1)
    if direction == "west":
        return torch.cat([generated, seed], dim=-1)
    if direction == "north":
        return torch.cat([generated, seed], dim=-2)
    if direction == "south":
        return torch.cat([seed, generated], dim=-2)
    raise ValueError(f"Unknown direction: {direction}")


def extend_tile(
    config: TrainingConfig,
    checkpoint_path: str | Path,
    seed_archive: str | Path,
    destination: str | Path,
    *,
    direction: str = "east",
    device_name: str | None = None,
) -> dict[str, str]:
    if config.data.task != "extension":
        raise ValueError("The selected config is not an extension config")
    if config.model.input_channels != 16:
        raise ValueError("The extension model expects 16 input channels")

    archive_path = Path(seed_archive).expanduser().resolve()
    output_dir = Path(destination).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(archive_path, allow_pickle=False) as archive:
        layers = torch.from_numpy(archive["layers"].astype(np.float32))
        centerlines = torch.from_numpy(archive["road_centerlines"].astype(np.float32))
    if layers.shape[0] != 12:
        raise ValueError("The seed archive must contain twelve city channels")

    direction = direction.lower()
    turns = _turns_to_east(direction)
    canonical_seed = torch.rot90(layers, turns, dims=(-2, -1)) if turns else layers
    canonical_centerlines = (
        torch.rot90(centerlines, turns, dims=(-2, -1)) if turns else centerlines
    )
    height, width = canonical_seed.shape[-2:]

    known_layers = torch.zeros((12, height, width * 2), dtype=canonical_seed.dtype)
    known_layers[:, :, :width] = canonical_seed
    known_mask = torch.zeros((1, height, width * 2), dtype=canonical_seed.dtype)
    known_mask[:, :, :width] = 1
    guide = make_boundary_guide(
        canonical_centerlines,
        boundary_width=config.data.boundary_width,
        guide_length=config.data.guide_length,
    )
    model_input = torch.cat([known_layers, known_mask, guide], dim=0).unsqueeze(0)

    device = select_device(device_name or config.run.device)
    model = build_model(config.model).to(device)
    load_checkpoint(checkpoint_path, model=model, device=device)
    model.eval()
    with torch.inference_mode():
        outputs = model(model_input.to(device))
        predicted = reconstruct_layers(outputs)[0].cpu()
        predicted_centerlines = (torch.sigmoid(outputs["centerline_logits"])[0] >= 0.5).cpu()

    composed = known_layers + predicted * (1.0 - known_mask)
    canonical_generated = composed[:, :, width:]
    canonical_generated_centerlines = predicted_centerlines[:, :, width:]
    inverse_turns = (-turns) % 4
    generated = (
        torch.rot90(canonical_generated, inverse_turns, dims=(-2, -1))
        if inverse_turns
        else canonical_generated
    )
    generated_centerlines = (
        torch.rot90(canonical_generated_centerlines, inverse_turns, dims=(-2, -1))
        if inverse_turns
        else canonical_generated_centerlines
    )
    combined = _combined(layers, generated, direction)

    npz_path = output_dir / "extension.npz"
    np.savez_compressed(
        npz_path,
        layers=generated.numpy().astype(np.float32),
        road_centerlines=generated_centerlines.numpy().astype(np.uint8),
        combined_layers=combined.numpy().astype(np.float32),
        direction=np.asarray(direction),
    )
    preview_path = output_dir / "preview.png"
    render_preview(combined.numpy()).save(preview_path, optimize=True)
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "seed_archive": str(archive_path),
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "direction": direction,
        "seed_preserved_exactly": True,
        "generated_tile_shape": list(generated.shape),
        "boundary_width": config.data.boundary_width,
        "guide_length": config.data.guide_length,
        "architecture": config.model.architecture,
        "note": "This is a deterministic conditional baseline, not a diffusion sample.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "archive": str(npz_path),
        "preview": str(preview_path),
        "metadata": str(metadata_path),
    }
