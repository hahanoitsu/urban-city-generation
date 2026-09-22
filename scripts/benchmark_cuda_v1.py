from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.optim import AdamW

from urban_model.config import load_layered_diffusion_config
from urban_model.model import autocast_context
from urban_model.morphology_control import (
    CONTROLS,
    POSITION_CHANNELS,
    build_model,
    cuda_setup,
)


def run_case(config, size: int, *, optimized: bool, steps: int, warmup: int) -> dict:
    device = torch.device("cuda")
    case = replace(config, resolution=(size, size), batch_size=1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(5132)

    model = build_model(case).to(device)
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    if optimized:
        cuda_setup(device)
        model = model.to(memory_format=torch.channels_last)

    try:
        optimizer = AdamW(
            model.parameters(),
            lr=case.learning_rate,
            weight_decay=case.weight_decay,
            fused=optimized,
        )
    except (TypeError, RuntimeError):
        optimizer = AdamW(
            model.parameters(),
            lr=case.learning_rate,
            weight_decay=case.weight_decay,
        )

    channels = 8 + POSITION_CHANNELS + len(CONTROLS)
    x = torch.randn(1, channels, size, size, device=device)
    target = torch.randn(1, 8, size, size, device=device)
    if optimized:
        x = x.contiguous(memory_format=torch.channels_last)
        target = target.contiguous(memory_format=torch.channels_last)
    timestep = torch.tensor([900], device=device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(case, device):
            prediction = model(x, timestep).sample
            loss = (prediction.float() - target.float()).square().mean()
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    result = {
        "resolution": size,
        "optimized": optimized,
        "steps": steps,
        "seconds_per_step": elapsed / steps,
        "steps_per_second": steps / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }

    del x, target, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print(torch.cuda.get_device_name(0), flush=True)
    results = []

    for size in args.sizes:
        modes = (False, True) if size == 256 else (True,)
        for optimized in modes:
            try:
                result = run_case(
                    load_layered_diffusion_config(args.config),
                    size,
                    optimized=optimized,
                    steps=args.steps,
                    warmup=args.warmup,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                result = {
                    "resolution": size,
                    "optimized": optimized,
                    "error": "CUDA out of memory",
                }
            results.append(result)
            print(json.dumps(result), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
