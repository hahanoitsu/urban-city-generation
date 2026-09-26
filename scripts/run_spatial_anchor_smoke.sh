#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
RUN="${SPATIAL_ANCHOR_RUN:-$MAIN_ROOT/runs/spatial-anchor-smoke-v3}"
GENERATIONS="${SPATIAL_ANCHOR_GENERATIONS:-$RUN/generations}"
ZIP="${SPATIAL_ANCHOR_GENERATION_ZIP:-$MAIN_ROOT/spatial-anchor-smoke-v3-generations.zip}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_GPUS=1 MAXIMUM_SAMPLES="${MAXIMUM_SAMPLES:-64}" BATCH_SIZE="${BATCH_SIZE:-1}" EPOCHS="${EPOCHS:-8}" SAVE_EVERY="${SAVE_EVERY:-4}" SPATIAL_ANCHOR_RUN="$RUN" bash "$SCRIPT_ROOT/scripts/run_spatial_anchor_train.sh"

SPATIAL_ANCHOR_CHECKPOINT="$RUN/best.pt" SPATIAL_ANCHOR_GENERATIONS="$GENERATIONS" SPATIAL_ANCHOR_GENERATION_ZIP="$ZIP" SAMPLES="${SAMPLES:-6}" SEEDS="${SEEDS:-3}" TEMPERATURE="${TEMPERATURE:-1.0}" bash "$SCRIPT_ROOT/scripts/run_spatial_anchor_generate.sh"
