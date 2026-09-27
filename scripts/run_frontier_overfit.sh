#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
RUN="${FRONTIER_RUN:-$MAIN_ROOT/runs/frontier-overfit-v1}"
FINAL_OUTPUT="${FRONTIER_GENERATIONS:-$RUN/generations-final}"
FINAL_ZIP="${FRONTIER_GENERATION_ZIP:-$MAIN_ROOT/frontier-overfit-v1-generations.zip}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_GPUS=1 OVERFIT=1 MAXIMUM_SAMPLES="${MAXIMUM_SAMPLES:-32}" BATCH_SIZE="${BATCH_SIZE:-4}" EPOCHS="${EPOCHS:-100}" SAVE_EVERY="${SAVE_EVERY:-20}" LEARNING_RATE="${LEARNING_RATE:-2e-4}" FRONTIER_RUN="$RUN" bash "$SCRIPT_ROOT/scripts/run_frontier_train.sh"

FRONTIER_CHECKPOINT="$RUN/best.pt" FRONTIER_GENERATIONS="$FINAL_OUTPUT" FRONTIER_GENERATION_ZIP="$FINAL_ZIP" SPLIT=all SAMPLES="${SAMPLES:-6}" SEEDS="${SEEDS:-3}" TEMPERATURE="${TEMPERATURE:-0.7}" bash "$SCRIPT_ROOT/scripts/run_frontier_generate.sh"

ls -lh "$FINAL_ZIP"
