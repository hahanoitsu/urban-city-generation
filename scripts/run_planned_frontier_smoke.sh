#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
RUN="${PLANNED_FRONTIER_RUN:-$MAIN_ROOT/runs/planned-frontier-overfit-v1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_GPUS=1 MAXIMUM_SAMPLES="${MAXIMUM_SAMPLES:-32}" BATCH_SIZE="${BATCH_SIZE:-4}" EPOCHS="${EPOCHS:-100}" SAVE_EVERY="${SAVE_EVERY:-20}" LEARNING_RATE="${LEARNING_RATE:-2e-4}" PLANNED_FRONTIER_RUN="$RUN" bash "$SCRIPT_ROOT/scripts/run_planned_frontier_overfit.sh"

for EPOCH in 20 60 100; do
    CHECKPOINT="$RUN/epoch-$(printf '%03d' "$EPOCH").pt"
    if [[ ! -f "$CHECKPOINT" ]]; then
        continue
    fi
    OUTPUT="$RUN/generations-e$(printf '%03d' "$EPOCH")"
    ZIP="$MAIN_ROOT/planned-frontier-overfit-v1-e$(printf '%03d' "$EPOCH").zip"
    PLANNED_FRONTIER_CHECKPOINT="$CHECKPOINT"     PLANNED_FRONTIER_GENERATIONS="$OUTPUT"     PLANNED_FRONTIER_GENERATION_ZIP="$ZIP"     SAMPLES="${MILESTONE_SAMPLES:-3}"     SEEDS=1     TEMPERATURE=0     SPLIT=all     bash "$SCRIPT_ROOT/scripts/run_planned_frontier_generate.sh"
done

PLANNED_FRONTIER_CHECKPOINT="$RUN/best.pt" PLANNED_FRONTIER_GENERATIONS="$RUN/generations-best" PLANNED_FRONTIER_GENERATION_ZIP="$MAIN_ROOT/planned-frontier-overfit-v1-generations.zip" SAMPLES="${SAMPLES:-6}" SEEDS="${SEEDS:-3}" TEMPERATURE="${TEMPERATURE:-0.7}" SPLIT=all bash "$SCRIPT_ROOT/scripts/run_planned_frontier_generate.sh"

ls -lh "$MAIN_ROOT"/planned-frontier-overfit-v1-*.zip
