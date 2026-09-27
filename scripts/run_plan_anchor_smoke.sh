#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
RUN="${PLAN_ANCHOR_RUN:-$MAIN_ROOT/runs/plan-anchor-overfit-v1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_GPUS=1 MAXIMUM_SAMPLES="${MAXIMUM_SAMPLES:-32}" BATCH_SIZE="${BATCH_SIZE:-2}" EPOCHS="${EPOCHS:-100}" SAVE_EVERY=20 PLAN_ANCHOR_RUN="$RUN" bash "$SCRIPT_ROOT/scripts/run_plan_anchor_overfit.sh"

for EPOCH in 20 60 100; do
    CHECKPOINT="$RUN/epoch-$(printf '%03d' "$EPOCH").pt"
    if [[ ! -f "$CHECKPOINT" ]]; then
        continue
    fi
    OUTPUT="$RUN/generations-e$(printf '%03d' "$EPOCH")"
    ZIP="$MAIN_ROOT/plan-anchor-overfit-v1-e$(printf '%03d' "$EPOCH").zip"
    PLAN_ANCHOR_CHECKPOINT="$CHECKPOINT"     PLAN_ANCHOR_GENERATIONS="$OUTPUT"     PLAN_ANCHOR_GENERATION_ZIP="$ZIP"     SAMPLES="${MILESTONE_SAMPLES:-3}"     bash "$SCRIPT_ROOT/scripts/run_plan_anchor_generate.sh"
done

PLAN_ANCHOR_CHECKPOINT="$RUN/best.pt" PLAN_ANCHOR_GENERATIONS="$RUN/generations-best" PLAN_ANCHOR_GENERATION_ZIP="$MAIN_ROOT/plan-anchor-overfit-v1-generations.zip" SAMPLES="${SAMPLES:-6}" bash "$SCRIPT_ROOT/scripts/run_plan_anchor_generate.sh"

ls -lh "$MAIN_ROOT"/plan-anchor-overfit-v1-*.zip
