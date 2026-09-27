#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
RUN="${CITY_PLAN_RUN:-$MAIN_ROOT/runs/city-planner-overfit-v2}"
PREVIEWS="${CITY_PLAN_PREVIEWS:-$RUN/previews}"
ZIP="${CITY_PLAN_PREVIEW_ZIP:-$MAIN_ROOT/city-planner-overfit-v2-previews.zip}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_GPUS=1 MAXIMUM_SAMPLES="${MAXIMUM_SAMPLES:-32}" BATCH_SIZE="${BATCH_SIZE:-4}" EPOCHS="${EPOCHS:-100}" SAVE_EVERY="${SAVE_EVERY:-20}" CITY_PLAN_RUN="$RUN" bash "$SCRIPT_ROOT/scripts/run_city_planner_overfit.sh"

CITY_PLAN_CHECKPOINT="$RUN/best.pt" CITY_PLAN_PREVIEWS="$PREVIEWS" CITY_PLAN_PREVIEW_ZIP="$ZIP" SAMPLES="${SAMPLES:-6}" SPLIT=all bash "$SCRIPT_ROOT/scripts/run_city_planner_generate.sh"

ls -lh "$ZIP"
