#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MAXIMUM_SAMPLES="${MAXIMUM_SAMPLES:-32}" EPOCHS="${PLANNER_EPOCHS:-100}" BATCH_SIZE="${PLANNER_BATCH_SIZE:-4}" bash "$SCRIPT_ROOT/scripts/run_city_planner_smoke.sh"

MAXIMUM_SAMPLES="${MAXIMUM_SAMPLES:-32}" EPOCHS="${FRONTIER_EPOCHS:-100}" BATCH_SIZE="${FRONTIER_BATCH_SIZE:-4}" bash "$SCRIPT_ROOT/scripts/run_planned_frontier_smoke.sh"
