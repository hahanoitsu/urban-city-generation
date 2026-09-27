#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
RUN="${PLAN_CELL_RUN:-$MAIN_ROOT/runs/plan-cell-graph-overfit-v1}"
CHECKPOINT="${PLAN_CELL_CHECKPOINT:-$RUN/best.pt}"
OUTPUT="${PLAN_CELL_TOPOLOGY_AUDIT:-$RUN/topology-audit}"
ZIP="${PLAN_CELL_TOPOLOGY_AUDIT_ZIP:-$MAIN_ROOT/plan-cell-graph-overfit-v1-topology-audit.zip}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PLAN_CELL_CHECKPOINT="$CHECKPOINT" PLAN_CELL_GENERATIONS="$OUTPUT" PLAN_CELL_GENERATION_ZIP="$ZIP" SAMPLES="${SAMPLES:-6}" bash "$SCRIPT_ROOT/scripts/run_plan_cell_graph_generate.sh"
