#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${CONTEXT_DATA:-$MAIN_ROOT/data/context-graph-v1/singapore}"
CHECKPOINT="${CONTEXT_CHECKPOINT:-$MAIN_ROOT/runs/context-graph-model-v1/best.pt}"
OUTPUT="${CONTEXT_ROLLOUT:-$MAIN_ROOT/runs/context-graph-model-v1/rollout}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_ROOT"

python scripts/evaluate_context_graph_rollout.py     --data "$DATA"     --checkpoint "$CHECKPOINT"     --output "$OUTPUT"     --samples "${SAMPLES:-8}"
