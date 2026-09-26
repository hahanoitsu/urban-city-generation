#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${CONTEXT_DATA:-$MAIN_ROOT/data/context-graph-v1/singapore}"
CHECKPOINT="${CONTEXT_CHECKPOINT:-$MAIN_ROOT/runs/context-graph-model-v1/best.pt}"
OUTPUT="${CONTEXT_ROLLOUT:-$MAIN_ROOT/runs/context-graph-model-v1/rollout}"
ZIP="${CONTEXT_ROLLOUT_ZIP:-$MAIN_ROOT/context-graph-model-v1-rollout.zip}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BASE="$HOME/miniconda3"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    CONDA_BASE="$HOME/miniforge3"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BASE="$HOME/anaconda3"
elif [[ -x "$HOME/mambaforge/bin/conda" ]]; then
    CONDA_BASE="$HOME/mambaforge"
elif [[ -x "/opt/conda/bin/conda" ]]; then
    CONDA_BASE="/opt/conda"
else
    echo "conda installation not found" >&2
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_ROOT"

rm -rf "$OUTPUT"
rm -f "$ZIP"

python scripts/evaluate_context_graph_rollout.py \
    --data "$DATA" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --samples "${SAMPLES:-8}"

cd "$(dirname "$OUTPUT")"
zip -qr "$ZIP" "$(basename "$OUTPUT")"
ls -lh "$ZIP"
