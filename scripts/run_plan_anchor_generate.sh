#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${PLAN_ANCHOR_DATA:-$MAIN_ROOT/data/spatial-world-v1/singapore}"
CHECKPOINT="${PLAN_ANCHOR_CHECKPOINT:-$MAIN_ROOT/runs/plan-anchor-overfit-v1/best.pt}"
OUTPUT="${PLAN_ANCHOR_GENERATIONS:-$MAIN_ROOT/runs/plan-anchor-overfit-v1/generations}"
ZIP="${PLAN_ANCHOR_GENERATION_ZIP:-$MAIN_ROOT/plan-anchor-overfit-v1-generations.zip}"

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
else
    echo "conda installation not found" >&2
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SCRIPT_ROOT"
rm -rf "$OUTPUT"
rm -f "$ZIP"

python scripts/sample_plan_anchor.py     --data "$DATA"     --checkpoint "$CHECKPOINT"     --output "$OUTPUT"     --samples "${SAMPLES:-6}"

cd "$(dirname "$OUTPUT")"
zip -qr "$ZIP" "$(basename "$OUTPUT")"
ls -lh "$ZIP"
