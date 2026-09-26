#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${CONTEXT_DATA:-$MAIN_ROOT/data/context-graph-v1/singapore}"
OUTPUT="${CONTEXT_RUN:-$MAIN_ROOT/runs/context-graph-model-v1}"

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

python -m py_compile     src/urban_model/context_graph.py     src/urban_model/context_graph_data.py     scripts/train_context_graph_overfit.py

pytest -q tests/test_context_graph_model.py

rm -rf "$OUTPUT"

python scripts/train_context_graph_overfit.py     --data "$DATA"     --output "$OUTPUT"     --samples "${SAMPLES:-32}"     --epochs "${EPOCHS:-200}"     --batch-size "${BATCH_SIZE:-2}"
