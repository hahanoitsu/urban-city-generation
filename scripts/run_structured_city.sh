#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${CONTEXT_DATA:-$MAIN_ROOT/data/context-graph-v1/singapore}"
OUTPUT="${STRUCTURED_RUN:-$MAIN_ROOT/runs/structured-city-v1}"
CACHE="${STRUCTURED_CACHE:-$MAIN_ROOT/data/structured-city-cache-v1}"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SCRIPT_ROOT"

python -m py_compile     src/urban_model/structured_city.py     src/urban_model/structured_city_data.py     src/urban_model/structured_city_diffusion.py     scripts/train_structured_city.py

pytest -q tests/test_structured_city.py

ARGS=(
    --data "$DATA"
    --output "$OUTPUT"
    --epochs "${EPOCHS:-30}"
    --batch-size "${BATCH_SIZE:-1}"
    --nodes "${NODE_SLOTS:-448}"
    --edges "${EDGE_SLOTS:-512}"
    --buildings "${BUILDING_SLOTS:-512}"
    --areas "${AREA_SLOTS:-160}"
    --ports "${PORT_SLOTS:-96}"
    --cache-dir "$CACHE"
    --save-every "${SAVE_EVERY:-5}"
)

if [[ -n "${MAXIMUM_SAMPLES:-}" ]]; then
    ARGS+=(--maximum-samples "$MAXIMUM_SAMPLES")
fi

python scripts/train_structured_city.py "${ARGS[@]}"
