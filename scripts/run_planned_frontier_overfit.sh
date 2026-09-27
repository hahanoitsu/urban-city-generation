#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${PLANNED_FRONTIER_DATA:-$MAIN_ROOT/data/spatial-world-v1/singapore}"
OUTPUT="${PLANNED_FRONTIER_RUN:-$MAIN_ROOT/runs/planned-frontier-overfit-v1}"

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

python -m py_compile     src/urban_model/planned_frontier_data.py     src/urban_model/planned_frontier.py     src/urban_model/planned_frontier_loss.py     scripts/train_planned_frontier.py

pytest -q     tests/test_planned_frontier.py

ARGS=(
    --data "$DATA"
    --output "$OUTPUT"
    --epochs "${EPOCHS:-100}"
    --batch-size "${BATCH_SIZE:-4}"
    --learning-rate "${LEARNING_RATE:-2e-4}"
    --max-steps "${MAX_STEPS:-1024}"
    --grid-size "${GRID_SIZE:-16}"
    --save-every "${SAVE_EVERY:-20}"
    --overfit
)

if [[ -n "${MAXIMUM_SAMPLES:-}" ]]; then
    ARGS+=(--maximum-samples "$MAXIMUM_SAMPLES")
fi

NUM_GPUS="${NUM_GPUS:-1}"
if (( NUM_GPUS > 1 )); then
    torchrun         --standalone         --nproc_per_node="$NUM_GPUS"         scripts/train_planned_frontier.py         "${ARGS[@]}"
else
    python scripts/train_planned_frontier.py "${ARGS[@]}"
fi
