#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${FRONTIER_DATA:-$MAIN_ROOT/data/spatial-world-v1/singapore}"
OUTPUT="${FRONTIER_RUN:-$MAIN_ROOT/runs/frontier-architect-v1}"

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

python -m py_compile     src/urban_model/frontier_data.py     src/urban_model/frontier_architect.py     src/urban_model/frontier_loss.py     scripts/train_frontier_architect.py

pytest -q     tests/test_frontier_program.py     tests/test_frontier_architect.py

ARGS=(
    --data "$DATA"
    --output "$OUTPUT"
    --epochs "${EPOCHS:-100}"
    --batch-size "${BATCH_SIZE:-2}"
    --learning-rate "${LEARNING_RATE:-2e-4}"
    --max-steps "${MAX_STEPS:-1024}"
    --save-every "${SAVE_EVERY:-20}"
)

if [[ -n "${MAXIMUM_SAMPLES:-}" ]]; then
    ARGS+=(--maximum-samples "$MAXIMUM_SAMPLES")
fi

if [[ "${OVERFIT:-0}" == "1" ]]; then
    ARGS+=(--overfit)
fi

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    ARGS+=(--resume "$RESUME_CHECKPOINT")
fi

NUM_GPUS="${NUM_GPUS:-1}"
if (( NUM_GPUS > 1 )); then
    torchrun         --standalone         --nproc_per_node="$NUM_GPUS"         scripts/train_frontier_architect.py         "${ARGS[@]}"
else
    python scripts/train_frontier_architect.py "${ARGS[@]}"
fi
