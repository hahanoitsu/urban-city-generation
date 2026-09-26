#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
CITY="${CITY_GPKG:-$MAIN_ROOT/data/cities/singapore-v2.gpkg}"
OUTPUT="${SPATIAL_WORLD_DATA:-$MAIN_ROOT/data/spatial-world-v1/singapore}"

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

cd "$SCRIPT_ROOT"

python -m py_compile     src/urban_dataset/spatial_world.py     scripts/build_spatial_world_v1.py

pytest -q tests/test_spatial_world.py

rm -rf "$OUTPUT"

python scripts/build_spatial_world_v1.py     --city "$CITY"     --output "$OUTPUT"     --context-size "${CONTEXT_SIZE_M:-5120}"     --local-vector-size "${LOCAL_VECTOR_SIZE_M:-2560}"     --target-size "${TARGET_SIZE_M:-1024}"     --stride "${TARGET_STRIDE_M:-1024}"     --context-cell "${CONTEXT_CELL_M:-512}"     --minimum-transport "${MINIMUM_TRANSPORT_M:-100}"
