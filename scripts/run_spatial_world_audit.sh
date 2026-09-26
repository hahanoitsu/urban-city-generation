#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${SPATIAL_WORLD_DATA:-$MAIN_ROOT/data/spatial-world-v1/singapore}"
OUTPUT="${SPATIAL_WORLD_PREVIEWS:-$MAIN_ROOT/runs/spatial-world-v1/data-previews}"
ZIP="${SPATIAL_WORLD_PREVIEW_ZIP:-$MAIN_ROOT/spatial-world-v1-data-previews.zip}"

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
rm -rf "$OUTPUT"
rm -f "$ZIP"

python scripts/audit_spatial_world.py --data "$DATA"
python scripts/preview_spatial_world_data.py     --data "$DATA"     --output "$OUTPUT"     --samples "${SAMPLES:-8}"

cd "$(dirname "$OUTPUT")"
zip -qr "$ZIP" "$(basename "$OUTPUT")"
ls -lh "$ZIP"
