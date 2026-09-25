#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${CONTEXT_DATA:-$MAIN_ROOT/data/context-graph-v1/singapore}"
CHECKPOINT="${STRUCTURED_CHECKPOINT:-$MAIN_ROOT/runs/structured-city-v1/best.pt}"
OUTPUT="${STRUCTURED_PREVIEWS:-$MAIN_ROOT/runs/structured-city-v1/previews}"
ZIP="${STRUCTURED_PREVIEW_ZIP:-$MAIN_ROOT/structured-city-v1-previews.zip}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_ROOT"
rm -rf "$OUTPUT"
rm -f "$ZIP"

python scripts/sample_structured_city.py     --data "$DATA"     --checkpoint "$CHECKPOINT"     --output "$OUTPUT"     --samples "${SAMPLES:-8}"     --steps "${STEPS:-40}"

cd "$(dirname "$OUTPUT")"
zip -qr "$ZIP" "$(basename "$OUTPUT")"
ls -lh "$ZIP"
