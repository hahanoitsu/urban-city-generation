#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${CONTEXT_DATA:-$MAIN_ROOT/data/context-graph-v1/singapore}"
OUTPUT="${STRUCTURED_RUN:-$MAIN_ROOT/runs/structured-city-v1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SCRIPT_ROOT"

python -m py_compile     src/urban_model/structured_city.py     src/urban_model/structured_city_data.py     src/urban_model/structured_city_diffusion.py     scripts/train_structured_city.py

pytest -q tests/test_structured_city.py

python scripts/train_structured_city.py     --data "$DATA"     --output "$OUTPUT"     --epochs "${EPOCHS:-30}"     --batch-size "${BATCH_SIZE:-1}"     --nodes "${NODE_SLOTS:-384}"     --edges "${EDGE_SLOTS:-640}"     --buildings "${BUILDING_SLOTS:-384}"     --areas "${AREA_SLOTS:-96}"     --ports "${PORT_SLOTS:-96}"     ${MAXIMUM_SAMPLES:+--maximum-samples "$MAXIMUM_SAMPLES"}
