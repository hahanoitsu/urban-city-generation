#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${CONTEXT_DATA:-$MAIN_ROOT/data/context-graph-v1/singapore}"
OUTPUT="${STRUCTURED_INTEGRITY_AUDIT:-$MAIN_ROOT/structured-city-integrity.json}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_ROOT"

python -m py_compile     src/urban_model/structured_city_data.py     scripts/audit_structured_representation.py

python scripts/audit_structured_representation.py     --data "$DATA"     --output "$OUTPUT"     --transport-samples "${TRANSPORT_SAMPLES:-400}"     --polygon-samples "${POLYGON_SAMPLES:-12000}"
