#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${CONTEXT_DATA:-$MAIN_ROOT/data/context-graph-v1/singapore}"
OUTPUT="${SCENE_AUDIT:-$MAIN_ROOT/structured-city-audit.json}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_ROOT"

python -m py_compile     src/urban_model/structured_city.py     src/urban_model/structured_city_data.py     scripts/audit_structured_city.py

pytest -q tests/test_structured_city.py

ARGS=(
    --data "$DATA"
    --output "$OUTPUT"
)

if [[ -n "${LIMIT:-}" ]]; then
    ARGS+=(--limit "$LIMIT")
fi

python scripts/audit_structured_city.py "${ARGS[@]}"
