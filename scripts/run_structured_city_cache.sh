#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
DATA="${CONTEXT_DATA:-$MAIN_ROOT/data/context-graph-v1/singapore}"
CACHE="${STRUCTURED_CACHE:-$MAIN_ROOT/data/structured-city-cache-v1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_ROOT"

python scripts/cache_structured_city.py     --data "$DATA"     --cache-dir "$CACHE"     --nodes "${NODE_SLOTS:-448}"     --edges "${EDGE_SLOTS:-512}"     --buildings "${BUILDING_SLOTS:-512}"     --areas "${AREA_SLOTS:-160}"     --ports "${PORT_SLOTS:-96}"
