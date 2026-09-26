#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"
OUTPUT="${STRUCTURED_RUN:-$MAIN_ROOT/runs/structured-city-pilot-v1}"

export STRUCTURED_RUN="$OUTPUT"
export MAXIMUM_SAMPLES="${MAXIMUM_SAMPLES:-256}"
export EPOCHS="${EPOCHS:-6}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NODE_SLOTS="${NODE_SLOTS:-448}"
export EDGE_SLOTS="${EDGE_SLOTS:-512}"
export BUILDING_SLOTS="${BUILDING_SLOTS:-512}"
export AREA_SLOTS="${AREA_SLOTS:-160}"
export PORT_SLOTS="${PORT_SLOTS:-96}"

rm -rf "$OUTPUT"

bash "$SCRIPT_ROOT/scripts/run_structured_city.sh"
