#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"

export STRUCTURED_RUN="${STRUCTURED_RUN:-$MAIN_ROOT/runs/structured-city-full-v1}"
export STRUCTURED_CACHE="${STRUCTURED_CACHE:-$MAIN_ROOT/data/structured-city-cache-v1}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export SAVE_EVERY="${SAVE_EVERY:-5}"
export NODE_SLOTS="${NODE_SLOTS:-448}"
export EDGE_SLOTS="${EDGE_SLOTS:-512}"
export BUILDING_SLOTS="${BUILDING_SLOTS:-512}"
export AREA_SLOTS="${AREA_SLOTS:-160}"
export PORT_SLOTS="${PORT_SLOTS:-96}"

rm -rf "$STRUCTURED_RUN"

bash "$SCRIPT_ROOT/scripts/run_structured_city.sh"
