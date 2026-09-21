#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${URBAN_ROOT:-$SCRIPT_ROOT}"
OUT="$DATA_ROOT/runs/structured-output-v1"
ZIP="$DATA_ROOT/structured-output-v1-results.zip"
CHECKPOINT="$DATA_ROOT/runs/morphology-control-v1/model/latest.pt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city
export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_ROOT"

python -m py_compile     src/urban_model/surface_vectorize.py     src/urban_analysis/surface_roundtrip.py     scripts/structured_output_v1.py

pytest -q tests/test_surface_roundtrip.py

rm -rf "$OUT"
rm -f "$ZIP"

python scripts/structured_output_v1.py     --checkpoint "$CHECKPOINT"     --output "$OUT"     --device cuda     --steps 250

cd "$DATA_ROOT/runs"
zip -qr "$ZIP" structured-output-v1

echo
echo "done"
echo "results: $OUT"
echo "upload:  $ZIP"
