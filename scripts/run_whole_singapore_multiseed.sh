#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="$ROOT/runs/whole-singapore-overfit-v2/latest.pt"
CITY="$ROOT/data/cities/singapore-v2.gpkg"
BOUNDARY="$ROOT/data/boundaries/singapore-planning-areas.geojson"
OUT="$ROOT/runs/whole-singapore-overfit-v2-multiseed"
ZIP="$ROOT/whole-singapore-overfit-v2-multiseed-results.zip"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$ROOT"

echo "=== CHECKS ==="
test -f "$CHECKPOINT" || { echo "Missing checkpoint: $CHECKPOINT"; exit 1; }
test -f "$CITY" || { echo "Missing city: $CITY"; exit 1; }
test -f "$BOUNDARY" || { echo "Missing boundary: $BOUNDARY"; exit 1; }
python -m py_compile scripts/eval_whole_singapore_multiseed.py
ls -lh "$CHECKPOINT"

echo
echo "=== SELECT GPU ==="
GPU_INDEX="$(
    nvidia-smi \
        --query-gpu=index,memory.free \
        --format=csv,noheader,nounits \
    | sort -t, -k2,2nr \
    | head -1 \
    | cut -d, -f1 \
    | xargs
)"
test -n "$GPU_INDEX" || { echo "Could not select a GPU"; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
echo "Using physical GPU $GPU_INDEX"
nvidia-smi -i "$GPU_INDEX"

echo
echo "=== UNSEEN-SEED REPRODUCIBILITY TEST ==="
echo "Checkpoint: $CHECKPOINT"
echo "Seeds: 1 123 2026 9999 8675309"
echo "Inference: full 1000-step DDIM chain"
echo "Original evaluation seed 424242 is deliberately excluded."

rm -rf "$OUT"
rm -f "$ZIP"

python scripts/eval_whole_singapore_multiseed.py \
    --checkpoint "$CHECKPOINT" \
    --city "$CITY" \
    --boundary "$BOUNDARY" \
    --output "$OUT" \
    --seeds 1 123 2026 9999 8675309 \
    --inference-steps 1000 \
    --device cuda

echo
echo "=== PACKAGE ==="
cd "$ROOT"
zip -qr "$ZIP" "$(basename "$OUT")"

echo
echo "=== COMPLETE ==="
ls -lh "$ZIP"
echo "Contact sheet: $OUT/contact-sheet.png"
echo "Upload: $ZIP"
