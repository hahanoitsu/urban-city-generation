#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${URBAN_ROOT:-$SCRIPT_ROOT}"
CHECKPOINT="$DATA_ROOT/runs/morphology-control-1024-v1/model/latest.pt"
OUTPUT="$DATA_ROOT/runs/morphology-control-1024-loss-probe"
ZIP="$DATA_ROOT/morphology-control-1024-loss-probe.zip"
UPDATES="${PROBE_UPDATES:-2500}"
STEPS="${PROBE_STEPS:-125}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to one physical GPU before running."
    exit 1
fi

if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "Use exactly one GPU."
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== loss probe ==="
echo "checkpoint: $CHECKPOINT"
echo "updates:    $UPDATES per fine-tune arm"
echo "sampling:   $STEPS DDIM steps"
echo "gpu:        physical $CUDA_VISIBLE_DEVICES"
echo
echo "Foreign GPU processes are not touched."
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"

cd "$SCRIPT_ROOT"

python -m py_compile scripts/run_1024_loss_probe.py
pytest -q tests/test_1024_loss_probe.py

rm -rf "$OUTPUT"
rm -f "$ZIP"

python scripts/run_1024_loss_probe.py \
    --source-root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --updates "$UPDATES" \
    --inference-steps "$STEPS"

cd "$DATA_ROOT/runs"
zip -qr "$ZIP" "$(basename "$OUTPUT")"

echo
echo "=== complete ==="
ls -lh "$ZIP"
echo "upload: $ZIP"
