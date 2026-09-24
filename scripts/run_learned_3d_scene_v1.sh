#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${URBAN_ROOT:-$SCRIPT_ROOT}"
RUN="$DATA_ROOT/runs/learned-3d-scene-v1"
ZIP="$DATA_ROOT/learned-3d-scene-v1-results.zip"
MAX_HOURS="${MAX_HOURS:-3}"
MAX_TOKENS="${MAX_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-2}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to one physical GPU before running."
    exit 1
fi

if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "Use exactly one GPU."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== learned 3d scene v1 ==="
echo "representation: unordered 3d road / rail / building objects"
echo "max tokens:     $MAX_TOKENS"
echo "max hours:      $MAX_HOURS"
echo "batch size:     $BATCH_SIZE"
echo "gpu:            physical $CUDA_VISIBLE_DEVICES"
echo
echo "Foreign GPU processes are not touched."
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"

echo
echo "=== city-state data ==="
STATE_COUNT="$(find "$DATA_ROOT/data/processed/corpus-v2" -type f -name city.json | wc -l | tr -d ' ')"
echo "city.json tiles: $STATE_COUNT"
if (( STATE_COUNT < 10 )); then
    echo "Too few city-state tiles. This experiment needs the vector city.json files."
    exit 1
fi

cd "$SCRIPT_ROOT"

echo
echo "=== checks ==="
python -m py_compile     src/urban_ai/object3d.py     src/urban_ai/object3d_scene.py     scripts/train_learned_3d_scene.py
pytest -q tests/test_object3d_model.py

rm -rf "$RUN"
rm -f "$ZIP"

echo
echo "=== training ==="
python scripts/train_learned_3d_scene.py     --source-root "$DATA_ROOT"     --output "$RUN"     --max-hours "$MAX_HOURS"     --maximum-tokens "$MAX_TOKENS"     --batch-size "$BATCH_SIZE"     --sample-steps 64     --preview-every 20

echo
echo "=== package ==="
PACKAGE="/tmp/learned-3d-scene-v1-package"
rm -rf "$PACKAGE"
mkdir -p "$PACKAGE"

for name in experiment.json summary.json metrics.csv sample-summary.csv; do
    [[ -f "$RUN/$name" ]] && cp "$RUN/$name" "$PACKAGE/"
done

if [[ -d "$RUN/samples" ]]; then
    cp -R "$RUN/samples" "$PACKAGE/"
fi

cd /tmp
zip -qr "$ZIP" "$(basename "$PACKAGE")"

echo
echo "=== complete ==="
ls -lh "$ZIP"
echo "checkpoint: $RUN/best.pt"
echo "samples:    $RUN/samples/"
echo "upload:     $ZIP"
