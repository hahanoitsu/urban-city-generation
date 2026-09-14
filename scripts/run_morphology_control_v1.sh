#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${URBAN_ROOT:-$SCRIPT_ROOT}"
RUN="$DATA_ROOT/runs/morphology-control-v1"
MODEL_OUT="$RUN/model"
ZIP="$DATA_ROOT/morphology-control-v1-results.zip"
CONFIG="$DATA_ROOT/configs/layered-corpus-v2-1km.yaml"
TRAIN_MANIFEST="$DATA_ROOT/data/manifests/corpus-v2/train.jsonl"
VAL_MANIFEST="$DATA_ROOT/data/manifests/corpus-v2/validation.jsonl"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city
export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

rm -rf "$RUN"
rm -f "$ZIP"
mkdir -p "$RUN"

echo "=== tests ==="
cd "$SCRIPT_ROOT"
python -m py_compile src/urban_model/morphology_control.py
pytest -q tests/test_morphology_control.py

echo
echo "=== morphology descriptors ==="
python -m urban_analysis     --manifest "$TRAIN_MANIFEST"     --output "$RUN/train-analysis"     --pca-components 5     --overwrite >/dev/null

python -m urban_analysis     --manifest "$VAL_MANIFEST"     --output "$RUN/validation-analysis"     --pca-components 5     --overwrite >/dev/null

echo
echo "=== gpu ==="
GPU_INDEX="$(
    nvidia-smi         --query-gpu=index,memory.free         --format=csv,noheader,nounits     | sort -t, -k2,2nr     | head -1     | cut -d, -f1     | xargs
)"
test -n "$GPU_INDEX" || { echo "No GPU found"; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
echo "using physical GPU $GPU_INDEX"
nvidia-smi -i "$GPU_INDEX"

echo
echo "=== morphology control v1 ==="
echo "controls:"
echo "  water coverage"
echo "  green coverage"
echo "  building coverage"
echo "  road density"
echo "  major-road share"
echo
echo "training budget: 4.75 h"
echo "final test: same noise, p10 / median / p90, 3 seeds"

python -m urban_model.morphology_control     --config "$CONFIG"     --train-descriptors "$RUN/train-analysis/tiles.csv"     --validation-descriptors "$RUN/validation-analysis/tiles.csv"     --output "$MODEL_OUT"     --max-hours 4.75     --max-epochs 3000     --device cuda     --preview-every 100     --sweep-steps 250     --overwrite

echo
echo "=== result ==="
python - <<PY
import json
from pathlib import Path

root = Path(r"$MODEL_OUT")
summary = json.loads((root / "summary.json").read_text())

print(f"epochs:  {summary['epochs']}")
print(f"updates: {summary['updates']}")
print(f"hours:   {summary['hours']}")
print()

for name, result in summary["control_results"].items():
    print(
        f"{name:28s} "
        f"corr={result['correlation']:+.3f} "
        f"monotonic={result['monotonic_seeds']}/{result['seeds']} "
        f"low/mid/high={result['low_mean']:.3f}/"
        f"{result['mid_mean']:.3f}/{result['high_mean']:.3f}"
    )
PY

echo
echo "=== package ==="
PACKAGE="/tmp/morphology-control-v1-package"
rm -rf "$PACKAGE"
mkdir -p "$PACKAGE/control-sweeps" "$PACKAGE/previews"

for name in     experiment.json control-stats.json metrics.csv     control-sweep.csv control-summary.json summary.json; do
    [[ -f "$MODEL_OUT/$name" ]] && cp "$MODEL_OUT/$name" "$PACKAGE/"
done

if [[ -d "$MODEL_OUT/control-sweeps" ]]; then
    cp "$MODEL_OUT/control-sweeps/"*.png "$PACKAGE/control-sweeps/" 2>/dev/null || true
fi

if [[ -d "$MODEL_OUT/previews" ]]; then
    mapfile -t PREVIEWS < <(
        find "$MODEL_OUT/previews" -maxdepth 1 -type f -name '*.png' | sort | tail -5
    )
    for preview in "${PREVIEWS[@]}"; do
        cp "$preview" "$PACKAGE/previews/"
    done
fi

cp "$RUN/train-analysis/summary.json" "$PACKAGE/train-analysis-summary.json"
cp "$RUN/validation-analysis/summary.json" "$PACKAGE/validation-analysis-summary.json"

cat > "$PACKAGE/provenance.txt" <<EOF
branch=morphology-control-v1
representation=8-class surface layout
controls=water_coverage,green_coverage,building_coverage,road_length_km_per_km2,road_major_share
conditioning=constant spatial channels
position_conditioning=xy
prediction_type=x0
high_noise_oversampling=true
same_noise_sweeps=true
sweep_levels=p10,median,p90
sweep_seeds=101,202,303
sweep_steps=250
vertical_transport=deferred
EOF

cd /tmp
zip -qr "$ZIP" "$(basename "$PACKAGE")"

echo
echo "=== complete ==="
ls -lh "$ZIP"
echo "checkpoint: $MODEL_OUT/latest.pt"
echo "sweeps:     $MODEL_OUT/control-sweeps/"
echo "upload:     $ZIP"
