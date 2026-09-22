#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${URBAN_ROOT:-$SCRIPT_ROOT}"
RUN="$DATA_ROOT/runs/morphology-control-1024-v1"
MODEL_OUT="$RUN/model"
ZIP="$DATA_ROOT/morphology-control-1024-v1-results.zip"
TRAIN_MANIFEST="$DATA_ROOT/data/manifests/corpus-v2-1024/train.jsonl"
VAL_MANIFEST="$DATA_ROOT/data/manifests/corpus-v2-1024/validation.jsonl"
MAX_HOURS="${MAX_HOURS:-22}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to one physical GPU before running."
    echo "Example: CUDA_VISIBLE_DEVICES=0 bash scripts/run_morphology_control_1024_v1.sh"
    exit 1
fi
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "Use exactly one GPU for this run."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== experiment ==="
echo "native raster: 1024 x 1024"
echo "physical tile: 1024 m x 1024 m"
echo "resolution:    1 m / pixel"
echo "training:      from scratch"
echo "precision:     bf16"
echo "batch:         1"
echo "max hours:     $MAX_HOURS"
echo "gpu:           physical $CUDA_VISIBLE_DEVICES"
echo
echo "Foreign GPU processes are not touched."
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"

echo
echo "=== disk ==="
df -h "$DATA_ROOT"
AVAILABLE_KB="$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')"
if (( AVAILABLE_KB < 20 * 1024 * 1024 )); then
    echo "Refusing to build high-res corpus with less than 20 GB free."
    exit 1
fi

echo
echo "=== tests ==="
cd "$SCRIPT_ROOT"
python -m py_compile     src/urban_model/morphology_control.py     scripts/build_corpus_1024.py     scripts/train_morphology_1024.py
pytest -q tests/test_morphology_control.py

echo
echo "=== native 1024 corpus ==="
python scripts/build_corpus_1024.py     --source-root "$DATA_ROOT"     --pixels 1024

echo
echo "=== split check ==="
for split in train validation test; do
    old_count="$(wc -l < "$DATA_ROOT/data/manifests/corpus-v2/$split.jsonl")"
    new_count="$(wc -l < "$DATA_ROOT/data/manifests/corpus-v2-1024/$split.jsonl")"
    printf "%-12s old=%s new=%s\n" "$split" "$old_count" "$new_count"
    [[ "$old_count" == "$new_count" ]] || {
        echo "Split count mismatch. Stopping before training."
        exit 1
    }
done

rm -rf "$RUN"
rm -f "$ZIP"
mkdir -p "$RUN"

echo
echo "=== 1024 morphology descriptors ==="
python -m urban_analysis     --manifest "$TRAIN_MANIFEST"     --output "$RUN/train-analysis"     --pca-components 5     --overwrite >/dev/null

python -m urban_analysis     --manifest "$VAL_MANIFEST"     --output "$RUN/validation-analysis"     --pca-components 5     --overwrite >/dev/null

echo
echo "=== scratch 1024 training ==="
python scripts/train_morphology_1024.py     --source-root "$DATA_ROOT"     --output "$MODEL_OUT"     --train-descriptors "$RUN/train-analysis/tiles.csv"     --validation-descriptors "$RUN/validation-analysis/tiles.csv"     --max-hours "$MAX_HOURS"     --max-epochs 3000     --preview-every 25     --sweep-steps 250

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
PACKAGE="/tmp/morphology-control-1024-v1-package"
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
        find "$MODEL_OUT/previews" -maxdepth 1 -type f -name '*.png' | sort | tail -6
    )
    for preview in "${PREVIEWS[@]}"; do
        cp "$preview" "$PACKAGE/previews/"
    done
fi

cp "$RUN/train-analysis/summary.json" "$PACKAGE/train-analysis-summary.json"
cp "$RUN/validation-analysis/summary.json" "$PACKAGE/validation-analysis-summary.json"
cp "$DATA_ROOT/data/processed/corpus-v2-1024/highres_summary.json" "$PACKAGE/corpus-1024-summary.json"
cp "$DATA_ROOT/data/manifests/corpus-v2-1024/manifest_summary.json" "$PACKAGE/manifest-1024-summary.json"

cat > "$PACKAGE/provenance.txt" <<EOF
experiment=morphology-control-1024-v1
training=from_scratch
source_city=$DATA_ROOT/data/cities/singapore-v2.gpkg
canonical_split_source=$DATA_ROOT/data/manifests/corpus-v2
native_raster_resolution=1024x1024
physical_extent_m=1024x1024
metres_per_pixel=1.0
representation=8-class surface layout
controls=water_coverage,green_coverage,building_coverage,road_length_km_per_km2,road_major_share
conditioning=constant spatial channels
position_conditioning=xy
prediction_type=x0
high_noise_oversampling=true
precision=bf16
batch_size=1
cuda_tuning=tf32,fused_adamw,channels_last
gradient_checkpointing=true
training_wallclock_budget_hours=$MAX_HOURS
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
echo "previews:   $MODEL_OUT/previews/"
echo "sweeps:     $MODEL_OUT/control-sweeps/"
echo "upload:     $ZIP"
