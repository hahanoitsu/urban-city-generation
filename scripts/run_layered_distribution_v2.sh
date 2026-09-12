#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${URBAN_ROOT:-$(git rev-parse --show-toplevel)}"
WORKROOT="$(dirname "$SOURCE_ROOT")/urban-city-generation-layered-distribution-v2"
BRANCH="layered-distribution-v2"
CONFIG="$SOURCE_ROOT/configs/layered-corpus-v2-1km.yaml"
RUN="$SOURCE_ROOT/runs/layered-distribution-v2"
ZIP="$SOURCE_ROOT/layered-distribution-v2-results.zip"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

cd "$SOURCE_ROOT"
git fetch origin "$BRANCH"
COMMIT="$(git rev-parse "origin/$BRANCH")"
echo "experiment commit: $COMMIT"

cleanup() {
    if git -C "$SOURCE_ROOT" worktree list --porcelain \
        | grep -Fxq "worktree $WORKROOT"; then
        git -C "$SOURCE_ROOT" worktree remove --force "$WORKROOT" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ -e "$WORKROOT" ]]; then
    if git -C "$SOURCE_ROOT" worktree list --porcelain \
        | grep -Fxq "worktree $WORKROOT"; then
        git -C "$SOURCE_ROOT" worktree remove --force "$WORKROOT"
    else
        echo "Refusing to remove non-worktree path: $WORKROOT"
        exit 1
    fi
fi

git -C "$SOURCE_ROOT" worktree prune
git -C "$SOURCE_ROOT" worktree add --detach "$WORKROOT" "$COMMIT"

export PYTHONPATH="$WORKROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo
echo "=== INPUT CHECK ==="
test -f "$CONFIG" || { echo "Missing config: $CONFIG"; exit 1; }
test -f "$SOURCE_ROOT/data/manifests/corpus-v2/train.jsonl" || {
    echo "Missing corpus-v2 train manifest"
    exit 1
}
test -f "$SOURCE_ROOT/data/manifests/corpus-v2/validation.jsonl" || {
    echo "Missing corpus-v2 validation manifest"
    exit 1
}
wc -l     "$SOURCE_ROOT/data/manifests/corpus-v2/train.jsonl"     "$SOURCE_ROOT/data/manifests/corpus-v2/validation.jsonl"

echo
echo "=== TESTS ==="
cd "$WORKROOT"
python -m py_compile src/urban_model/surface_distribution_v2.py
pytest -q tests/test_surface_distribution_v2.py

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
echo "=== LAYERED DISTRIBUTION V2 ==="
echo "Question:"
echo "  Can the successful x0 + XY + high-noise formulation learn a"
echo "  distribution across real Singapore layouts rather than one memorised map?"
echo
echo "Data:"
echo "  corrected corpus-v2 train/validation split"
echo "  256x256, 1 km tiles"
echo "  surface structure only for this first distribution test"
echo
echo "Training:"
echo "  wall-clock budget: 5.25 h"
echo "  max epochs: 3000"
echo "  full 1000-step final samples: 12"
echo "  no planner, graph repair, vectorizer or post-processing"
echo
echo "Existing whole-Singapore v2 checkpoints are untouched."

rm -rf "$RUN"
rm -f "$ZIP"

python -m urban_model.surface_distribution_v2 \
    --config "$CONFIG" \
    --output "$RUN" \
    --max-epochs 3000 \
    --max-hours 5.25 \
    --device cuda \
    --preview-every 100 \
    --checkpoint-every 25 \
    --final-samples 12 \
    --final-inference-steps 1000 \
    --overwrite

echo
echo "=== RESULT SUMMARY ==="
python - <<PY
import json
from pathlib import Path

root = Path(r"$RUN")
summary = json.loads((root / "summary.json").read_text())
distribution = json.loads((root / "distribution-metrics.json").read_text())

for key in [
    "epochs_completed",
    "updates",
    "training_hours",
    "best_high_noise_loss",
    "best_epoch",
    "mean_nearest_train_agreement_64",
    "max_nearest_train_agreement_64",
    "mean_pairwise_generated_agreement_64",
]:
    print(f"{key:38s}: {summary[key]}")

print("\ngenerated class fractions:")
for key, value in distribution["generated_class_fraction"].items():
    print(f"  {key:16s} {value:.4f}")

print("\ntraining class fractions:")
for key, value in distribution["train_class_fraction"].items():
    print(f"  {key:16s} {value:.4f}")
PY

echo
echo "=== PACKAGE ==="
PACKAGE="/tmp/layered-distribution-v2-package"
rm -rf "$PACKAGE"
mkdir -p "$PACKAGE/previews"

for name in \
    experiment.json metrics.csv \
    real-train.png real-validation.png \
    final-samples.png nearest-neighbours.png \
    distribution-metrics.json summary.json; do
    [[ -f "$RUN/$name" ]] && cp "$RUN/$name" "$PACKAGE/"
done

if [[ -d "$RUN/previews" ]]; then
    mapfile -t PREVIEWS < <(find "$RUN/previews" -maxdepth 1 -name '*.png' -type f | sort | tail -6)
    for preview in "${PREVIEWS[@]}"; do
        cp "$preview" "$PACKAGE/previews/"
    done
fi

cat > "$PACKAGE/provenance.txt" <<EOF
experiment_commit=$COMMIT
experiment=layered-distribution-v2
config=$CONFIG
train_manifest=$SOURCE_ROOT/data/manifests/corpus-v2/train.jsonl
validation_manifest=$SOURCE_ROOT/data/manifests/corpus-v2/validation.jsonl
representation=8-class surface semantic layout
resolution=256x256
physical_extent=1km x 1km per tile
prediction_type=x0/sample
position_conditioning=relative XY channels
high_noise_oversampling=true
class_balanced_loss=true
training_wallclock_budget_hours=5.25
final_inference_steps=1000
final_samples=12
postprocessing=none
checkpoint_directory=$RUN
EOF

cd /tmp
zip -qr "$ZIP" "$(basename "$PACKAGE")"

echo
echo "=== COMPLETE ==="
ls -lh "$ZIP"
echo "Final samples: $RUN/final-samples.png"
echo "Nearest-neighbour audit: $RUN/nearest-neighbours.png"
echo "Best checkpoint preserved: $RUN/best.pt"
echo "Latest checkpoint preserved: $RUN/latest.pt"
echo "Upload: $ZIP"
