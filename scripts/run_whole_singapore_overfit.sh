#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKROOT="$(dirname "$SOURCE_ROOT")/urban-city-generation-whole-singapore-overfit"
CITY="$SOURCE_ROOT/data/cities/singapore-v2.gpkg"
BOUNDARY="$SOURCE_ROOT/data/boundaries/singapore-planning-areas.geojson"
RUN="$SOURCE_ROOT/runs/whole-singapore-overfit"
ZIP="$SOURCE_ROOT/whole-singapore-overfit-results.zip"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

cd "$SOURCE_ROOT"
git fetch origin whole-singapore-overfit
COMMIT="$(git rev-parse origin/whole-singapore-overfit)"
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
test -f "$CITY" || { echo "Missing corrected city: $CITY"; exit 1; }
test -f "$BOUNDARY" || { echo "Missing official boundary: $BOUNDARY"; exit 1; }
ls -lh "$CITY" "$BOUNDARY"

echo
echo "=== TESTS ==="
cd "$WORKROOT"
python -m py_compile src/urban_model/whole_city_overfit.py
pytest -q tests/test_whole_city_overfit.py

echo
echo "=== BUILD WHOLE-SINGAPORE TARGET PREVIEW ==="
python - <<PY
from pathlib import Path
from urban_model.whole_city_overfit import build_overview_target, save_class_image

classes, summary = build_overview_target(
    r"$CITY",
    r"$BOUNDARY",
    resolution=512,
)
path = Path("/tmp/whole-singapore-target.png")
save_class_image(classes, path)
print("target preview:", path)
print("metres/pixel:", round(summary["metres_per_pixel"], 2))
print("class fractions:")
for name, value in summary["class_fraction"].items():
    print(f"  {name:14s} {value:.4f}")

road = summary["class_fraction"]["road_major"] + summary["class_fraction"]["road_minor"]
urban = summary["class_fraction"]["urban"]
if road < 0.002:
    raise SystemExit(f"Road coverage looks too small for city overview: {road:.6f}")
if urban < 0.02:
    raise SystemExit(f"Urban coverage looks too small for city overview: {urban:.6f}")
PY

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
echo "=== WHOLE-SINGAPORE PURE OVERFIT ==="
echo "One full-city target. No train/validation split. No planner. No vectorizer."
echo "20,000 single-scene optimisation steps; fixed pure-noise probe every 1,000 steps."

rm -rf "$RUN"
rm -f "$ZIP"

python -m urban_model.whole_city_overfit \
    --city "$CITY" \
    --boundary "$BOUNDARY" \
    --output "$RUN" \
    --resolution 512 \
    --steps 20000 \
    --diffusion-steps 1000 \
    --inference-steps 250 \
    --learning-rate 0.0002 \
    --sample-every 1000 \
    --checkpoint-every 1000 \
    --seed 5132 \
    --sample-seed 424242 \
    --device cuda \
    --overwrite

echo
echo "=== BEST PURE-NOISE RESULT ==="
python - <<PY
import csv
from pathlib import Path

p = Path(r"$RUN") / "metrics.csv"
rows = list(csv.DictReader(p.open()))
if not rows:
    raise SystemExit("No pure-noise sample metrics were written")

def score(row):
    return float(row["accuracy"]) + float(row["urban_iou"]) + float(row["road_iou"])

best = max(rows, key=score)
for key in ["step", "train_loss", "accuracy", "mean_iou", "urban_iou", "road_iou"]:
    print(f"{key:12s}: {best[key]}")
PY

echo
echo "=== PACKAGE ==="
cd "$SOURCE_ROOT"
rm -rf /tmp/whole-singapore-overfit-package
mkdir -p /tmp/whole-singapore-overfit-package/samples

for name in target.png target.json best.png best-comparison.png metrics.csv summary.json; do
    [[ -f "$RUN/$name" ]] && cp "$RUN/$name" /tmp/whole-singapore-overfit-package/
done
find "$RUN/samples" -maxdepth 1 -name '*.png' -type f -exec cp {} /tmp/whole-singapore-overfit-package/samples/ \;
find "$RUN/comparisons" -maxdepth 1 -name '*.png' -type f -exec cp {} /tmp/whole-singapore-overfit-package/samples/ \;

cat > /tmp/whole-singapore-overfit-package/provenance.txt <<EOF
experiment_commit=$COMMIT
corrected_city=$CITY
official_boundary=$BOUNDARY
resolution=512
training_examples=1 whole-Singapore scene
optimisation_steps=20000
pure_noise_sample_seed=424242
postprocessing=none before evaluation
EOF

cd /tmp
zip -qr "$ZIP" whole-singapore-overfit-package

echo
echo "=== COMPLETE ==="
ls -lh "$ZIP"
echo "Upload: $ZIP"
