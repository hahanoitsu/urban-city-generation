#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKROOT="$(dirname "$SOURCE_ROOT")/urban-city-generation-whole-singapore-overfit-v2"
CITY="$SOURCE_ROOT/data/cities/singapore-v2.gpkg"
BOUNDARY="$SOURCE_ROOT/data/boundaries/singapore-planning-areas.geojson"
RUN="$SOURCE_ROOT/runs/whole-singapore-overfit-v2"
ZIP="$SOURCE_ROOT/whole-singapore-overfit-v2-results.zip"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

cd "$SOURCE_ROOT"
git fetch origin whole-singapore-overfit-v2
COMMIT="$(git rev-parse origin/whole-singapore-overfit-v2)"
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
echo "=== MODEL SHAPE SMOKE TEST ==="
python - <<'PY'
import torch
from urban_model.whole_city_overfit import (
    OVERVIEW_CHANNELS,
    _build_model,
    _coordinate_grid,
)

model = _build_model(64)
noisy = torch.randn(1, OVERVIEW_CHANNELS, 64, 64)
coords = _coordinate_grid(64, torch.device("cpu"))
out = model(torch.cat([noisy, coords], dim=1), torch.tensor([999])).sample
assert out.shape == noisy.shape, (out.shape, noisy.shape)
print("model input:", tuple(noisy.shape[:1] + (noisy.shape[1] + 2,) + noisy.shape[2:]))
print("model output:", tuple(out.shape))
PY

echo
echo "=== TARGET PREVIEW ==="
python - <<PY
from pathlib import Path
from urban_model.whole_city_overfit import build_overview_target, save_class_image

classes, summary = build_overview_target(
    r"$CITY",
    r"$BOUNDARY",
    resolution=512,
)
path = Path("/tmp/whole-singapore-target-v2.png")
save_class_image(classes, path)
print("target preview:", path)
print("metres/pixel:", round(summary["metres_per_pixel"], 2))
print("class fractions:")
for name, value in summary["class_fraction"].items():
    print(f"  {name:14s} {value:.4f}")
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
echo "=== WHOLE-SINGAPORE OVERFIT V2 ==="
echo "Changes from v1:"
echo "  - x/y coordinate conditioning"
echo "  - direct clean-map (x0) prediction instead of epsilon prediction"
echo "  - high-noise timesteps deliberately oversampled"
echo "  - modest class-balanced pixel loss"
echo "  - tighter full-Singapore frame"
echo "  - final 1000-step DDIM pure-noise sample"
echo
echo "12,000 optimisation steps; quick pure-noise probe every 1,000 steps."

rm -rf "$RUN"
rm -f "$ZIP"

python -m urban_model.whole_city_overfit \
    --city "$CITY" \
    --boundary "$BOUNDARY" \
    --output "$RUN" \
    --resolution 512 \
    --steps 12000 \
    --diffusion-steps 1000 \
    --inference-steps 100 \
    --final-inference-steps 1000 \
    --learning-rate 0.0002 \
    --sample-every 1000 \
    --checkpoint-every 1000 \
    --seed 5132 \
    --sample-seed 424242 \
    --device cuda \
    --overwrite

echo
echo "=== RESULT ==="
python - <<PY
import csv, json
from pathlib import Path

root = Path(r"$RUN")
rows = list(csv.DictReader((root / "metrics.csv").open()))
if rows:
    def score(row):
        return float(row["accuracy"]) + float(row["urban_iou"]) + float(row["road_iou"])
    best = max(rows, key=score)
    print("best quick probe:")
    for key in ["step", "train_loss", "accuracy", "mean_iou", "urban_iou", "road_iou"]:
        print(f"  {key:12s}: {best[key]}")

final = json.loads((root / "final-full-chain-metrics.json").read_text())
print("final 1000-step probe:")
for key in ["accuracy", "mean_iou", "urban_iou", "road_iou"]:
    print(f"  {key:12s}: {final[key]:.6f}")
PY

echo
echo "=== PACKAGE ==="
cd "$SOURCE_ROOT"
PACKAGE="/tmp/whole-singapore-overfit-v2-package"
rm -rf "$PACKAGE"
mkdir -p "$PACKAGE/samples"

for name in \
    target.png target.json \
    best.png best-comparison.png \
    final-full-chain.png final-full-chain-comparison.png \
    final-full-chain-metrics.json metrics.csv summary.json; do
    [[ -f "$RUN/$name" ]] && cp "$RUN/$name" "$PACKAGE/"
done

find "$RUN/samples" -maxdepth 1 -name '*.png' -type f -exec cp {} "$PACKAGE/samples/" \;
find "$RUN/comparisons" -maxdepth 1 -name '*.png' -type f -exec cp {} "$PACKAGE/samples/" \;

cat > "$PACKAGE/provenance.txt" <<EOF
experiment_commit=$COMMIT
experiment=whole-singapore-overfit-v2
corrected_city=$CITY
official_boundary=$BOUNDARY
resolution=512
training_examples=1 whole-Singapore scene
optimisation_steps=12000
prediction_type=x0/sample
position_conditioning=x,y
high_noise_oversampling=true
class_balanced_loss=true
periodic_inference_steps=100
final_inference_steps=1000
pure_noise_sample_seed=424242
postprocessing=none before evaluation
EOF

cd /tmp
zip -qr "$ZIP" whole-singapore-overfit-v2-package

echo
echo "=== COMPLETE ==="
ls -lh "$ZIP"
echo "Upload: $ZIP"
