#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKROOT="$(dirname "$SOURCE_ROOT")/urban-city-generation-denoise-probe"
CONFIG="$SOURCE_ROOT/configs/layered-corpus-v2-1km.yaml"
CHECKPOINT="$SOURCE_ROOT/runs/layered-corpus-v2-1km/best.pt"
OUTPUT="$SOURCE_ROOT/runs/denoise-probe-v2-1km"
ZIP="$SOURCE_ROOT/singapore-denoise-probe-v2-1km.zip"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

cd "$SOURCE_ROOT"

echo "=== SOURCE CHECKOUT ==="
git status --short

echo
echo "=== FETCH CLEAN PROBE CODE ==="
git fetch origin morphology-analysis
COMMIT="$(git rev-parse origin/morphology-analysis)"
echo "probe commit: $COMMIT"

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
        echo "Refusing to remove unregistered path: $WORKROOT"
        exit 1
    fi
fi

git -C "$SOURCE_ROOT" worktree prune
git -C "$SOURCE_ROOT" worktree add --detach "$WORKROOT" "$COMMIT"

export PYTHONPATH="$WORKROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo
echo "=== INPUT CHECK ==="
test -f "$CONFIG" || {
    echo "Missing config: $CONFIG"
    exit 1
}
test -f "$CHECKPOINT" || {
    echo "Missing corrected checkpoint: $CHECKPOINT"
    exit 1
}
test -f "$SOURCE_ROOT/data/manifests/corpus-v2/validation.jsonl" || {
    echo "Missing corrected validation manifest"
    exit 1
}
ls -lh "$CHECKPOINT"
echo "validation tiles: $(grep -cve '^$' "$SOURCE_ROOT/data/manifests/corpus-v2/validation.jsonl")"

echo
echo "=== CODE TEST ==="
cd "$WORKROOT"
python -m py_compile src/urban_analysis/denoise_probe.py
pytest -q tests/test_denoise_probe.py

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

if [[ -z "$GPU_INDEX" ]]; then
    echo "Could not select a GPU"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
echo "Using physical GPU $GPU_INDEX"
nvidia-smi -i "$GPU_INDEX"

echo
echo "=== RUN DENOISING PROBE ==="
rm -rf "$OUTPUT"
rm -f "$ZIP"

python -m urban_analysis.denoise_probe \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --tile-count 4 \
    --timestep 25 \
    --timestep 100 \
    --timestep 250 \
    --timestep 500 \
    --timestep 750 \
    --seed 9142 \
    --device cuda \
    --overwrite

cat > "$OUTPUT/provenance.txt" <<EOF
experiment_commit=$COMMIT
checkpoint=$CHECKPOINT
config=$CONFIG
gpu_physical_index=$GPU_INDEX

SOURCE WORKING TREE (not used for probe code):
$(git -C "$SOURCE_ROOT" status --short)
EOF

echo
echo "=== COMPACT RESULTS ==="
python - <<PY
import json
from pathlib import Path

p = Path(r"$OUTPUT") / "summary.json"
data = json.loads(p.read_text())

metrics = [
    "surface_accuracy",
    "road_iou",
    "building_iou",
    "rail_iou",
    "mse_supervised",
    "road_assisted_components",
    "road_assisted_largest_length_fraction",
    "road_assisted_interior_component_length_fraction",
    "road_interior_dead_ends",
    "local_length_connected_to_higher_fraction",
    "building_count",
]

print("checkpoint epoch:", data["checkpoint_epoch"])
print("best validation loss:", data["checkpoint_best_validation_loss"])
print("validation tiles:", ", ".join(data["tile_ids"]))
print()

print(f"{'level':12s} {'signal':>8s} {'noise':>8s}")
print("-" * 32)
for level, block in data["levels"].items():
    m = block["metrics"]
    signal = m.get("signal_scale", {}).get("median", float("nan"))
    noise = m.get("noise_scale", {}).get("median", float("nan"))
    print(f"{level:12s} {signal:8.4f} {noise:8.4f}")

for metric in metrics:
    print()
    print(metric)
    for level, block in data["levels"].items():
        value = block["metrics"].get(metric, {}).get("median")
        if value is not None:
            print(f"  {level:12s} {value:.6f}")
PY

echo
echo "=== PACKAGE ==="
cd "$SOURCE_ROOT"
zip -qr "$ZIP" "runs/$(basename "$OUTPUT")"

echo
echo "=== COMPLETE ==="
ls -lh "$ZIP"
echo
echo "Upload:"
echo "  $ZIP"
