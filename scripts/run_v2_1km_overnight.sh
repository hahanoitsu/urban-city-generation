#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKROOT="$(dirname "$SOURCE_ROOT")/urban-city-generation-overnight-v2-1km"
CONFIG="$SOURCE_ROOT/configs/layered-corpus-v2-1km.yaml"
RUN="$SOURCE_ROOT/runs/layered-corpus-v2-1km"
SAMPLES="$SOURCE_ROOT/runs/layered-corpus-v2-1km-samples"
AUDIT="$SOURCE_ROOT/runs/generated-city-audit-v2-1km"
PACKAGE="$SOURCE_ROOT/runs/generated-city-audit-v2-1km-package"
ZIP="$SOURCE_ROOT/singapore-generated-v2-1km-audit.zip"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

cd "$SOURCE_ROOT"

echo "=== SOURCE CHECKOUT ==="
git status --short

echo
echo "=== FETCH CLEAN EXPERIMENT CODE ==="
git fetch origin morphology-analysis
COMMIT="$(git rev-parse origin/morphology-analysis)"
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
        echo "Refusing to remove unregistered path: $WORKROOT"
        exit 1
    fi
fi

git -C "$SOURCE_ROOT" worktree prune
git -C "$SOURCE_ROOT" worktree add --detach "$WORKROOT" "$COMMIT"

export PYTHONPATH="$WORKROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo
echo "=== DATA CHECK ==="
for file in \
    "$SOURCE_ROOT/data/manifests/corpus-v2/train.jsonl" \
    "$SOURCE_ROOT/data/manifests/corpus-v2/validation.jsonl" \
    "$SOURCE_ROOT/data/manifests/corpus-v2/test.jsonl"; do
    test -f "$file" || {
        echo "Missing $file"
        exit 1
    }
    echo "$(basename "$file"): $(grep -cve '^$' "$file") tiles"
done

echo
echo "=== CODE TESTS ==="
cd "$WORKROOT"
python -m py_compile \
    src/urban_analysis/generated_city_audit.py \
    src/urban_model/vectorize.py \
    src/urban_model/train.py

pytest -q \
    tests/test_generated_city_audit.py \
    tests/test_generated_scene.py \
    tests/model/test_layered_diffusion.py \
    tests/model/test_diffusion_supervision.py

echo
echo "=== LAYERED DATA CHECK ==="
python -m urban_model check \
    --config "$CONFIG" \
    --samples 4

echo
echo "=== WAIT FOR FREE GPU ==="
while true; do
    mapfile -t GPU_PIDS < <(
        nvidia-smi \
            --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>/dev/null \
            | awk 'NF {print $1}' \
            | sort -u
    )

    if [[ ${#GPU_PIDS[@]} -eq 0 ]]; then
        echo "GPU is free."
        break
    fi

    echo "GPU compute process(es) active: ${GPU_PIDS[*]}"
    ps -o user,pid,etime,cmd -p "$(IFS=,; echo "${GPU_PIDS[*]}")" || true
    echo "Waiting 5 minutes rather than competing for the shared GPU..."
    sleep 300
done

nvidia-smi

echo
echo "=== CLEAN NEW EXPERIMENT OUTPUTS ==="
rm -rf "$RUN" "$SAMPLES" "$AUDIT" "$PACKAGE"
rm -f "$ZIP"
mkdir -p "$SAMPLES" "$AUDIT" "$PACKAGE"

echo
echo "=== TRAIN CORRECTED 1 KM BASELINE ==="
cd "$WORKROOT"
python -m urban_model train \
    --config "$CONFIG" \
    --epochs 200 \
    --batch-size 4 \
    --device cuda \
    --overwrite

cat > "$RUN/experiment.json" <<JSON
{
  "git_commit": "$COMMIT",
  "training_corpus": "data/manifests/corpus-v2",
  "resolution_pixels": 256,
  "crop_size_pixels": 256,
  "physical_tile_m": 1024,
  "epochs_requested": 200,
  "purpose": "corrected 1 km diffusion baseline for generated-city functionality audit"
}
JSON

echo
echo "=== CHECKPOINTS ==="
ls -lh "$RUN/best.pt" "$RUN/latest.pt"

sample_checkpoint() {
    local label="$1"
    local checkpoint="$2"
    local base_seed="$3"

    echo
    echo "--- sampling $label ---"

    for batch in 0 1 2 3; do
        local number=$((batch + 1))
        local seed=$((base_seed + batch * 100))
        local destination="$SAMPLES/$label/batch-$(printf '%02d' "$number")"

        python -m urban_model sample \
            --config "$CONFIG" \
            --checkpoint "$checkpoint" \
            --output "$destination" \
            --count 4 \
            --seed "$seed" \
            --device cuda \
            --overwrite
    done
}

sample_checkpoint best "$RUN/best.pt" 6150
sample_checkpoint latest "$RUN/latest.pt" 8150

echo
echo "=== GENERATED CITY AUDITS ==="

run_audit() {
    local generated="$1"
    local output="$2"

    python -m urban_analysis.generated_city_audit \
        --generated "$generated" \
        --real-manifest "$SOURCE_ROOT/data/manifests/corpus-v2/train.jsonl" \
        --real-manifest "$SOURCE_ROOT/data/manifests/corpus-v2/validation.jsonl" \
        --real-manifest "$SOURCE_ROOT/data/manifests/corpus-v2/test.jsonl" \
        --output "$output" \
        --overwrite
}

run_audit "$SAMPLES/best" "$AUDIT/best"
run_audit "$SAMPLES/latest" "$AUDIT/latest"
run_audit "$SAMPLES" "$AUDIT/combined"

echo
echo "=== COMPACT GENERATED VS REAL RESULTS ==="
python - <<PY
import json
from pathlib import Path

root = Path(r"$AUDIT")

metrics = [
    "road_assisted_largest_length_fraction",
    "road_assisted_interior_component_length_fraction",
    "road_interior_dead_ends",
    "buildings_within_20m_road_fraction",
    "buildings_beyond_80m_road_fraction",
    "building_road_distance_p90_m",
    "local_length_connected_to_higher_fraction",
    "road_component_length_serving_buildings_fraction",
    "building_area_median_m2",
    "building_area_max_m2",
]

for label in ("best", "latest"):
    data = json.loads((root / label / "summary.json").read_text())
    print()
    print("=" * 72)
    print(label.upper())
    print("=" * 72)
    print("generated samples:", data["generated"]["count"])
    print("real reference tiles:", data["real"]["count"])
    print()
    print(f"{'metric':48s} {'generated':>10s} {'real':>10s}")
    print("-" * 72)
    for metric in metrics:
        comparison = data["comparison"].get(metric)
        if not comparison:
            continue
        generated = comparison["generated_median"]
        real = comparison["real_median"]
        print(f"{metric:48s} {generated:10.4f} {real:10.4f}")
PY

echo
echo "=== PACKAGE RESULTS ==="
mkdir -p \
    "$PACKAGE/training" \
    "$PACKAGE/sample-previews"

for name in \
    config.json \
    environment.json \
    experiment.json \
    metrics.jsonl \
    preview.json \
    summary.json; do
    if [[ -f "$RUN/$name" ]]; then
        cp "$RUN/$name" "$PACKAGE/training/$name"
    fi
done

if [[ -d "$RUN/previews" ]]; then
    cp -r "$RUN/previews" "$PACKAGE/training/previews"
fi

cp -r "$AUDIT" "$PACKAGE/audit"

while IFS= read -r preview; do
    relative="${preview#$SAMPLES/}"
    safe="${relative//\//__}"
    cp "$preview" "$PACKAGE/sample-previews/$safe"
done < <(find "$SAMPLES" -path '*/sample-*/preview.png' -type f | sort)

{
    echo "experiment_commit=$COMMIT"
    echo
    echo "SOURCE WORKING TREE (not used for experiment code):"
    git -C "$SOURCE_ROOT" status --short
    echo
    echo "CHECKPOINTS LEFT ON SERVER:"
    ls -lh "$RUN/best.pt" "$RUN/latest.pt"
} > "$PACKAGE/provenance.txt"

cd "$SOURCE_ROOT"
zip -qr "$ZIP" "runs/$(basename "$PACKAGE")"

echo
echo "=== COMPLETE ==="
echo "Full training/checkpoints remain on the server:"
echo "  $RUN"
echo
echo "All generated samples remain on the server:"
echo "  $SAMPLES"
echo
echo "Upload this compact result package:"
ls -lh "$ZIP"
