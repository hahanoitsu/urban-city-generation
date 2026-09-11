#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKROOT="$(dirname "$SOURCE_ROOT")/urban-city-generation-structural-v2"
CONFIG="$SOURCE_ROOT/configs/structural-v2.yaml"
PROGRAMS="$SOURCE_ROOT/data/programs/structural-v2"
RUN="$SOURCE_ROOT/runs/structural-v2"
SAMPLES="$SOURCE_ROOT/runs/structural-v2-samples"
AUDIT="$SOURCE_ROOT/runs/structural-v2-audit"
PACKAGE="$SOURCE_ROOT/runs/structural-v2-package"
ZIP="$SOURCE_ROOT/singapore-structural-v2.zip"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

cd "$SOURCE_ROOT"

echo "=== FETCH CLEAN EXPERIMENT CODE ==="
git fetch origin structural-generator-v2
COMMIT="$(git rev-parse origin/structural-generator-v2)"
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
echo "=== INPUT CHECK ==="
for split in train validation test; do
    file="$SOURCE_ROOT/data/manifests/corpus-v2/$split.jsonl"
    test -f "$file" || { echo "Missing $file"; exit 1; }
    echo "$split: $(grep -cve '^$' "$file") tiles"
done

echo
echo "=== CODE TESTS ==="
cd "$WORKROOT"
python -m py_compile \
    src/urban_ai/prepare.py \
    src/urban_analysis/structural_generator_audit.py
pytest -q \
    tests/test_structural_prepare.py \
    tests/test_structural_generator_audit.py \
    tests/test_structural_relative_codec.py \
    tests/test_structural_planarity.py \
    tests/test_structural_densify.py \
    tests/test_graph_program.py \
    tests/test_graph_model.py \
    tests/test_generated_scene.py

echo
echo "=== PREPARE CONNECTED ROAD PROGRAMS ==="
rm -rf "$PROGRAMS"
python -m urban_ai prepare --config "$CONFIG" --overwrite
python -m urban_ai check --config "$CONFIG"

python - <<PY
import json
from pathlib import Path

p = Path(r"$PROGRAMS") / "summary.json"
d = json.loads(p.read_text())
print()
print("Prepared structural corpus:")
for split, value in d["splits"].items():
    print(
        f"  {split:10s} accepted={value['accepted']:3d} "
        f"rejected={value['rejected']:3d} "
        f"nodes_med={value['node_count']['median']:.1f} "
        f"commands_p90={value['command_count']['p90']:.1f}"
    )

train = d["splits"].get("train", {}).get("accepted", 0)
validation = d["splits"].get("validation", {}).get("accepted", 0)
if train < 80 or validation < 10:
    raise SystemExit(
        f"Too few connected-road programs for a useful run: "
        f"train={train}, validation={validation}"
    )
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
echo "=== CLEAN EXPERIMENT OUTPUTS ==="
rm -rf "$RUN" "$SAMPLES" "$AUDIT" "$PACKAGE"
rm -f "$ZIP"
mkdir -p "$SAMPLES" "$AUDIT" "$PACKAGE"

echo
echo "=== TRAIN LOCAL PLANAR STRUCTURAL GRAPH MODEL ==="
cd "$WORKROOT"
python -m urban_ai train \
    --config "$CONFIG" \
    --epochs 80 \
    --batch-size 8 \
    --device cuda \
    --overwrite

echo
echo "=== SAMPLE THREE TEMPERATURES ==="
for spec in low:0.40 mid:0.65 high:0.85; do
    label="${spec%%:*}"
    temperature="${spec##*:}"
    python -m urban_ai sample \
        --config "$CONFIG" \
        --checkpoint "$RUN/best.pt" \
        --output "$SAMPLES/$label" \
        --count 8 \
        --seed "$((7000 + ${#label} * 100))" \
        --temperature "$temperature" \
        --device cuda \
        --overwrite
done

echo
echo "=== STRUCTURAL ROAD AUDIT ==="
python -m urban_analysis.structural_generator_audit \
    --generated "$SAMPLES" \
    --real-manifest "$SOURCE_ROOT/data/manifests/corpus-v2/train.jsonl" \
    --real-manifest "$SOURCE_ROOT/data/manifests/corpus-v2/validation.jsonl" \
    --real-manifest "$SOURCE_ROOT/data/manifests/corpus-v2/test.jsonl" \
    --output "$AUDIT/roads" \
    --overwrite

echo
echo "=== END-TO-END CITY AUDIT ==="
python -m urban_analysis.generated_city_audit \
    --generated "$SAMPLES" \
    --real-manifest "$SOURCE_ROOT/data/manifests/corpus-v2/train.jsonl" \
    --real-manifest "$SOURCE_ROOT/data/manifests/corpus-v2/validation.jsonl" \
    --real-manifest "$SOURCE_ROOT/data/manifests/corpus-v2/test.jsonl" \
    --output "$AUDIT/city" \
    --overwrite

echo
echo "=== COMPACT STRUCTURAL RESULTS ==="
python - <<PY
import json
from pathlib import Path

d = json.loads((Path(r"$AUDIT") / "roads" / "summary.json").read_text())
metrics = [
    "road_components",
    "road_length_km",
    "largest_length_fraction",
    "interior_component_length_fraction",
    "junctions_per_km",
    "interior_dead_ends_per_km",
    "boundary_endpoints",
    "edge_length_median_m",
    "edge_length_p90_m",
    "major_length_share",
    "secondary_length_share",
    "local_length_share",
    "local_direct_higher_contact_fraction",
    "secondary_direct_major_contact_fraction",
    "unnoded_crossings",
    "overlap_pair_count",
]

print(f"{'metric':43s} {'generated':>12s} {'real':>12s}")
print("-" * 70)
for metric in metrics:
    c = d["comparison"].get(metric)
    if c:
        print(
            f"{metric:43s} "
            f"{c['generated_median']:12.4f} "
            f"{c['real_median']:12.4f}"
        )
PY

echo
echo "=== PACKAGE ==="
mkdir -p \
    "$PACKAGE/training" \
    "$PACKAGE/program-corpus" \
    "$PACKAGE/sample-previews" \
    "$PACKAGE/sample-programs"

for name in config.json environment.json metrics.jsonl summary.json; do
    [[ -f "$RUN/$name" ]] && cp "$RUN/$name" "$PACKAGE/training/$name"
done

for name in summary.json style-stats.json codec.json train-rejected.json validation-rejected.json test-rejected.json; do
    [[ -f "$PROGRAMS/$name" ]] && cp "$PROGRAMS/$name" "$PACKAGE/program-corpus/$name"
done

cp -r "$AUDIT" "$PACKAGE/audit"

for folder in low mid high; do
    [[ -f "$SAMPLES/$folder/summary.json" ]] && cp "$SAMPLES/$folder/summary.json" "$PACKAGE/$folder-summary.json"
done

while IFS= read -r preview; do
    relative="${preview#$SAMPLES/}"
    safe="${relative//\//__}"
    cp "$preview" "$PACKAGE/sample-previews/$safe"
done < <(find "$SAMPLES" -path '*/sample-*/preview.png' -type f | sort)

while IFS= read -r program; do
    relative="${program#$SAMPLES/}"
    safe="${relative//\//__}"
    cp "$program" "$PACKAGE/sample-programs/$safe"
done < <(find "$SAMPLES" -path '*/sample-*/program.json' -type f | sort)

# Keep representative generated programs/cities/OBJs selected by the city audit.
if [[ -d "$AUDIT/city/representatives" ]]; then
    cp -r "$AUDIT/city/representatives" "$PACKAGE/representatives"
fi

cat > "$PACKAGE/provenance.txt" <<EOF
experiment_commit=$COMMIT
config=$CONFIG
gpu_physical_index=$GPU_INDEX
training_target=largest connected surface-road component per corrected real tile
geometry_encoding=parent-relative ADD displacement, max 200 m training segments
generation_constraint=crossing-safe local surface-road segments
comparison_reference=largest connected real surface-road component
generation_maximum_components=1
EOF

cd "$SOURCE_ROOT"
zip -qr "$ZIP" "runs/$(basename "$PACKAGE")"

echo
echo "=== COMPLETE ==="
ls -lh "$ZIP"
echo "Upload: $ZIP"
