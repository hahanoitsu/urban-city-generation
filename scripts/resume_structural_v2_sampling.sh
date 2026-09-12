#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKROOT="$(dirname "$SOURCE_ROOT")/urban-city-generation-structural-v2-resume"
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
git fetch origin structural-generator-v2
COMMIT="$(git rev-parse origin/structural-generator-v2)"
echo "sampling commit: $COMMIT"

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

test -f "$RUN/best.pt" || {
    echo "Missing trained checkpoint: $RUN/best.pt"
    exit 1
}
test -f "$PROGRAMS/style-stats.json" || {
    echo "Missing prepared structural-v2 corpus: $PROGRAMS"
    exit 1
}

cd "$WORKROOT"
python -m py_compile src/urban_ai/generate.py src/urban_ai/sampling.py
pytest -q \
    tests/test_structural_planarity.py \
    tests/test_structural_relative_codec.py \
    tests/test_structural_densify.py

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

rm -rf "$SAMPLES" "$AUDIT" "$PACKAGE"
rm -f "$ZIP"
mkdir -p "$SAMPLES" "$AUDIT" "$PACKAGE"

for spec in low:0.40 mid:0.65 high:0.85; do
    label="${spec%%:*}"
    temperature="${spec##*:}"
    echo
    echo "=== SAMPLE $label temperature=$temperature ==="
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

if [[ -d "$AUDIT/city/representatives" ]]; then
    cp -r "$AUDIT/city/representatives" "$PACKAGE/representatives"
fi

cat > "$PACKAGE/provenance.txt" <<EOF
experiment_commit=$COMMIT
checkpoint=$RUN/best.pt
sampling_only_resume=true
gpu_physical_index=$GPU_INDEX
EOF

cd "$SOURCE_ROOT"
zip -qr "$ZIP" "runs/$(basename "$PACKAGE")"

echo
echo "=== COMPLETE ==="
ls -lh "$ZIP"
echo "Upload: $ZIP"
