#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKROOT="$(dirname "$SOURCE_ROOT")/urban-city-generation-structural-v3"
CONFIG="$SOURCE_ROOT/configs/structural-v3.yaml"
SAMPLES="$SOURCE_ROOT/runs/structural-v3-samples"
AUDIT="$SOURCE_ROOT/runs/structural-v3-audit"
PACKAGE="$SOURCE_ROOT/runs/structural-v3-package"
ZIP="$SOURCE_ROOT/singapore-structural-v3.zip"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

cd "$SOURCE_ROOT"
git fetch origin structural-generator-v3
COMMIT="$(git rev-parse origin/structural-generator-v3)"
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

cd "$WORKROOT"

echo
echo "=== TESTS ==="
python -m py_compile \
    src/urban_ai/planner.py \
    src/urban_analysis/structural_generator_audit.py
pytest -q \
    tests/test_structural_planner_v3.py \
    tests/test_structural_generator_audit.py \
    tests/test_generated_scene.py

echo
echo "=== GENERATE HIERARCHICAL PLANNER CITIES ==="
rm -rf "$SAMPLES" "$AUDIT" "$PACKAGE"
rm -f "$ZIP"

python -m urban_ai plan \
    --config "$CONFIG" \
    --output "$SAMPLES" \
    --count 24 \
    --seed 9301 \
    --overwrite

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
echo "=== COMPACT ROAD RESULTS ==="
python - <<PY
import json
from pathlib import Path

d = json.loads((Path(r"$AUDIT") / "roads" / "summary.json").read_text())
metrics = [
    "road_components",
    "road_length_km",
    "largest_length_fraction",
    "interior_component_length_fraction",
    "network_span_x_fraction",
    "network_span_y_fraction",
    "network_hull_area_fraction",
    "boundary_endpoints",
    "junctions_per_km",
    "interior_dead_ends_per_km",
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
    value = d["comparison"].get(metric)
    if value:
        print(
            f"{metric:43s} "
            f"{value['generated_median']:12.4f} "
            f"{value['real_median']:12.4f}"
        )
PY

echo
echo "=== PACKAGE ==="
mkdir -p "$PACKAGE/previews" "$PACKAGE/cities"
cp -r "$AUDIT" "$PACKAGE/audit"
cp "$SAMPLES/summary.json" "$PACKAGE/summary.json"

while IFS= read -r preview; do
    cp "$preview" "$PACKAGE/previews/$(basename "$(dirname "$preview")").png"
done < <(find "$SAMPLES" -path '*/sample-*/preview.png' -type f | sort)

while IFS= read -r city; do
    cp "$city" "$PACKAGE/cities/$(basename "$(dirname "$city")").json"
done < <(find "$SAMPLES" -path '*/sample-*/city.json' -type f | sort)

if [[ -d "$AUDIT/city/representatives" ]]; then
    cp -r "$AUDIT/city/representatives" "$PACKAGE/representatives"
fi

cat > "$PACKAGE/provenance.txt" <<EOF
experiment_commit=$COMMIT
generator=hierarchical_stochastic_planner
profile_source=data/manifests/corpus-v2/train.jsonl
geometry_copying=false
planner_uses=scalar real-road statistics only; geometry is newly sampled
EOF

cd "$SOURCE_ROOT"
zip -qr "$ZIP" "runs/$(basename "$PACKAGE")"

echo
echo "=== COMPLETE ==="
ls -lh "$ZIP"
echo "Upload: $ZIP"
