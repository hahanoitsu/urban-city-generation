#!/usr/bin/env bash
set -euo pipefail

branches=(
  citygen-tree-test
  context-graph-model-v1
  context-graph-v1
  cuda-training-v1
  dev
  diagnose-diffusion-sampling
  gpkg-source-audit
  layered-distribution-v2
  learned-3d-scene-v1
  mixture-vertical-generator
  morphology-analysis
  morphology-control-1024-audit
  morphology-control-1024-loss-probe
  morphology-control-1024-v1
  morphology-control-composition
  morphology-control-v1
  seeded-extension-clean-temp
  structural-generator-v1
  structural-generator-v2
  structural-generator-v3
  structured-output-v1
  unreal-pcg-research-v1
  vectorizer-roundtrip-fix
  whole-singapore-overfit
  whole-singapore-overfit-v2
)

git fetch origin --prune

deletions=()
leases=()
for branch in "${branches[@]}"; do
    if ! git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
        continue
    fi
    sha="$(git rev-parse "origin/$branch")"
    tag="archive/$branch"
    git push origin "$sha:refs/tags/$tag"
    deletions+=(":refs/heads/$branch")
    leases+=("--force-with-lease=refs/heads/$branch:$sha")
done

if [[ ${#deletions[@]} -eq 0 ]]; then
    echo "No old branches remain."
    exit 0
fi

git push --atomic "${leases[@]}" origin "${deletions[@]}"
git fetch origin --prune
git branch -r
