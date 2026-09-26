#!/usr/bin/env bash
set -euo pipefail

branches=(
  citygen-tree-test
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

for branch in "${branches[@]}"; do
    sha="$(git rev-parse "origin/$branch")"
    tag="archive/$branch"
    git tag -f "$tag" "$sha"
    git push -f origin "refs/tags/$tag"
done

git push origin --delete "${branches[@]}"
git fetch origin --prune
git branch -r
