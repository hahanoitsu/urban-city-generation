#!/usr/bin/env bash
set -euo pipefail

ROOT="${URBAN_ROOT:-$(git rev-parse --show-toplevel)}"
OUTROOT="$ROOT/runs/gpkg-source-audit"
ZIP="$ROOT/gpkg-source-audit-results.zip"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"

rm -rf "$OUTROOT"
rm -f "$ZIP"
mkdir -p "$OUTROOT"

FILES=()
for candidate in     "$ROOT/data/cities/singapore-v2.gpkg"     "$ROOT/data/cities/singapore.gpkg"; do
    if [[ -f "$candidate" ]]; then
        FILES+=("$candidate")
    fi
done

if [[ "${#FILES[@]}" -eq 0 ]]; then
    echo "No Singapore GeoPackage found under data/cities/"
    exit 1
fi

echo "=== FILES TO AUDIT ==="
printf '  %s\n' "${FILES[@]}"

for gpkg in "${FILES[@]}"; do
    name="$(basename "$gpkg" .gpkg)"
    echo
    echo "=== AUDIT: $name ==="
    python scripts/audit_singapore_gpkg.py         --gpkg "$gpkg"         --output "$OUTROOT/$name"
done

echo
echo "=== QUICK COMPARISON ==="
python - <<'PY'
import json
from pathlib import Path

root = Path("runs/gpkg-source-audit")
for folder in sorted(p for p in root.iterdir() if p.is_dir()):
    summary = json.loads((folder / "summary.json").read_text())
    print(f"\n{folder.name}")
    for layer in ("roads", "rail"):
        item = summary["layer_summaries"].get(layer)
        if not item:
            continue
        print(f"  {layer}: {item['total_length_km']:.3f} km")
        for mode in item["modes"]:
            print(
                f"    {mode['mode']:12s} "
                f"{mode['length_km']:8.3f} km "
                f"{mode['length_fraction']*100:6.2f}%"
            )
        print("    top issues:")
        for issue in item["issues"][:8]:
            print(
                f"      {issue['issue']:48s} "
                f"{int(issue['features']):5d} "
                f"{float(issue['length_km']):8.3f} km"
            )
PY

echo
echo "=== PACKAGE ==="
cd "$ROOT/runs"
zip -qr "$ZIP" gpkg-source-audit

echo
echo "=== COMPLETE ==="
ls -lh "$ZIP"
echo "Upload: $ZIP"
