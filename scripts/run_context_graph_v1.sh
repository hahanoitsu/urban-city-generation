#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)/urban-city-generation"

if [[ -n "${URBAN_ROOT:-}" ]]; then
    DATA_ROOT="$URBAN_ROOT"
elif [[ -f "$MAIN_ROOT/data/cities/singapore-v2.gpkg" ]]; then
    DATA_ROOT="$MAIN_ROOT"
else
    DATA_ROOT="$SCRIPT_ROOT"
fi

CITY="${CITY_GPKG:-$DATA_ROOT/data/cities/singapore-v2.gpkg}"
OUTPUT="$DATA_ROOT/data/context-graph-v1/singapore"
AUDIT="$DATA_ROOT/context-graph-v1-audit.zip"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate urban-city

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$SCRIPT_ROOT"

echo "context graph v1"
echo "city: $CITY"
echo "2048 m context / 512 m target"

if [[ ! -f "$CITY" ]]; then
    echo "Missing prepared city: $CITY"
    exit 1
fi

echo
echo "running checks"
python -m py_compile     src/urban_dataset/context_graph.py     scripts/build_context_graph_v1.py
pytest -q tests/test_context_graph.py

echo
echo "building dataset"
rm -rf "$OUTPUT"

python scripts/build_context_graph_v1.py     --city "$CITY"     --output "$OUTPUT"     --region-size-m 2048     --target-size-m 512     --target-stride-m 512     --minimum-transport-length-m 40

echo
echo "summary"
cat "$OUTPUT/summary.json"

echo
echo "packing audit"
PACKAGE="/tmp/context-graph-v1-audit"
rm -rf "$PACKAGE"
mkdir -p "$PACKAGE/samples"

cp "$OUTPUT/summary.json" "$PACKAGE/"
cp "$OUTPUT/context-graph.json" "$PACKAGE/"
cp "$OUTPUT/targets.jsonl" "$PACKAGE/"
cp "$OUTPUT/context-graph-preview.png" "$PACKAGE/"
cp "$OUTPUT/target-preview.png" "$PACKAGE/" 2>/dev/null || true

python - "$OUTPUT" "$PACKAGE" <<'PY'
import gzip
import json
import shutil
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])

rows = [
    json.loads(line)
    for line in (source / "targets.jsonl").read_text().splitlines()
    if line.strip()
]
rows.sort(
    key=lambda row: (
        row["rail"] > 0,
        row["boundary_ports"],
        row["transport_length_m"],
    ),
    reverse=True,
)

for row in rows[:24]:
    path = source / row["sample_path"]
    shutil.copy2(path, target / "samples" / path.name)

summary = {
    "included_samples": min(24, len(rows)),
    "selection": "rail first, then boundary ports and transport length",
}
(target / "audit-samples.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

rm -f "$AUDIT"
cd /tmp
zip -qr "$AUDIT" "$(basename "$PACKAGE")"

echo
echo "done"
ls -lh "$AUDIT"
echo "full dataset: $OUTPUT"
echo "upload:       $AUDIT"
