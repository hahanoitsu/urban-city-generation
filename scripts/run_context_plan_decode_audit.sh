#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?Usage: bash scripts/run_context_plan_decode_audit.sh /path/to/run}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "urban-city" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    else
        for CANDIDATE in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/mambaforge"; do
            if [[ -f "$CANDIDATE/etc/profile.d/conda.sh" ]]; then
                CONDA_BASE="$CANDIDATE"
                break
            fi
        done
    fi
    source "${CONDA_BASE:?Activate the urban-city conda environment first}/etc/profile.d/conda.sh"
    conda activate urban-city
fi

RUN="$(cd "$RUN" && pwd)"
export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SCRIPT_ROOT"

DATA="${REPAIR_DATA:-$(python - "$RUN/experiment.json" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["arguments"]["data"])
PY
)}"
OUTPUT="${DECODE_OUTPUT:-$RUN/decode-audit-$(date +%Y%m%d-%H%M%S)}"

python scripts/sample_context_plan_graph.py \
    --data "$DATA" --checkpoint "$RUN/best.pt" \
    --output "$OUTPUT" --samples "${AUDIT_SAMPLES:-6}" \
    --compare-decoders --save-predictions

python - "$RUN" "$OUTPUT" <<'PY'
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

run, output = map(Path, sys.argv[1:])
destination = output.with_suffix(".zip")
with ZipFile(destination, "x", ZIP_DEFLATED) as archive:
    for path in sorted(output.iterdir()):
        if path.is_file():
            archive.write(path, path.name)
    for name in ("experiment.json", "metrics.jsonl"):
        if (run / name).is_file():
            archive.write(run / name, name)
print(f"Results: {destination}")
PY
