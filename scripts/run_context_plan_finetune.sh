#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="${1:?Usage: bash scripts/run_context_plan_finetune.sh /path/to/run}"

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

SOURCE_RUN="$(cd "$SOURCE_RUN" && pwd)"
export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SCRIPT_ROOT"

DATA="${REPAIR_DATA:-$(python - "$SOURCE_RUN/experiment.json" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["arguments"]["data"])
PY
)}"
RUN="${FINETUNE_RUN:-$(dirname "$SOURCE_RUN")/context-plan-geometry-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$RUN" ]]; then
    echo "Output already exists: $RUN. Set FINETUNE_RUN to a new path." >&2
    exit 1
fi
mkdir -p "$RUN"

python scripts/sample_context_plan_graph.py \
    --data "$DATA" --checkpoint "$SOURCE_RUN/best.pt" \
    --output "$RUN/before" --samples "${AUDIT_SAMPLES:-6}" \
    --compare-decoders --save-predictions

python scripts/train_context_plan_graph.py \
    --data "$DATA" --init-checkpoint "$SOURCE_RUN/best.pt" \
    --output "$RUN/training" --epochs "${EPOCHS:-80}" \
    --batch-size "${BATCH_SIZE:-2}" --learning-rate "${LEARNING_RATE:-0.00005}" \
    --geometry-scale-m 10 --count-error-weight 0.1 --save-every 20

python scripts/sample_context_plan_graph.py \
    --data "$DATA" --checkpoint "$RUN/training/best.pt" \
    --output "$RUN/after" --samples "${AUDIT_SAMPLES:-6}" \
    --compare-decoders --save-predictions

python - "$RUN" <<'PY'
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

run = Path(sys.argv[1])
before = json.loads((run / "before/summary.json").read_text())
after = json.loads((run / "after/summary.json").read_text())
print(f"Compared checkpoint epochs {before['epoch']} and {after['epoch']}")
destination = run.with_suffix(".zip")
with ZipFile(destination, "x", ZIP_DEFLATED) as archive:
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".png", ".npz"}:
            archive.write(path, path.relative_to(run))
print(f"Results: {destination}")
PY
