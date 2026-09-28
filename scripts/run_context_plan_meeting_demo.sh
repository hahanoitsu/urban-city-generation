#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="${1:?Usage: bash scripts/run_context_plan_meeting_demo.sh /path/to/source-run}"

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

DATA="${MEETING_DATA:-$(python - "$SOURCE_RUN/experiment.json" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text())["arguments"]["data"])
PY
)}"

RUN="${MEETING_RUN:-$(dirname "$SOURCE_RUN")/context-plan-meeting-demo-$(date +%Y%m%d-%H%M%S)}"
EPOCHS="${EPOCHS:-5}"
SAMPLES="${SAMPLES:-3}"

if [[ -e "$RUN" ]]; then
    echo "Output already exists: $RUN" >&2
    exit 1
fi
mkdir -p "$RUN"

echo "source=$SOURCE_RUN"
echo "data=$DATA"
echo "output=$RUN"
echo "epochs=$EPOCHS"
echo "samples=$SAMPLES"

python scripts/sample_context_plan_graph.py     --data "$DATA"     --checkpoint "$SOURCE_RUN/best.pt"     --output "$RUN/before"     --samples "$SAMPLES"     --compare-decoders

python scripts/train_context_plan_graph.py     --data "$DATA"     --init-checkpoint "$SOURCE_RUN/best.pt"     --output "$RUN/training"     --epochs "$EPOCHS"     --batch-size "${BATCH_SIZE:-2}"     --learning-rate "${LEARNING_RATE:-0.00005}"     --geometry-scale-m 10     --count-error-weight 0.1     --save-every 1 | tee "$RUN/training.log"

python scripts/sample_context_plan_graph.py     --data "$DATA"     --checkpoint "$RUN/training/best.pt"     --output "$RUN/after"     --samples "$SAMPLES"     --compare-decoders

python scripts/export_context_plan_city_demo.py     --input "$RUN/after"     --output "$RUN/city-demo"

python - "$RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
before = json.loads((run / "before/summary.json").read_text())
after = json.loads((run / "after/summary.json").read_text())

print()
print(f"checkpoint epoch: {before['epoch']} -> {after['epoch']}")
print(f"raw graph comparisons: {run / 'after'}")
print(f"city previews: {run / 'city-demo'}")
print(f"training log: {run / 'training.log'}")
PY
