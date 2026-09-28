#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_ROOT="${MAIN_ROOT:-$(dirname "$SCRIPT_ROOT")/urban-city-generation}"
CITY="${CITY:-$MAIN_ROOT/data/cities/singapore-v2.gpkg}"
DATA="${REPAIR_DATA:-$MAIN_ROOT/data/spatial-world-repair-512}"
RUN="${REPAIR_RUN:-$MAIN_ROOT/runs/context-plan-graph-repair-$(date +%Y%m%d-%H%M%S)}"

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

export PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SCRIPT_ROOT"

python -m pytest -q tests/test_transport_repairs.py tests/test_plan_cell_graph.py tests/test_context_plan_graph.py tests/test_graph_decode.py

if [[ ! -f "$DATA/summary.json" ]]; then
    if [[ -e "$DATA" ]]; then
        echo "An incomplete dataset exists at $DATA. Set REPAIR_DATA to a new directory." >&2
        exit 1
    fi
    python scripts/build_spatial_world_v1.py \
        --city "$CITY" --output "$DATA" \
        --target-size 512 --stride 512 --context-size 2048 \
        --local-vector-size 1536 --context-cell 256 \
        --maximum-samples "${BUILD_SAMPLES:-64}"
fi

python - "$DATA" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads((Path(sys.argv[1]) / "summary.json").read_text())
if summary["version"] != "0.2.0" or summary["config"]["target_size_m"] != 512:
    raise SystemExit("This run needs the repaired 512 m dataset. Set REPAIR_DATA to a new path.")
PY

EXTRA_ARGS=()
if [[ "${NO_CONTEXT:-0}" == 1 ]]; then
    EXTRA_ARGS+=(--no-context)
fi
python scripts/train_context_plan_graph.py \
    --data "$DATA" --output "$RUN" --overfit \
    --maximum-samples "${MAXIMUM_SAMPLES:-16}" \
    --epochs "${EPOCHS:-200}" --batch-size "${BATCH_SIZE:-2}" \
    --grid-size 8 --save-every 50 "${EXTRA_ARGS[@]}"

python scripts/sample_context_plan_graph.py \
    --data "$DATA" --checkpoint "$RUN/best.pt" \
    --output "$RUN/previews" --samples 6 --compare-decoders --save-predictions

python - "$RUN" <<'PY'
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

run = Path(sys.argv[1])
destination = run.with_suffix(".zip")
with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".png", ".npz"}:
            archive.write(path, path.relative_to(run))
print(f"Results: {destination}")
PY
