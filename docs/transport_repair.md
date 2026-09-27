# Transport repair experiment

This branch repairs the transport representation and connects the existing context planner to the cell graph decoder. It is an experiment, not a completed 3D city generator.

## What changed

- Missing OSM layers and explicit layer 0 use the same junction key.
- Graph construction keeps source way boundaries so class and width changes survive.
- Simplification keeps layer, bridge, tunnel, width and direction metadata. Unknown stacking stays separate.
- Loops and parallel paths gain intermediate shape nodes before entering the simple adjacency decoder. No path is dropped to fit one edge per node pair.
- Curves use both x and y residuals. The old normal-only curve could not represent backtracking.
- Coordinate arithmetic uses float32 even when attention runs in bfloat16.
- Partially hidden context cells no longer reveal target statistics.
- The context planner and graph decoder share the context encoder. Generation predicts its own plan and counts.

The first run uses 512 m targets, 2,048 m statistical context, and 1,536 m visible transport context. It builds a deterministic scattered subset from the GeoPackage. Context statistics are computed as needed instead of scanning the whole context grid first.

## Run on the server

From a checkout of this branch:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_transport_repair.sh
```

The runner uses `urban-city`, builds up to 64 samples, and fits 16 of them for 200 epochs. It uses one GPU and writes a new run directory. It does not delete existing datasets or runs. Set `MAIN_ROOT` if the original repository is not the adjacent `urban-city-generation` directory.

```bash
EPOCHS=100 MAXIMUM_SAMPLES=8 CUDA_VISIBLE_DEVICES=0 bash scripts/run_transport_repair.sh
```

The shorter command is a debugging run. More epochs are useful only if reconstruction is improving. An overfit experiment uses the same examples for training and evaluation, so its evaluation loss is not evidence of generalization.

The final line prints a ZIP containing previews, graph JSON, experiment settings and metrics. Checkpoints stay in the run directory. If interrupted after training, run:

```bash
python scripts/sample_context_plan_graph.py \
  --data /path/to/spatial-world-repair-512 \
  --checkpoint /path/to/run/best.pt \
  --output /path/to/run/new-previews --samples 6 --seeds 7 19 37
```

## Read the comparisons

Each image shows the target, reconstruction with the target plan, and three generations from context. The generated columns receive no target nodes, edges, plan or target density controls. They still use the visible surroundings and the supplied city style statistics.

Reconstruction uses the target edge count and reports node error and edge precision/recall. Generation uses predicted counts. Its node IDs do not correspond to the target's IDs, so it reports graph statistics rather than misleading indexed edge recall.

Generation samples cell occupancy and a shifted Poisson distribution for node counts. This is a simple stochastic baseline. It is not a learned city-level latent distribution. The decoder selects the highest scoring same-mode pairs up to its predicted edge budget. It does not repair the graph to match a target component count. Connectivity is still something to measure, not a guarantee.

The graph decoder is trained with target plans. Poor generated plans can still cause poor graphs even when reconstruction works. Compare both columns before changing the architecture again.

## Remaining gaps

- Buildings, water, green space and land use contribute contextual statistics; this experiment generates transport only.
- The graph carries x/y coordinates and vertical categories. Metric z is unavailable. The old compiler's default deck heights are removed from these training targets.
- Bridge approaches can still be disconnected when the source uses different vertical groups. Coincident x/y positions alone are not sufficient proof of an OSM connection. Source node identity needs to be retained during extraction to resolve these reliably.
- Boundary ports are context inputs, not hard attachment constraints. Adjacent generated targets are not yet guaranteed to join.
- Eight interior curve samples remain an approximation of source polylines.
- Direction metadata survives preprocessing but is not a prediction head yet. The generated graph is not a traffic-ready lane network.
- Multi-city mixing and joint building/transport generation have not been trained or validated.

## Research test after the debugging run

The broad idea of contextual urban generation already has related work, including [COHO](https://arxiv.org/abs/2407.11294) and [ScenarioDreamer](https://princeton-computational-imaging.github.io/scenario-dreamer/). The question to test is whether regional context improves joint road/rail continuity and vertical relationships compared with the same model without that context.

For a held-out experiment, build a larger dataset without `--maximum-samples`, then call `train_context_plan_graph.py` without `--overfit`. It divides each city's longest spatial axis into train/validation/test regions and excludes boundary windows whose context would cross those partitions. Normalization is fitted on training samples and saved in the checkpoint. City style remains a supplied condition, so this is not an unseen-city test.

Repeat with `--no-context` using the same data, seed and settings. The ablation removes regional cells, visible vectors and ports while retaining city style. Compare connectivity, largest component size, road/rail errors, boundary alignment and diversity on held-out regions. An improvement on a small memorization run is not a novelty result.

Unreal PCG can consume points, splines and attributes through an importer. It can place meshes and detail around this structure. It will not establish that a disconnected generated transport network is correct.
