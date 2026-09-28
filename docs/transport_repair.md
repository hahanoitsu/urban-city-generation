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
  --output /path/to/run/new-previews --samples 6 \
  --compare-decoders --save-predictions
```

## Read the comparisons

Each comparison image has five columns: the target, reconstruction with old edge selection, reconstruction with predicted degree scores, context generation with old edge selection, and context generation with predicted degree scores. Both reconstruction columns use the target plan. Both generated columns use the same deterministic predicted plan. They receive no target nodes, edges, plan or target density controls. They still use the visible surroundings and the supplied city style statistics.

The old reconstruction decoder uses the target edge count and reports node error and edge precision/recall. The degree decoder uses the predicted edge and node degree scores without a supplied edge budget. Generation uses predicted node counts. Its node IDs do not correspond to the target's IDs, so it reports graph statistics rather than indexed edge recall. Comparing the two generation columns isolates edge selection. Comparing their plan counts with the target measures a separate source of error.

Generation is deterministic by default. Supplying `--seeds 7 19 37` adds independent occupancy and shifted Poisson count noise as a separate experiment. That noise is not a learned city-level latent distribution. Changing cell counts also changes the decoder's node slots, which were trained using target counts.

The experimental degree decoder uses each node's most likely predicted degree as its capacity. Within these capacities, it jointly selects edges and degree states using edge logits and learned degree probabilities. It estimates the training BCE class-weight correction from predicted degrees, so calibration is approximate. It allows road-to-road and rail-to-rail edges, without a fixed degree limit such as four, a triangle ban, a geometric template, or a component-count constraint. It does not move nodes or smooth curves. A five-second solver limit applies to each graph; the JSON records the result, gap and any fallback. If no feasible solver result is available, the fallback accepts positive corrected edge scores only while both endpoints remain below their predicted degrees. Wrong degree predictions can now suppress real connections, so this experiment still needs comparison on the trained checkpoint.

The graph decoder is trained with target plans. Poor generated plans can still cause poor graphs even when reconstruction works. Compare both columns before changing the architecture again.

## Check the existing 200-epoch run

The run `context-plan-graph-repair-20260928-064549` selected epoch 195. Its logged mean node error was 9.43 m and curve residual error was 5.31 m. The low total loss did not mean that road geometry was accurate.

In preview `singapore_w+00278_+00725`, the target had five triangles. Reconstruction had 61 and left 50 nodes isolated. The three generated graphs had 64 to 90 triangles. These counts show that connections accumulated in small clusters even with the target plan supplied.

Two inference problems are now addressed: count conversion no longer uses unconstrained absent-mode counts to set the road/rail ratio, and independent count noise is optional. The degree comparison checks a third problem: the old global edge ranking ignored the already-trained degree head. Fixing selection cannot repair position or curve errors stored in the checkpoint.

Run the comparison without building data or training:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_context_plan_decode_audit.sh /path/to/run
```

This loads `best.pt` and the data path in `experiment.json`. It writes a new `decode-audit-<time>` folder and ZIP inside the existing run. Set `REPAIR_DATA` if the dataset moved. The ZIP includes graph JSON, PNG comparisons, metrics and compressed prediction arrays. The arrays contain the edge and degree logits, coordinates, curves and planner predictions so later decoding checks can run without the model weights. No checkpoint is changed.

The overfit set has only 16 examples. Any improvement here is a debugging result. It does not establish generalization, diverse generation or novelty.

## Audit result and geometry fine-tuning

The six examples in `decode-audit-20260928-114140` contain 518 target edges. Degree selection raised reconstruction recall from 372/518 (71.8%) to 417/518 (80.5%), and removed all 75 isolated nodes. The dense example went from 61 triangles to 13, against five in the target. Generation still had 50 triangles across the six examples, against 169 with the old selection. Removing isolated nodes did not always improve the largest connected component.

The saved predictions exposed another inference mismatch. Corridor directions are trained only where that corridor exists. The target plan supplies zeros elsewhere, but generation was passing the unconstrained predictions through. Absent direction channels averaged about 0.36 to 0.45 in absolute value. They are now zeroed using the predicted corridor presence. This changes generated conditioning without supplying target data.

The old geometry loss used cell-relative coordinates for junctions and whole-window coordinates for curves. In a 512 m window, a four-metre curve error is only 0.0156 in normalized coordinates. The new optional metric loss applies the same ten-metre scale and one-metre Huber transition to junction positions and curve residuals. It still supervises real bends and their second differences. `edge_shape_mae_m` measures interior point error including both endpoint and curve errors.

Fine-tune the saved model and compare both stages:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_context_plan_finetune.sh /path/to/run
```

This first exports the existing checkpoint with the orientation fix. It then runs 80 additional epochs at a learning rate of 0.00005 using the saved model configuration, sample IDs and normalization. The geometry scale is 10 m. An additional count loss of weight 0.1 supervises actual conditional counts rather than only their logarithms. The optimizer starts fresh because the objective changed. `EPOCHS`, `LEARNING_RATE`, `AUDIT_SAMPLES` and `FINETUNE_RUN` can override the runner settings.

The result directory has `before`, `training` and `after` folders. `best.pt` starts as a copy of the initial weights and is replaced only when validation improves under the new objective. The old and new total losses use different scales; compare metre errors, decoded cell counts and exported graph metrics instead. The runner packages the comparisons and predictions in one ZIP. This remains an overfit experiment with target-plan graph supervision; it is not validation of multi-city or diverse generation.

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
