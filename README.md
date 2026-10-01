# Urban City Generation

A research project on **context-conditioned structured urban generation** for Unreal Engine PCG.

The goal is not to generate a city as an image and trace it afterwards. The model should reason over structured spatial context and produce a city state that can be consumed directly as roads, rail, buildings and urban-area geometry.

The current active branch is `context-plan-graph-v1`.

## Research question

**Can a context-conditioned structured generative model learn to generate a coherent 512 m urban region from surrounding city context, including transport topology, building form and semantic urban spaces, while preserving cross-boundary continuity better than an equivalent model without regional context?**

This gives the project a measurable comparison rather than only asking whether generated cities look realistic.

The main questions underneath it are:

- can the model infer road and rail structure inside a hidden region from the surrounding city;
- can it preserve junction topology, hierarchy and boundary continuation;
- can the same structured representation include building footprints, building heights and land-use/environment polygons;
- does regional context improve these results compared with a no-context ablation;
- can the generated state be exported to Unreal Engine without relying on raster tracing as the source of truth.

## Intended learning outcomes

By the end of the project, the work should demonstrate:

1. **Structured geospatial representation**  
   Converting real city data into graph, polygon and contextual features suitable for machine learning instead of treating the city only as a raster image.

2. **Context-conditioned generative modelling**  
   Designing and training models that use surrounding urban structure, boundary ports and regional descriptors to infer a hidden local city region.

3. **Urban topology and geometry evaluation**  
   Evaluating more than visual similarity, including junction error, connectivity, component structure, boundary continuation, road/rail class, building geometry and semantic-area geometry.

4. **3D-ready procedural handoff**  
   Exporting learned structured outputs as explicit coordinates and attributes that Unreal Engine PCG can use for splines, buildings, land-use regions and environment generation.

## Final project deliverable

The target end-of-project system is:

```text
prepared city data
        |
        v
multi-kilometre structured context
        |
        v
context-conditioned neural generator
        |
        v
512 m generated structured city state
        |
        +-- road and rail graph
        +-- junction and spline geometry
        +-- road/rail class and vertical mode
        +-- building footprints
        +-- building type and height
        +-- green and water polygons
        +-- residential/commercial/industrial/civic areas
        |
        v
JSON / Unreal Engine PCG
```

The final research deliverables should include:

- a reproducible dataset pipeline from real city data;
- a trained context-conditioned model that generates structured 512 m city regions;
- a held-out evaluation and a matched no-context ablation;
- generated examples containing transport, buildings and semantic urban spaces;
- structured JSON suitable for an Unreal Engine importer;
- an Unreal or structural 3D demonstration showing that the learned city state can drive scene generation.

Metric transport Z is only part of the deliverable if reliable elevation supervision is available. Until then, the model should predict vertical categories such as surface, underground and elevated without inventing fake metric heights.

## Current approach

The project has moved through several raster and graph baselines. The current work uses a continuous city context representation rather than unrelated tiles.

### Spatial context

The current Singapore experiments use:

- **512 m x 512 m target region**;
- **2048 m x 2048 m regional context**;
- **1536 m x 1536 m detailed visible transport context**;
- explicit road and rail boundary ports;
- structured buildings, green, water and land-use targets.

The target interior is hidden from the input. Boundary ports describe transport entering the region but do not prescribe the route inside it.

### Context-plan transport model

The most developed current model is a context-conditioned hierarchical Transformer for transport-graph infilling.

```text
context cells + visible transport + boundary ports
                    |
                    v
          shared context encoder
                    |
                    v
             coarse 8x8 planner
                    |
                    v
          graph architect Transformer
                    |
                    +-- junction positions
                    +-- road / rail mode
                    +-- vertical category
                    +-- boundary probability
                    +-- junction degree
                    +-- edge existence
                    +-- edge class
                    +-- edge width
                    +-- curve geometry
```

A degree-aware discrete decoder selects a graph that is consistent with the model's learned edge and degree predictions. It does not procedurally decide where roads should go.

The current overfit experiment is an architecture-debugging run on 16 samples. Geometry fine-tuning reduced junction error substantially on that memorised set, but generated topology and boundary continuation still need improvement. These results are not evidence of held-out generalisation.

### Full structured-city experiment

The repository also contains a joint structured neural model, `StructuredCityDenoiser`, which directly predicts:

```text
transport:
  node count and position
  edge count and connectivity
  road / rail mode
  class
  vertical mode
  width
  spline geometry

buildings:
  building count
  footprint geometry
  building type
  height

urban areas:
  area count
  polygon geometry
  green
  water
  residential
  commercial / mixed
  industrial
  civic
```

This model is experimental. Earlier joint slot-denoising runs failed on held-out generation because of variable-cardinality and permutation problems. It is currently useful as a small overfit test for whether all scene layers can be learned jointly, not as a validated final architecture.

The longer-term direction is to keep the stronger context and transport reasoning from the context-plan model while using suitable learned representations for buildings and semantic ground areas.

## Data representation

The city is represented as structured geometry rather than one mutually exclusive image.

### Transport graph

Nodes contain position, transport mode, vertical category and boundary information.

Edges contain:

- from/to nodes;
- road or rail mode;
- hierarchy/class;
- vertical mode;
- width;
- polyline geometry.

### Buildings

Building targets contain:

- footprint polygon;
- building type;
- height where supported by source evidence;
- height confidence/source metadata.

### Urban areas

Area targets include:

- green;
- water;
- residential;
- commercial/mixed;
- industrial;
- civic.

These layers may overlap where the source semantics permit it. They are not reduced to a single colour per pixel.

## Setup

```bash
git clone https://github.com/hahanoitsu/urban-city-generation.git
cd urban-city-generation
git switch context-plan-graph-v1

conda env create -f environment.yml
conda activate urban-city
python -m pip install -e '.[ml]'
```

## Current transport experiment

For the existing context-plan repair/fine-tuning workflow and interpretation of the comparison panels, see `docs/transport_repair.md`.

A checkpoint can be sampled with:

```bash
python scripts/sample_context_plan_graph.py \
  --data /path/to/spatial-world-data \
  --checkpoint /path/to/best.pt \
  --output /path/to/previews \
  --samples 6 \
  --compare-decoders \
  --save-predictions
```

## Full structured-city overfit experiment

A small full-scene run can test whether transport, buildings and semantic urban areas can all be memorised by the neural model:

```bash
python scripts/train_structured_city.py \
  --data /path/to/context-graph-v1/singapore \
  --output /path/to/run \
  --cache-dir /path/to/cache \
  --maximum-samples 16 \
  --overfit \
  --nodes 448 \
  --edges 512 \
  --buildings 512 \
  --areas 256 \
  --batch-size 1 \
  --epochs 40 \
  --save-every 5
```

Generate directly from the trained network:

```bash
python scripts/sample_structured_city.py \
  --data /path/to/context-graph-v1/singapore \
  --checkpoint /path/to/run/best.pt \
  --output /path/to/run/generations \
  --cache-dir /path/to/cache \
  --samples 6 \
  --steps 40 \
  --split all
```

The generated JSON contains neural predictions for transport, buildings and semantic areas. Rendering that JSON is a visualisation step, not the source of the generated geometry.

## Evaluation

The final research test should use geographic train/validation/test separation rather than the current memorisation runs.

The main comparison is:

```text
same architecture + regional context
vs
same architecture without regional context
```

Useful evaluation measures include:

### Transport

- junction position error;
- edge precision/recall where correspondence is available;
- junction degree distribution;
- connected components;
- largest connected component fraction;
- boundary-port alignment;
- road/rail class and vertical-mode accuracy;
- spline geometry error.

### Buildings

- footprint geometry error / overlap;
- building count;
- building type accuracy;
- height error where height supervision is valid.

### Urban areas

- area count;
- class accuracy;
- polygon overlap or boundary distance;
- green/water/land-use coverage.

Visual inspection remains important, but it should support rather than replace quantitative evaluation.

## Unreal Engine path

The intended handoff is structured JSON, not a preview image.

Unreal Engine PCG should receive attributes such as:

```text
transport spline points
width and hierarchy
vertical mode
building footprints
building heights and types
green and water polygons
land-use regions
```

Unreal is responsible for scene assets and detailed rendering. It should not be responsible for repairing a disconnected learned city topology.

## Current limitations

- the strongest transport results are still from a 16-sample overfit/debugging experiment;
- the graph stage is trained with target plans, while generation uses predicted plans;
- boundary ports are conditioning signals rather than guaranteed attachments;
- the current transport model does not generate buildings or semantic areas;
- the joint structured-city model has not yet demonstrated good held-out generation;
- exact metric transport Z is not supervised;
- multi-city generalisation has not yet been established.

These are active research problems, not hidden post-processing assumptions.
