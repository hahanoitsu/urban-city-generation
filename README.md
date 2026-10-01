# Urban City Generation

A research project on generating **large-scale, controllable 3D urban environments** from structured spatial data for Unreal Engine PCG.

The end goal is not a 512 m tile generator. The local 512 m target is a training and debugging unit inside a larger hierarchical system. The intended system should be able to build a city region-by-region while keeping a persistent structured city state so roads, rail, buildings, land use, green space and water remain coherent across boundaries.

The current active branch is `context-plan-graph-v1`.

## Research question

**Can a context-aware hierarchical generative model learn to generate coherent, controllable large-scale urban layouts from structured spatial data, while maintaining continuity across regions and producing a structured city state that can be realised as a 3D city in Unreal Engine PCG?**

The main questions underneath this are:

- can the model learn urban relationships from structured city data rather than raster images;
- can it generate roads and rail with realistic topology and continuation across local regions;
- can it jointly or hierarchically generate buildings, green space, water and land-use structure around that transport network;
- can regional context and persistent city state prevent independent local generations from becoming disconnected;
- can the generated structured state be passed directly to Unreal Engine PCG to construct a complete 3D urban environment.

## Learning outcomes

By the end of the project, the work should demonstrate:

1. **Structured geospatial modelling**  
   Representing a city using transport graphs, polygons, building geometry, land-use information and multi-scale spatial context.

2. **Hierarchical generative modelling**  
   Designing and training models that reason at more than one spatial scale: regional context, local urban structure and detailed geometry.

3. **Persistent large-scale generation**  
   Extending generation beyond an isolated window by keeping previously generated city structure as context and enforcing continuity at region boundaries.

4. **Urban topology and geometry evaluation**  
   Measuring connectivity, junction structure, boundary continuation, geometry, building layout and semantic-space quality rather than judging only by appearance.

5. **3D procedural integration**  
   Converting the learned structured city state into roads, rail, buildings, terrain/environment regions and other 3D scene elements in Unreal Engine PCG.

## Final deliverable

The final deliverable is a **generated 3D city in Unreal Engine PCG**, driven by a learned structured urban generator.

The intended end-to-end pipeline is:

```text
real structured city data
        |
        v
city ingester / dataset
        |
        v
multi-scale context representation
        |
        v
hierarchical neural city architect
        |
        v
persistent structured city state
        |
        +-- roads and rail
        +-- junctions and connectivity
        +-- building footprints and heights
        +-- land use
        +-- green space
        +-- water
        +-- vertical relationships
        |
        v
Unreal Engine PCG
        |
        v
complete 3D urban environment
```

The structured JSON/state is an intermediate representation, not the final product. Unreal Engine PCG is the final scene-realisation stage.

The final project should therefore include:

- a reproducible structured-city dataset pipeline;
- a trained generative model or hierarchy of models;
- large-scale generation that extends beyond one local target;
- continuity across generated regions;
- generated transport, buildings and environmental/land-use spaces;
- a structured handoff format for Unreal;
- an Unreal Engine PCG scene demonstrating the generated city in 3D;
- quantitative evaluation and relevant ablations of the learned model.

## Why 512 m targets are used

The current experiments use 512 m x 512 m detailed targets because they are small enough to train and inspect while still containing meaningful road, rail and block structure.

They are **not** the intended final city extent.

A typical local generation setup currently uses:

- 512 m x 512 m detailed target;
- about 2 km of regional context;
- visible neighbouring transport geometry;
- boundary ports indicating roads and rail entering the hidden target.

The longer-term system should repeatedly generate or refine neighbouring regions while carrying forward a persistent city graph/state. That is what allows the model to scale from a local training unit to a much larger city.

## Current model: context-plan graph

The strongest current learned component is a context-conditioned hierarchical Transformer for transport generation.

```text
regional context + visible transport + boundary ports
                         |
                         v
                 context encoder
                         |
                         v
                   coarse planner
                         |
                         v
                 graph architect
                         |
             +-----------+-----------+
             |                       |
             v                       v
       junction geometry       transport edges
                               class / width
                               road / rail
                               vertical mode
                               curve geometry
```

The model predicts its own local plan at generation time. A degree-aware decoder then chooses an edge set from the model's learned edge and degree predictions.

The current best transport results are still from a small overfit/debugging run. They show that the representation and model can learn meaningful junction geometry and transport structure, but topology, boundary attachment and held-out generalisation still need work.

## Full structured-city experiment

The repository also contains a neural structured-city model that predicts more than transport.

It has learned outputs for:

```text
transport
  nodes and positions
  edge connectivity
  road / rail class
  vertical mode
  width and spline geometry

buildings
  count
  footprint geometry
  type
  height

urban areas
  count
  polygon geometry
  green
  water
  residential
  commercial / mixed
  industrial
  civic
```

This joint model is currently experimental. Earlier direct slot-denoising runs exposed problems with variable-cardinality sets and polygon generation. It is useful for testing full-scene learning, but it is not yet the final architecture.

The likely final system will remain hierarchical: strong transport/context reasoning first, followed by learned generation of blocks, buildings and environmental/land-use spaces conditioned on the generated city structure.

## Data representation

The project uses structured geometry rather than one mutually exclusive city image.

### Transport

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
- height where supported by source data;
- height confidence/source metadata.

### Urban spaces

The dataset includes:

- green;
- water;
- residential land use;
- commercial/mixed land use;
- industrial land use;
- civic land use.

The goal is for these spaces to be generated as part of the city, not treated as empty background around roads and buildings.

## 3D and vertical structure

The final Unreal city is 3D.

The current dataset reliably supervises categorical vertical relationships such as:

```text
surface
underground
elevated
unknown
```

Exact metric road/rail Z is not yet treated as ground truth where the source data does not support it. The project should add metric terrain/elevation supervision when reliable data is available rather than inventing bridge or tunnel heights.

Buildings can use learned height where source supervision is available. Unreal PCG can then realise the generated structured state as actual 3D geometry.

## Evaluation

The final model should be evaluated on more than visual quality.

### Transport

- junction position error;
- edge/connectivity accuracy;
- connected components;
- largest connected component;
- junction degree;
- boundary continuation;
- road/rail class;
- vertical-mode accuracy;
- spline geometry error.

### Buildings

- building count;
- footprint geometry;
- building type;
- height error where height labels are valid;
- spatial relationship to transport and neighbouring buildings.

### Urban spaces

- green/water/land-use class;
- polygon geometry and overlap;
- coverage;
- spatial relationship with transport and buildings.

### Large-scale generation

- cross-region road and rail continuity;
- persistence of previously generated geometry;
- morphology drift across repeated expansion;
- consistency of district-scale structure.

## Unreal Engine PCG

Unreal Engine is not just a preview renderer for this project. It is the final scene-generation environment.

The learned model should provide structured data such as:

```text
road and rail splines
junctions
width and hierarchy
vertical mode / height information
building footprints
building heights and types
green and water polygons
land-use regions
persistent IDs and region relationships
```

Unreal PCG then turns this city state into the final 3D roads, buildings, terrain, vegetation and scene assets.

## Current limitations

- the current transport model operates on 512 m local targets rather than generating an entire city in one pass;
- persistent multi-region rollout is not complete yet;
- boundary ports condition generation but are not yet guaranteed attachments;
- the strongest transport checkpoint is still an overfit architecture test;
- the full structured-city model has not yet reached reliable held-out generation;
- metric transport Z still needs reliable elevation supervision;
- multi-city generalisation remains to be demonstrated.

These are the current research gaps between the working local model and the final whole-city Unreal deliverable.
