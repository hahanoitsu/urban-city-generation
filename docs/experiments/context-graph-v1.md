# Context graph v1

This experiment replaces isolated image tiles with a continuous city context layer.

It does not yet train a generator. The first gate is whether the dataset representation carries the information a learned structured generator would need without hiding urban planning inside procedural code.

## Scales

The first Singapore build uses two scales:

- context region: 2048 m × 2048 m;
- generation target: 512 m × 512 m.

The sizes are configuration values, not architectural constants.

## Whole-city context graph

The prepared city GeoPackage is treated as one continuous source.

Each context-region node stores measured urban descriptors and its position inside the city. Adjacent region nodes are connected by graph edges.

A region edge can contain data-derived transport continuation ports:

- road or rail;
- class;
- vertical mode;
- boundary position;
- heading;
- width where known;
- source feature id.

The context layer therefore knows that transport leaves one region and enters its neighbour. It does not decide where a road should be drawn inside either region.

Diagonal region edges are kept as spatial context edges. Only regions with a real shared boundary receive transport ports.

## Masked 512 m targets

Training examples are 512 m target windows sampled from the continuous city.

For each target the dataset stores:

Input:

- context-region ids;
- the parent context region marked as masked;
- visible neighbouring region ids;
- urban descriptors measured from the 2 km context with the target removed;
- road and rail ports crossing the target boundary.

Target:

- vector road polylines;
- vector rail polylines;
- building footprint polygons and height metadata;
- categorical vertical mode.

The target interior is not used to calculate the visible local context descriptors.

Boundary ports are an intentional continuation signal. They specify what transport enters the masked area, not the route it must take after entering.

## What is not procedural

This dataset does not contain rules such as:

- extend every major road straight ahead;
- create a junction when a road disconnects;
- place rail beside a major road;
- use a Singapore-specific road hierarchy.

Those relationships must be learned from data.

A later compiler may merge nearly coincident endpoints or enforce physical constraints. It must not invent missing urban structure.

## Z policy

The first context build does not pretend that categorical OSM vertical tags contain metric Z.

Transport targets currently supervise:

- surface / underground / elevated / unknown;
- horizontal vector geometry.

Exact transport Z is explicitly marked unsupervised.

The next dataset revision should add terrain/DEM context and learn terrain-relative vertical profiles where metric evidence exists. Unknown bridge/tunnel heights remain masked rather than being assigned fake training targets.

## Why this is different from the raster baseline

The raster baseline remains useful evidence for learned morphology and control.

Context-graph-v1 changes the future training source of truth:

- whole prepared city instead of unrelated image tiles;
- vector transport geometry instead of road-colour pixels;
- explicit cross-region continuation relationships;
- building polygons instead of connected grey components;
- masked local generation conditioned on city context.

## First gate

Before building a new neural model:

1. build the Singapore context graph;
2. inspect context graph and target previews;
3. measure how many region edges carry road and rail continuations;
4. inspect 512 m samples with many boundary ports;
5. confirm curved polylines and vertical-mode metadata survive intact;
6. only then design the graph/geometry diffusion objective.

The first learned model should be a small overfit test on a few masked regions, not a full overnight distribution run.
