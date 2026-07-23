# Layered city-state architecture

## Purpose

The final generator should not treat a city as one mutually exclusive colour image. Real urban systems contain surface, underground and elevated infrastructure that can overlap horizontally, while roads, blocks, parcels and buildings have explicit geometric relationships.

The authoritative representation is therefore a layered vector city state. Raster tensors are derived model views used for learning and evaluation.

## Design principles

1. Surface occupancy is separate from underground and elevated infrastructure.
2. Geometric crossings do not automatically create graph intersections.
3. Surface transport reserves land before blocks, parcels and buildings are generated.
4. Underground transport may overlap surface buildings except at stations, portals, ramps, vents and other interfaces.
5. Elevated transport uses clearance and support reservations rather than deleting all surface occupancy below it.
6. Unknown or ambiguous source data masks supervision instead of becoming a confident target.
7. Learned models propose morphology; graph and polygon constraints own validity.
8. Repeated expansion updates persistent city state rather than completing isolated images.

## Current dataset v0.3 foundation

The twelve primary channels now describe surface occupancy:

1. water;
2. major surface road;
3. secondary surface road;
4. local surface road;
5. residential land use;
6. commercial/mixed land use;
7. industrial land use;
8. green;
9. building footprint;
10. normalised building height;
11. civic land use;
12. surface rail.

Additional arrays are stored in `layers.npz`:

- `road_vertical_masks`: `uint8[4,H,W]`;
- `rail_vertical_masks`: `uint8[4,H,W]`;
- `surface_transport_reservation`: `uint8[H,W]`;
- `buildable_surface_mask`: `uint8[H,W]`;
- `buildability_known_mask`: `uint8[H,W]`.

Vertical-mode order is:

```text
surface
underground
elevated
unknown
```

The first implementation uses explicit `tunnel`, `bridge` and `location` evidence when available. A non-zero OSM `layer` without an explicit structural tag is kept as `unknown`: `layer` is a local stacking relation, not metric elevation or proof that a feature is underground.

The compatibility semantic and reconstruction models may still read the twelve primary channels, but they now see only confirmed surface transport. Later models should use the layered auxiliaries directly.

## Canonical vector state

### Site and environment

```text
site boundary
valid/known masks
terrain elevation and slope
water and coastline
protected/open land
development suitability
```

### Transport graph

Nodes contain:

```text
id
position
node type
vertical mode
local layer
source confidence
```

Edges contain:

```text
id
from/to node
polyline geometry
mode: road or rail
class/hierarchy
vertical mode
local layer
width or right-of-way
lanes or tracks
bridge/tunnel tags
one-way state
source confidence
```

Road and rail edges at different vertical modes may cross without connecting.

### Surface reservations

Derived polygons include:

```text
road right-of-way
surface rail right-of-way
station footprint
portal/ramp footprint
elevated support footprint
clearance envelope
```

### Blocks, parcels and buildings

Surface reservations are subtracted from developable land to form blocks. Blocks are divided into parcels with street access. Building footprints are generated inside parcel buildable envelopes and cannot overlap confirmed surface reservations.

Underground alignments do not remove ordinary buildable surface. Elevated alignments reserve only the required supports, stations and clearance envelope.

## Generation stages

### 1. Strategic plan

Generate multi-kilometre development intensity, arterial roads, strategic rail, stations, major green/water corridors and district structure.

### 2. Local surface transport

Conditioned on the strategic plan, generate secondary/local roads and surface rail. The first practical baseline will predict signed distance fields, convert them to centre-lines and repair the resulting vector graph.

### 3. Blocks and parcels

Polygonise the validated surface network into blocks. Begin with deterministic parcel splitting and optimisation because cadastral ground truth is inconsistent in OSM.

### 4. Land use and density

Assign block/parcel use and density from accessibility, station proximity, environmental context and city-style controls.

### 5. Buildings

Generate parcel-conditioned footprints, then height and typology. Hard geometry checks enforce containment, access, setbacks and non-overlap.

### 6. Vertical infrastructure

Compile underground tunnels, elevated viaducts, stations, portals, ramps and bridges using categorical vertical mode and procedural clearance/depth rules. Precise depth is not inferred from OSM `layer`.

### 7. Unreal Engine PCG

Export deterministic splines, graph IDs, blocks, parcels, footprints, heights, vertical modes, cross-section rules and asset seeds. Unreal assembles geometry; it does not repair urban topology.

## Stateful expansion

An expandable boundary stores graph ports rather than only edge pixels. Each port includes position, heading, hierarchy and vertical mode. The generator also receives strategic-plan fields, protected regions, water/green continuation and accumulated graph state.

New regions overlap the committed city. Existing geometry remains fixed, proposed graph edges are reconciled in the overlap, and the proposal is committed only after topology, vertical, block and parcel checks pass.

## Failure gates

### Dataset gate

- underground rail remains visible below buildings without replacing them;
- elevated rail can overlap a surface road;
- ambiguous `layer` values enter the unknown mask;
- grade-separated crossings do not become graph junctions;
- per-mode distributions and source confidence are audited.

### Transport gate

Using real upstream context, a tiny model must memorise a few examples and produce topology close to real crops. Failure to memorise or hundreds of disconnected components stops the experiment.

### Block/parcel gate

Using real transport, derived blocks and parcels must be valid, non-sliver polygons with public-street access.

### Building gate

Using real parcels, footprints must remain within buildable envelopes with low repair and rejection rates.

### Composition gate

Generated upstream stages replace real inputs one at a time so error propagation is measured rather than hidden.

### Expansion gate

Repeated extension must preserve the seed, satisfy boundary graph ports, maintain global connectivity and avoid morphology drift.

## Research baselines

The project should preserve and compare:

1. deterministic adjacent-tile extension;
2. skip-connected deterministic extension;
3. flat CityGen-style semantic diffusion;
4. layered transport distance fields plus graph repair;
5. the complete hierarchical city-state system.

The flat semantic diffusion model remains a useful negative/partial baseline: it learned broad morphology and stochastic variation, but its road and rail outputs were topologically fragmented despite low diffusion loss.

## Immediate implementation sequence

1. finish vertical-mode extraction and layered dataset audits;
2. rebuild Singapore and inspect surface/underground/elevated atlases;
3. construct an attributed transport graph with grade separation;
4. derive surface reservations and buildable polygons;
5. add held-out cities and geographic evaluation splits;
6. build a deterministic graph-to-block-to-parcel-to-building baseline;
7. train the first transport distance-field model;
8. add persistent state and graph-port outpainting only after local stages pass their gates.
