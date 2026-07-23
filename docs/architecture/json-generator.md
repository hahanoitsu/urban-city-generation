# JSON-first city generator

The generated city is a structured graph, not an image or mesh. Raster tiles remain useful for inspection and optional spatial conditioning, while Unreal Engine consumes the generated JSON.

## First model

The first model generates one complete fictional transport graph from a blank start token and a morphology vector. It is not trained to extend an existing map.

Each real `city.json` tile is converted into a compact graph program:

- `root` starts a connected road or rail component;
- `add` creates a node and an edge from an existing node;
- `connect` closes a loop between existing nodes.

Every edge carries its transport mode, class, width, vertical mode and layer. Coordinates are quantised only for model training and are converted back to metres when the program is decoded.

The sequence grammar prevents references to missing nodes, self-connections and missing edge attributes. Every node created by `add` is connected by construction. Geometry checks after decoding still handle crossings, clearances, duplicate edges and other constraints that are easier to enforce deterministically.

## Conditioning

The model receives a continuous morphology vector rather than an existing city fragment. Current fields include road and rail density, hierarchy fractions, vertical-mode fractions, intersection density, mean edge length, building coverage, height, water and green coverage.

Prepared corpora store one mean style vector per city. Generation may interpolate these profiles, for example Singapore and Amsterdam, without copying a source tile.

## Model

A decoder-only Transformer predicts graph-program tokens autoregressively. Sampling is grammar constrained. The canonical JSON remains the source of truth; the token stream is only a compact training representation.

The first output covers transport only. Blocks and parcels are derived from the generated surface network, followed by parcel-conditioned buildings and Unreal PCG compilation. A later global planner can provide terrain, water, land use and density fields for larger multi-region cities.
