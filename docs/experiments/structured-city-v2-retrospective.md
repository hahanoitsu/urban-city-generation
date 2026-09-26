# Structured city v2 retrospective

## Result

The joint structured denoiser should not be scaled further.

The epoch-15 v1 run collapsed variable-cardinality slots and semantic classes. The v2 run improved counts, but the generated geometry remained invalid and disconnected.

Inspection of the v2 preview export found:

- 783 generated transport nodes, 738 edges, 743 buildings and 149 areas across 12 held-out samples.
- 11.4% of generated node coordinates were outside the 512 m target.
- 8.2% of generated spline points were outside the target.
- 99.7% of generated building footprint vertices were outside the target.
- 86.0% of generated area vertices were outside the target.
- every generated building was classified as generic.
- generated transport graphs had many disconnected components and isolated nodes.

The count problem was not the main remaining failure. The representation and learning objective were.

## Main problems

### Direct data-space denoising

The model denoises hundreds of raw coordinates, polygon vertices, attributes and topology pointers directly. High-noise validation remains much worse than medium-noise reconstruction. Sampling from pure noise therefore asks the model to solve a much harder problem than the training metrics imply.

### Permutation ambiguity

Buildings, areas and graph elements are sets. Canonical sorting helps but does not remove the ambiguity of regressing exact object identities after heavy noising. Direct coordinate losses encourage averaged geometry when context admits multiple valid layouts.

### Artificial rectangular target

The 512 m window is useful for batching but is not a natural urban unit. Roads define blocks and blocks constrain buildings. Jointly generating transport, buildings and ground polygons inside one rectangular slot tensor ignores this hierarchy.

### Weak spatial conditioning

The context graph provides useful regional relationships, but the immediate surroundings are mostly aggregate features plus boundary ports. It does not provide detailed neighboring vector geometry. The model is therefore trained against one exact target even though the conditioning does not determine one exact target.

### Mixed representation

Road graphs, building objects and ground surfaces have different structure. Treating all of them as similar slot sets is unnecessarily difficult.

## Literature direction

COHO represents the whole city as a hierarchy of city blocks and neighboring relations, then generates block-level building-layout latents with masked graph modeling.

Scenario Dreamer learns an autoencoder over vectorized lane graphs and objects, freezes it, then performs diffusion in latent space. It explicitly discusses permutation ambiguity in structured sets.

MapDreamer also uses vectorized graph autoencoding plus latent diffusion. Its ablation reports that fixed-size queries with an existence head are brittle for variable map cardinality. It uses a discrete cardinality model plus ghost latents and preserves city-scale continuity through boundary conditioning.

BlockPlanner models a city block as a structured graph rather than a rectangular canvas and adds geometric validity losses.

PrITTI uses a hybrid urban representation: vectorized object primitives and rasterized ground surfaces instead of forcing every scene component into one representation.

## Revised architecture

### Stage A: strategic city context

Keep the city-scale context graph. It should represent coarse transport demand, land-use character, density, terrain and inter-region continuation.

For fictional generation, this graph must eventually be generated rather than copied from Singapore.

### Stage B: transport latent model

Train a transport graph autoencoder first.

Inputs and outputs:

- road and rail polylines
- explicit connectivity
- hierarchy and mode
- vertical relation
- boundary continuation

The autoencoder must demonstrate near-lossless reconstruction and valid connectivity before generative training.

Then train a latent diffusion or masked generative model over the transport latents, conditioned on regional context and detailed neighboring boundary geometry.

### Stage C: natural blocks

Polygonize the generated road network into city blocks and parcels. These become the natural units for building generation.

### Stage D: building-layout latent model

Train a block-level layout autoencoder or VAE. Generate building layouts conditioned on:

- block boundary
- road frontage
- land use
- neighboring block embeddings
- density and height controls

Do not diffuse absolute footprint vertices for hundreds of buildings at once.

### Stage E: ground and environment

Use a hybrid representation for continuous ground semantics. Water, green space and broad land-use regions should not share the same object-slot mechanism as buildings.

### Stage F: metric 3D

Add terrain-relative Z only when real elevation supervision is available. Vertical transport mode and ordering can be learned before metric Z.

## Scale

Do not make a 5 km by 5 km full-detail target.

Use several kilometres of context around a smaller detailed generation region. The current 6 km coarse context idea is reasonable, but the local generator should receive real neighboring vector geometry rather than only aggregate summaries.

A 512 m detailed transport target is acceptable for the first transport model. A later 1024 m target can be tested after the latent representation is stable.

## Next gate

The next experiment is not another full city denoiser training run.

The next gate is a transport graph autoencoder on the Singapore corpus.

It passes only if held-out reconstruction preserves:

- connected components
- boundary ports
- junction degrees
- road and rail mode
- topology
- spline geometry

Only after that passes should a generative prior be trained in the latent space.
