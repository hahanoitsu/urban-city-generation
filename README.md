# Urban City Generation

A research pipeline for generating completely fictional urban layouts from real OpenStreetMap cities. The active model is a purpose-built 2D diffusion model. It generates surface structure and continuous underground/elevated transport profiles together, then converts them into structured JSON for Unreal Engine PCG.

Current work is on `dev`. `main` remains stable.

## Architecture

```text
OpenStreetMap cities
  -> tagged layered raster and vector corpus
  -> 19-channel 2D diffusion model
  -> novel surface layout + transport height/depth fields
  -> grade-limited raster-to-spline conversion
  -> generated city.json
  -> Unreal Engine PCG
```

The neural model chooses the city layout and its vertical profiles. The deterministic conversion stage traces masks into splines, smooths noisy Z predictions and enforces configured road/rail grade limits.

The earlier autoregressive graph Transformer is retained as a failed research baseline. It generated valid command sequences but did not learn usable geometry from the small corpus.

## Repository layout

```text
configs/            city, corpus and model configuration
src/urban_dataset/  OSM extraction, profile targets and deterministic geometry
src/urban_model/    active multilayer diffusion model
src/urban_ai/       earlier graph-program experiment
tests/              automated checks
docs/               format and architecture notes
data/               local source and processed data, ignored by Git
runs/               checkpoints and generated cities, ignored by Git
```

## Setup

```bash
git clone https://github.com/hahanoitsu/urban-city-generation.git
cd urban-city-generation
git switch dev

conda env create -f environment.yml
conda activate urban-city
python -m pip install -e '.[diffusion]'
```

The geometry pipeline can still be installed without PyTorch:

```bash
python -m pip install -e .
```

## Rebuild the prepared city after this schema change

The older `singapore.gpkg` discarded several OSM fields that are now used for vertical profiles. Recreate it from the source PBF before rebuilding the corpus:

```bash
python -m urban_dataset prepare-city --config configs/cities/singapore.yaml
python -m urban_dataset build-corpus --config configs/corpus.yaml
```

The extractor now retains transport evidence including:

```text
bridge / bridge:structure / bridge:movable
tunnel / location / layer / level
incline / ele / height / min_height / depth
embankment / cutting
maxheight / maxheight:physical
```

OSM does not provide complete engineering-grade elevation profiles. Explicit metric tags receive stronger confidence; bridge/tunnel/layer evidence is converted into smooth inferred ramps with lower confidence. A terrain DEM can later replace the zero-ground reference without changing the model schema.

Each accepted tile stores:

- the original twelve surface channels;
- independent surface/underground/elevated/unknown road and rail masks;
- signed road and rail Z-profile rasters for surface, underground and elevated modes;
- confidence rasters describing whether each profile is missing, inferred, tag-derived or measured;
- an authoritative real `city.json` used for data checking and Unreal-format development.

## Model output

The diffusion model generates nineteen channels at once.

Surface is one categorical map represented by eight channels:

```text
terrain
vegetation
building
major road
secondary road
local road
surface rail
water
```

Five overlapping channels contain:

```text
underground roads
elevated roads
underground rail
elevated rail
building height
```

Six continuous channels contain:

```text
surface-road offset
underground-road depth
elevated-road height
surface-rail offset
underground-rail depth
elevated-rail height
```

Because these are independent layers, an underground railway may pass beneath a building or surface road without either feature being erased. Bridges and tunnels can ramp continuously instead of occupying one fixed Z plane.

## Check the training tensors

```bash
python -m urban_model check --config configs/layered.yaml
```

This verifies manifests, crop counts, `[19, 128, 128]` tensors, channel coverage and the supervised fraction of each confidence-weighted profile channel.

## Train

Run a one-epoch CUDA smoke test first:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m urban_model train \
  --config configs/layered.yaml \
  --epochs 1 \
  --overwrite
```

The new model writes to:

```text
runs/layered-v2/
```

A longer run can then start cleanly:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m urban_model train \
  --config configs/layered.yaml \
  --epochs 100 \
  --overwrite
```

The training loop uses BF16, a cosine diffusion schedule, per-channel confidence masks, reduced vertical oversampling and an EMA with warm-up. Early stopping uses validation loss.

## Generate new cities

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m urban_model sample \
  --config configs/layered.yaml \
  --checkpoint runs/layered-v2/best.pt \
  --output runs/layered-v2-samples \
  --count 8 \
  --overwrite
```

Every sample contains:

```text
layers.npz    generated 19-channel tensor
preview.png   surface, underground-depth and elevated-height panels
city.json     variable-Z roads, rail, buildings, water and green space
city.obj      simple structural 3D preview
city.mtl      OBJ materials
```

The vectoriser skeletonises generated transport masks, samples the learned height/depth field along each path, smooths it and clamps gradients to configured limits. JSON edge geometry therefore contains full `[x, y, z]` points plus minimum/maximum Z and maximum grade.

## Unreal Engine path

Unreal consumes the generated JSON rather than reading the preview image. The importer can map:

```text
geometry_local_m -> 3D road and rail spline points
width_m          -> spline cross-section width
vertical_mode    -> surface, tunnel, elevated or transition system
maximum_grade    -> validation/debug information
building solids  -> PCG building volumes
water and green  -> landscape and vegetation masks
```

The OBJ exporter is only a structural inspection tool. Unreal Engine PCG remains responsible for final roads, intersections, buildings, terrain, vegetation, traffic and rendering.

## Deterministic source-scene compiler

The existing compiler remains useful for checking real source data:

```bash
python -m urban_dataset build-scene \
  --dataset data/processed/corpus \
  --city singapore \
  --area central \
  --max-tiles 8 \
  --output runs/central-scene-small
```

It is a validation utility, not the generative model.
