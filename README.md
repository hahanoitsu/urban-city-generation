# Urban City Generation

A JSON-first generative city project built from OpenStreetMap data. The intended output is a completely fictional city state containing connected transport graphs, vertical levels, blocks, parcels and buildings. Preview images and OBJ files are derived from that JSON; Unreal Engine PCG is the final renderer and asset assembler.

Current work is on `dev`. `main` remains stable.

## Layout

```text
configs/            city, corpus and generator configuration
src/urban_dataset/  OSM preparation and deterministic geometry
src/urban_ai/       graph-program dataset, model and sampling
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
python -m pip install -e .
pytest -q
```

The geometry pipeline does not require PyTorch. On a training machine with PyTorch installed:

```bash
python -m pip install -e '.[ml]'
```

## Build the city corpus

Place the prepared Singapore GeoPackage at:

```text
data/cities/singapore.gpkg
```

Then run:

```bash
python -m urban_dataset build-corpus --config configs/corpus.yaml
```

Each accepted tile contains raster inspection arrays and an authoritative vector `city.json`. The JSON records road and rail connectivity, hierarchy, width, surface/underground/elevated mode, buildings, land use, water and green space.

## Prepare the AI dataset

The model does not learn from coloured road pixels or raw JSON text. Each `city.json` graph is converted into a compact sequence of structured graph commands.

```bash
python -m urban_ai prepare --config configs/generator.yaml --overwrite
python -m urban_ai check --config configs/generator.yaml
```

The commands are:

- `root`: start a transport component;
- `add`: create a node connected to an existing node;
- `connect`: close a loop between existing nodes.

Edge commands include transport type, road or rail class, width and vertical level. This makes connectivity part of the representation rather than a property inferred from touching pixels.

## Check the representation before training

Convert one real tile to the graph program and back:

```bash
STATE=$(find data/processed/corpus -name city.json | head -n 1)
python -m urban_ai roundtrip --state "$STATE" --output runs/roundtrip
```

This writes `program.json`, reconstructed `city.json`, `preview.png` and `city.obj`.

## Train the complete-city graph model

A one-epoch CUDA smoke test:

```bash
python -m urban_ai train \
  --config configs/generator.yaml \
  --epochs 1 \
  --overwrite
```

The first model is a decoder-only Transformer. It starts from a blank site and a morphology vector, then generates a complete fictional transport graph command by command. It is not given a real tile to continue.

## Generate fictional cities

```bash
python -m urban_ai sample \
  --config configs/generator.yaml \
  --checkpoint runs/generator/best.pt \
  --output runs/generated \
  --count 4 \
  --mix singapore=1.0 \
  --overwrite
```

Each sample contains:

```text
program.json   model command sequence
city.json      generated structured city state
preview.png    top-down inspection image
city.obj       simple structural 3D preview
city.mtl       OBJ materials
```

When more cities are added, style vectors can be mixed without using any source map as a seed:

```bash
--mix singapore=0.6 --mix amsterdam=0.25 --mix kuala-lumpur=0.15
```

Individual morphology values can also be overridden, for example:

```bash
--set building_coverage=0.35 --set elevated_fraction=0.08
```

## Deterministic scene compiler

The existing compiler can still stitch and inspect real source tiles:

```bash
python -m urban_dataset build-scene \
  --dataset data/processed/corpus \
  --city singapore \
  --area central \
  --max-tiles 8 \
  --output runs/central-scene-small
```

It is used to validate the data representation and geometry code. It is not the generative model.

## Architecture

The current path is:

```text
OSM cities
  -> layered city JSON
  -> graph-program corpus
  -> fictional transport graph Transformer
  -> deterministic topology and geometry checks
  -> blocks and parcels
  -> buildings
  -> generated city JSON
  -> Unreal Engine PCG
```

The first model works at one neighbourhood scale. A later global planner will generate terrain, water, districts, density, major roads and rail for a larger city, while the local graph model fills coordinated regions. See `docs/architecture/json-generator.md`.
