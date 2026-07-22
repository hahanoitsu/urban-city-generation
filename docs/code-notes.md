# Code notes

These notes describe the parts of the project that are most likely to be changed during the research.

## Dataset code

The dataset package is in `src/urban_dataset`.

- `config.py` reads the single-area and city-preparation YAML files.
- `extract.py` reads an OSM PBF and separates roads, buildings, land use, water, green areas and rail.
- `classify.py` maps OSM tags to the classes used by the dataset.
- `enrich.py` estimates road widths and missing building heights.
- `project.py` repairs invalid geometry, projects it into metres and clips it to a boundary.
- `prepared.py` writes and reopens the prepared city GeoPackage.
- `corpus.py` reads the corpus YAML file and builds every configured study area.
- `tile.py` creates the fixed 1,024 m grid.
- `raster.py` converts vector features into the twelve model channels and the auxiliary masks.
- `pipeline.py` writes the tiles, metadata, indexes and previews for one study area.
- `audit.py` checks saved tensors and creates summary statistics.
- `manifests.py` creates the train, validation and test lists.

### Prepared city functions

`prepare_city(config)` extracts a city from its PBF, projects and enriches the layers, and writes one GeoPackage.

`save_city_gpkg(layers, path, metadata)` writes the seven vector layers and a small `urban_metadata` table.

`load_city_gpkg(path)` reopens those layers and checks that they use one coordinate system.

### Corpus functions

`load_corpus_config(path)` reads the output settings, tile settings, city files and coordinate boxes.

`build_corpus(config)` opens each GeoPackage once, checks that boxes do not overlap, clips the prepared layers to each box and builds all areas. It also creates an atlas and audit for every area, a combined audit and the three manifests.

`run_prepared_build(...)` is the part of `pipeline.py` used after a GeoPackage has already been prepared. The older `run_build(...)` path remains for small direct PBF experiments.

## Model code

The reconstruction model is in `src/urban_model`.

- `data.py` converts the twelve saved channels into the targets used during training.
- `model.py` contains the encoder, decoder and output heads.
- `losses.py` combines the road, land-use, binary-mask, height and centre-line losses.
- `metrics.py` calculates IoU and height error.
- `train.py` runs training, validation, checkpointing and preview generation.
- `evaluate.py` evaluates a saved checkpoint without changing it.
- `config.py` reads the model YAML file.

The encoder reduces a 256 × 256 tile to a 32 × 32 latent map. There are no encoder-to-decoder skip connections, so the reconstruction must pass through the latent representation.

## Current limitations

The first corpus still comes from one city. It is large enough for a more useful reconstruction experiment, but it cannot test transfer to a different city. Local roads, rail and individual building shapes also remain weaker reconstruction targets than broad land-use regions.
