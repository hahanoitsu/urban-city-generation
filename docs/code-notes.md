# Code notes

These notes cover the parts of the project that are most likely to change during the research.

## Dataset package

The dataset package is in `src/urban_dataset`.

- `config.py` reads dataset YAML files.
- `extract.py` reads OSM PBF data.
- `classify.py` maps OSM tags to the project classes.
- `enrich.py` estimates road widths and missing building heights.
- `project.py` repairs and projects geometry.
- `tile.py` creates the fixed metric grid.
- `raster.py` writes the twelve channels and auxiliary masks.
- `pipeline.py` runs single-area builds.
- `prepared.py` and `corpus.py` handle reusable GeoPackages and multiple study areas.
- `audit.py` checks saved tensors.
- `manifests.py` creates train, validation and test lists.

## Baseline models

The older model code is in `src/urban_model`.

- `model.py` contains the reconstruction autoencoder.
- `data.py` creates reconstruction and adjacent-tile extension samples.
- `losses.py` and `metrics.py` implement the baseline objectives.
- `train.py`, `evaluate.py` and `extend.py` run the deterministic experiments.

These models remain useful as comparison points, but they are no longer the active generation design.

## Semantic diffusion

`semantic_diffusion.py` is the public import surface for the active paper-based experiment. The implementation is split into `semantic_config.py`, `semantic_data.py`, `semantic_model.py` and `semantic_train.py`.

Together these modules:

- converts twelve raster channels to one categorical semantic field;
- creates dense crops for block generation;
- creates adjacent masked pairs for outpainting;
- builds a Hugging Face Diffusers U-Net and schedulers;
- trains with DDPM epsilon prediction;
- maintains exponential moving average weights;
- transfers a block checkpoint into the wider outpainting input layer;
- preserves the known semantic seed throughout denoising;
- samples complete blocks or OSM-seeded extensions.

The command-line entry points are kept in `urban_model.__main__` so experiments can be run with direct Python commands rather than wrapper scripts.
