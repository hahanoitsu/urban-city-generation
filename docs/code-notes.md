# Code notes

These notes explain the parts of the project that are most likely to be changed during the research.

## Dataset code

The dataset package is in `src/urban_dataset`.

- `config.py` reads the YAML build files.
- `extract.py` reads an OSM PBF and separates roads, buildings, land use, water, green areas and rail.
- `classify.py` maps OSM tags to the classes used by the dataset.
- `enrich.py` estimates road widths and missing building heights.
- `project.py` repairs invalid geometry, projects it into metres and clips it to the selected area.
- `tile.py` creates the fixed 1,024 m grid.
- `raster.py` converts vector features into the twelve model channels and the auxiliary masks.
- `pipeline.py` runs one complete dataset build.
- `audit.py` checks saved tensors and creates summary statistics.
- `manifests.py` creates the train, validation and test lists.

A build currently starts from a PBF. The next data refactor should prepare one GeoPackage per city and then build several manually selected coordinate boxes from that file. This avoids repeating the expensive extraction step whenever a box changes.

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

The current corpus contains one central Singapore study area. It is useful for checking the software and the representation, but it is too small for a final model comparison. Local roads, rail and individual building shapes are also reconstructed less accurately than broad land-use regions.
