# Urban City Generation

This repository contains the first two parts of a research project on learning urban layouts from real cities:

1. a dataset pipeline that converts OpenStreetMap data into fixed 1,024 m tiles;
2. a convolutional autoencoder that tests whether those tiles can be compressed and reconstructed.

The reconstruction model is a baseline, not the final city generator. Its purpose is to check whether roads, land use, buildings and heights survive the chosen representation.

Raw map files, generated tiles and model checkpoints are kept outside Git.

```bash
python -m urban_dataset --help
python -m urban_model --help
```

The current example uses a manually selected central Singapore rectangle. Later datasets should use several non-overlapping study areas and reserve complete cities for validation and testing.
