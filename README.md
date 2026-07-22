# Urban City Generation

This repository contains two early parts of a research project on learning urban layouts from real cities:

1. a dataset pipeline that converts OpenStreetMap data into fixed 1,024 m tiles;
2. a convolutional autoencoder that tests whether those tiles can be compressed and reconstructed.

The normal dataset workflow prepares one GeoPackage for each city and then builds several manually selected study areas from it. Raw map files, prepared cities, generated tiles and model checkpoints are kept outside Git.

The reconstruction model is a baseline rather than the final city generator. Its purpose is to test whether roads, land use, buildings and height patterns survive the chosen representation.

```bash
python -m urban_dataset --help
python -m urban_model --help
```

## Notes

- [Code notes](docs/code-notes.md)
- [Dataset method](docs/methodology/dataset-pipeline.md)
- [Reconstruction baseline](docs/methodology/reconstruction-baseline.md)
- [Tile format](docs/reference/data-format.md)
- [Singapore corpus check](docs/results/singapore-corpus.md)
- [Roadmap](docs/roadmap.md)
