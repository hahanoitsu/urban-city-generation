# Urban City Generation

This repository contains three early parts of a research project on learning urban layouts from real cities:

1. a dataset pipeline that converts OpenStreetMap data into fixed 1,024 m tiles;
2. a convolutional autoencoder that tests whether those tiles can be compressed and reconstructed;
3. a boundary-conditioned baseline that extends a real tile into a neighbouring area while preserving the seed.

The normal dataset workflow prepares one GeoPackage for each city and then builds several manually selected study areas from it. Raw map files, prepared cities, generated tiles and model checkpoints are kept outside Git.

The reconstruction and extension networks are baselines rather than the final city generator. They are used to test the representation, losses and road-boundary constraints before training a probabilistic latent model.

```bash
python -m urban_dataset --help
python -m urban_model --help
```

## Notes

- [Code notes](docs/code-notes.md)
- [Dataset method](docs/methodology/dataset-pipeline.md)
- [Reconstruction baseline](docs/methodology/reconstruction-baseline.md)
- [Seeded extension baseline](docs/methodology/seeded-extension.md)
- [Tile format](docs/reference/data-format.md)
- [Singapore corpus check](docs/results/singapore-corpus.md)
- [Roadmap](docs/roadmap.md)
