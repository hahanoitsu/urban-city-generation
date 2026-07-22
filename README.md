# Urban City Generation

This repository contains the dataset preparation part of a research project on urban layout generation.

The code converts OpenStreetMap data into fixed 1,024 m square samples. Each sample stores roads, land use, buildings, height, water, green areas and rail as raster channels, together with confidence masks used during training.

The current example uses a manually selected central Singapore rectangle. Choosing the study area explicitly is simpler and easier to explain than trying to detect urban areas across the whole island.

Raw map files and generated datasets are kept outside Git.

```bash
python -m urban_dataset --help
```

See [the dataset method](docs/methodology/dataset-pipeline.md) and [the current study area](docs/results/singapore-study-area.md).
