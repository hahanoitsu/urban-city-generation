# Urban City Generation

This repository contains the dataset preparation part of a research project on urban layout generation.

The code converts OpenStreetMap data into fixed 1,024 m square samples. Each sample stores roads, land use, buildings, height, water, green areas and rail as raster channels, together with a small set of confidence masks.

Raw map files and generated datasets are kept outside Git.

```bash
python -m urban_dataset --help
```
