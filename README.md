# Urban City Generation

This repository contains early research code for learning urban layouts from real map data:

1. a dataset pipeline that prepares reusable city GeoPackages and fixed-scale raster tiles;
2. reconstruction and deterministic extension baselines;
3. a CityGen-style semantic diffusion experiment for local block generation and masked outpainting.

The deterministic extension models are retained as negative baselines. The active generation path first learns complete categorical semantic blocks, then transfers that model to boundary-conditioned outpainting. Height synthesis, vector repair and Unreal Engine PCG are later stages.

Raw map files, prepared GeoPackages, generated tiles and model checkpoints are kept outside Git.

## Commands

```bash
python -m urban_dataset --help
python -m urban_model --help
```

The normal data workflow is:

```text
OSM PBF → prepared city GeoPackage → selected study areas → raster corpus
```

The current generation workflow is:

```text
semantic block diffusion → masked semantic outpainting → later refinement and height
```

## Notes

- [Code notes](docs/code-notes.md)
- [Dataset method](docs/methodology/dataset-pipeline.md)
- [Reconstruction baseline](docs/methodology/reconstruction-baseline.md)
- [Seeded extension baseline](docs/methodology/seeded-extension.md)
- [Semantic diffusion](docs/methodology/semantic-diffusion.md)
- [Tile format](docs/reference/data-format.md)
- [Roadmap](docs/roadmap.md)
