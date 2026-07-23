# Urban City Generation

This repository contains early research code for learning urban layouts from real map data:

1. a dataset pipeline that prepares reusable city GeoPackages and fixed-scale raster tiles;
2. reconstruction and deterministic extension baselines;
3. a CityGen-style flat semantic diffusion baseline;
4. an in-progress layered city-state representation for topology-aware generation and Unreal Engine PCG.

The deterministic and flat semantic models are retained as research baselines. The current design direction separates surface, underground and elevated infrastructure, treats transport as a graph, derives blocks and parcels from surface rights-of-way, and constrains buildings to valid buildable envelopes.

Raw map files, prepared GeoPackages, generated tiles and model checkpoints are kept outside Git.

## Commands

```bash
python -m urban_dataset --help
python -m urban_model --help
```

The normal data workflow is:

```text
OSM PBF → prepared city GeoPackage → selected study areas → layered raster/vector corpus
```

The target generation workflow is:

```text
strategic plan
→ layered transport graph
→ surface reservations
→ blocks and parcels
→ buildings and height
→ stateful expansion
→ Unreal Engine PCG
```

## Notes

- [Code notes](docs/code-notes.md)
- [Dataset method](docs/methodology/dataset-pipeline.md)
- [Layered city-state architecture](docs/architecture/layered-city-state.md)
- [Research paper plan](docs/research/paper-plan.md)
- [Reconstruction baseline](docs/methodology/reconstruction-baseline.md)
- [Seeded extension baseline](docs/methodology/seeded-extension.md)
- [Semantic diffusion baseline](docs/methodology/semantic-diffusion.md)
- [Tile format](docs/reference/data-format.md)
- [Roadmap](docs/roadmap.md)
