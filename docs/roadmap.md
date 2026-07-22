# Roadmap

## Data

The reusable GeoPackage and multi-area corpus path are now in place. The next data work is to:

- calculate tile-level morphology statistics;
- add cities with different road and block patterns;
- reserve complete cities for validation and testing;
- check near-duplicate samples across different city sources.

## Reconstruction

- retrain the baseline on the larger Singapore corpus;
- compare bottleneck sizes;
- adjust loss weights for local roads and rail;
- measure road connectivity after vectorisation;
- decide which latent representation to keep.

## Generation

- train a conditional model in the chosen latent space;
- condition neighbouring tiles on road crossings at their shared edges;
- convert predicted centre-lines into a road graph;
- repair disconnected roads and invalid blocks;
- export the repaired city plan to Unreal Engine PCG.
