# Roadmap

## Data

- prepare one GeoPackage for each city;
- define several non-overlapping study boxes in one YAML file;
- add more types of Singapore neighbourhood;
- add cities with different road and block patterns;
- reserve complete cities for validation and testing;
- calculate tile-level morphology statistics.

## Reconstruction

- compare bottleneck sizes;
- adjust loss weights for local roads and rail;
- measure road connectivity after vectorisation;
- decide which latent representation to keep.

## Generation

- train a conditional model in the chosen latent space;
- convert predicted centre-lines into a road graph;
- repair disconnected roads and invalid blocks;
- export the repaired city plan to Unreal Engine PCG.
