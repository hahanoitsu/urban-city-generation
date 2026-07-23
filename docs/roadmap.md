# Roadmap

## Data

- keep one prepared GeoPackage per city;
- add cities with different road and block patterns;
- reserve complete cities for validation and testing;
- build dense semantic crops from every split;
- calculate morphology statistics for conditioning and evaluation.

## Published layout baseline

- train the unconditional semantic block generator;
- verify sample diversity and class balance;
- initialise masked outpainting from the block checkpoint;
- compare plain CityGen-style outpainting with explicit road-boundary guides;
- add multi-scale semantic refinement.

## Project contribution

- preserve a georeferenced OSM seed exactly;
- measure road crossing, continuation and connectivity;
- propagate or predict road hierarchy;
- repair the generated road mask into a valid graph;
- compare generated morphology with real and held-out cities.

## 3D pipeline

- assign building heights from observed city distributions;
- convert semantic masks to roads, blocks, parcels and footprints;
- export explicit geometry and metadata;
- build the scene with Unreal Engine PCG;
- use UrbanWorld and CityX as references for asset and procedural-scene organisation.
