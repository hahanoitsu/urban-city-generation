# Dataset method

The dataset represents each area as a 1,024 m square with a 256 × 256 raster. One pixel therefore covers four metres.

The current conversion path is:

```text
OpenStreetMap PBF
→ tag extraction
→ metric projection
→ road and height enrichment
→ fixed tile grid
→ raster channels and masks
→ audit and manifests
```

Study areas are chosen using explicit longitude and latitude rectangles. The pipeline does not try to decide which parts of a whole country are urban. The rejection checks only remove unusable samples such as empty tiles or tiles that are almost entirely water.

Road widths use an explicit OSM width first, lane count second and a road-class default last. Parking aisles, private driveways and similar service roads are excluded from the morphology target.

Building heights use four confidence levels:

- 0: default based on building type;
- 1: median height from the same building type in the city;
- 2: derived from the number of floors;
- 3: explicit height in metres.

The next version should store one prepared GeoPackage per city. Coordinate boxes can then be changed without parsing the source PBF again.
