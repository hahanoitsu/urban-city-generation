# Dataset method

The dataset represents each area as a 1,024 m square with a 256 × 256 raster. One pixel therefore covers four metres.

The normal conversion path is:

```text
OpenStreetMap PBF
→ tag extraction
→ metric projection
→ road and height enrichment
→ prepared city GeoPackage
→ manually selected study areas
→ fixed tile grid
→ raster channels and masks
→ audit and manifests
```

Preparing the city is separate from building the corpus. The PBF only has to be parsed again when the source data or the extraction rules change. Coordinate boxes can be edited and rebuilt from the GeoPackage without repeating extraction and enrichment.

Study areas use explicit longitude and latitude rectangles. The pipeline does not try to decide which parts of a whole country count as urban. It only rejects samples that are unusable, such as nearly empty or almost entirely water tiles.

Road widths use an explicit OSM width first, lane count second and a road-class default last. Parking aisles, private driveways and similar service roads are excluded from the morphology target.

Building heights use four confidence levels:

- 0: default based on building type;
- 1: median height from the same building type in the city;
- 2: derived from the number of floors;
- 3: explicit height in metres.

The prepared GeoPackage contains roads, buildings, classified land use, known land-use coverage, water, green areas and rail. Source details, the source checksum and the preparation settings are stored in an `urban_metadata` table inside the same file.

Study boxes from the same city are checked for overlap before any tiles are written. The tile grid remains globally aligned in the projected coordinate system, so the same location keeps the same tile identifier between builds.
