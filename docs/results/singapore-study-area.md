# Singapore study area

The first reconstruction dataset uses this central Singapore rectangle:

```text
WGS84: [103.7900, 1.2900, 103.8700, 1.3700]
Tile size: 1,024 m
Raster size: 256 × 256
```

Using the July 2026 Singapore OSM extract, the rectangle produced 56 complete tiles. All 56 passed the minimal validity checks.

An earlier whole-island experiment was discarded because missing GDAL tag values were read as the string `nan`. This incorrectly selected many polygons as buildings, water and mapped land use at the same time. The tag normalisation now treats missing values correctly.

A 20-epoch run on the corrected study area reduced validation loss from 3.7116 to 1.5014. Broad land-use regions and major roads reconstructed reasonably well, while local roads, rail and detailed building footprints remained weaker.
