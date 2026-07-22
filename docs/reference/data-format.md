# Data format

Each accepted tile contains:

```text
tiles/<tile-id>/
├── layers.npz
├── metadata.json
├── city.json
└── preview.png
```

## `layers.npz`

`layers` is a `float32[12,H,W]` array with values in `[0,1]`:

1. water
2. major road surface
3. secondary road surface
4. local road surface
5. residential land use
6. commercial or mixed land use
7. industrial land use
8. green or recreation land
9. building footprint
10. normalised building height
11. civic or institutional land use
12. rail or transit corridor

Auxiliary arrays:

- `height_confidence`: `uint8[H,W]`
- `landuse_known_mask`: `uint8[H,W]`
- `road_centerlines`: `uint8[3,H,W]`
- `valid_data_mask`: `uint8[H,W]`
- `affine`: projected raster transform

## `city.json`

Vector geometry is stored in local metres from the tile's south-west corner. Roads and buildings include the derived width and height values used during rasterisation.

## `metadata.json`

Metadata records source attribution, source snapshot, CRS, tile bounds, scale, channel order, and quality metrics.

`preview.png` is for inspection only and must not be used as model input.
