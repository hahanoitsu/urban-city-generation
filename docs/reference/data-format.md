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

`layers` is a `float32[12,H,W]` surface-occupancy array with values in `[0,1]`:

1. water;
2. major surface road;
3. secondary surface road;
4. local surface road;
5. residential land use;
6. commercial or mixed land use;
7. industrial land use;
8. green or recreation land;
9. building footprint;
10. normalised building height;
11. civic or institutional land use;
12. surface rail.

Underground and elevated transport are not inserted into these mutually exclusive surface targets.

Auxiliary arrays:

- `height_confidence`: `uint8[H,W]`;
- `height_known_mask`: `uint8[H,W]` compatibility view;
- `landuse_known_mask`: `uint8[H,W]`;
- `road_centerlines`: `uint8[3,H,W]` for surface major/secondary/local roads;
- `valid_data_mask`: `uint8[H,W]`;
- `road_vertical_masks`: `uint8[4,H,W]`;
- `rail_vertical_masks`: `uint8[4,H,W]`;
- `surface_transport_reservation`: `uint8[H,W]`;
- `buildable_surface_mask`: `uint8[H,W]`;
- `buildability_known_mask`: `uint8[H,W]`;
- `affine`: projected raster transform.

Vertical-mode order is:

```text
surface
underground
elevated
unknown
```

`surface_transport_reservation` combines confirmed surface road and rail right-of-way masks. `buildable_surface_mask` excludes water and confirmed surface transport. It is zero where `buildability_known_mask` is zero, so ambiguous vertical-mode features are not treated as confident buildability supervision.

The first version does not yet reserve station footprints, tunnel portals, ramps or elevated supports. Those interfaces are part of the next layered-city-state milestone.

## `city.json`

Vector geometry is stored in local metres from the tile's south-west corner. Roads and buildings include the derived width and height values used during rasterisation. Transport vectors retain available bridge, tunnel and layer attributes.

## `metadata.json`

Metadata records source attribution, source snapshot, CRS, tile bounds, scale, surface channel order, vertical-mode order, auxiliary-array shapes and quality metrics.

`preview.png` is for inspection only and must not be used as model input.
