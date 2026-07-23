# City-state format

The layered dataset keeps raster tensors for model training, but `city.json` is the
vector source of truth for topology and 3D compilation.

Each tile state uses local metres from the south-west tile corner and records the
projected origin needed to place it back into city coordinates.

## Top-level fields

```text
format: urban-city-state-tile
version: 0.1.0
tile
coordinate_system
vertical_defaults
transport_graph
building_solids
roads / rail / buildings / landuse / water / green
```

The original feature collections remain present for inspection and backward-compatible
consumers. New topology-aware and Unreal code should use `transport_graph` and
`building_solids`.

## Transport graph

A node records:

- stable tile-local `id`;
- transport mode (`road` or `rail`);
- vertical mode (`surface`, `underground`, `elevated`, or `unknown`);
- local and projected positions;
- local OSM layer order when present;
- degree and node type;
- a `boundary_port_key` when the node lies on a tile boundary.

Boundary port keys are based on projected position, mode and vertical state. Adjacent
tiles can therefore reconcile the same continuation without relying on image pixels.

An edge records:

- from/to node ids;
- mode and hierarchy/class;
- vertical mode and OSM layer order;
- width and whether it came from a road estimate or a rail-mode default;
- local XYZ polyline;
- source OSM id/type where available;
- bridge, tunnel and one-way attributes;
- whether the vertical state needs review.

Road and rail graphs are noded independently. Surface, underground and elevated groups
are also noded independently, so a bridge, tunnel or viaduct does not become connected
to a crossing surface feature. Features with ambiguous vertical state are not connected
to other unknown features merely because their 2D lines cross.

## Vertical values

The initial compiler uses these procedural Z defaults:

```text
surface:       0 m
underground: -12 m
elevated:     +8 m
unknown:     null
```

These values make the state immediately importable into a 3D prototype. They are not
claimed as measured tunnel depths or deck elevations. Future terrain, station, portal,
grade and clearance stages will replace them with solved profiles.

## Building solids

A building solid records:

- stable id and source OSM identity;
- local footprint geometry;
- surface base elevation;
- estimated height;
- height source and confidence.

At this stage a building is a vertical extrusion of its footprint. Roof shape, podiums,
setbacks, entrances and facade/asset rules are later PCG stages.

## Multi-city bundle

Run:

```bash
python -m urban_dataset compile-city-state \
  --dataset data/processed/world-corpus \
  --output data/processed/world-corpus/city-state-index.json
```

The bundle indexes every tile state, grouped by city and area. A future Unreal importer
can map each tile to a World Partition cell while resolving matching boundary ports.
The bundle does not duplicate geometry; it references each tile's `city.json`.
