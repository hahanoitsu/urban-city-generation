# Layered city model

This replaces the one-big-denoiser approach (structured city, grid city). The code is in `src/urban_layers`.

## What it has to do

1. train on several cities and know which ones it was trained on
2. learn from context: roads continue from the neighbours, a tile knows what is around it
3. make a full realistic city from scratch
4. be steerable: sliders for roads, buildings, height, green, water, land use, and a % mix of city styles
5. export something an Unreal importer can turn into a 3D city with PCG

## Why layers

Real cities are built in layers. Districts decide density and character, the road network follows, blocks fall out of the roads, and buildings fill the blocks. Each layer has a different shape and scale, so each gets its own representation and its own model. Every layer is conditioned on the layers above it, and every layer's output can be checked on its own.

The old joint denoiser asked one network to place hundreds of roads, buildings and polygons at once from noise. That failed because of slot identity, invalid polygons and edges that did not know where they were. Here each of those problems is avoided by construction:

| problem in the old model | how the layered model avoids it |
|---|---|
| slots with no fixed meaning | roads and buildings are sequences in a fixed order, so there are no slots |
| star shaped polygons | buildings are oriented boxes, which are always valid |
| edges pointing at the wrong node | roads are polylines and junctions come from where they meet, using the repo's own graph builder |
| roads breaking at tile edges | each tile sees the lines already generated next to it and the exact points (ports) where they enter |
| no control | a coarse plan with city style and slider inputs drives everything below it |

## Layers

```text
city style weights + sliders + seed
              |
              v
  1. plan          diffusion U-Net on a 128 m grid
              |    road density by class, rail, building coverage, height,
              |    green, water, land use
              v
  2. water, green  polygons traced from the plan
              |
              v
  3. streets       transformer, one 512 m tile at a time, as polylines
              |    sees: plan around the tile, neighbour lines, ports
              v
  4. network       snap dead ends, node junctions, graph (no learning)
              |
              v
  5. blocks        polygons between surface roads and rail (no learning)
              |
              v
  6. buildings     transformer, one block at a time, as oriented boxes
              |    sees: block shape, plan values, road frontage
              v
     city.json + city.obj preview + city.png
```

### 1. Plan

`plan.py`. The training data is the real city measured on a 128 m grid (`data.py: plan_raster`). There are 12 channels: major, secondary and local road density, rail density, building coverage, mean height, green, water, and residential, commercial, industrial and civic land use.

The model is a small diffusion U-Net. It is conditioned on:

- a city style vector, one weight per training city. One-hot means "make it like this city", and mixed weights mean a style mix.
- a control vector, the mean of every channel over the crop. This is what the sliders move. "1.3 x roads" asks for a plan whose road channels average 1.3 times the city's normal value.

Both conditions are dropped 15% of the time during training, so classifier free guidance works at generation. The network is fully convolutional, so it trains on 8 km crops and can still generate any size of city. It can also inpaint, keeping some cells fixed, which is how an existing area or a hand drawn outline can be locked.

A city's bounding box has a lot of nothing in it (for Singapore, mostly sea). Only cells within 256 m of a road or rail line count as urban (`urban_cells`). Training crops must be at least half urban, and each city's default slider values are the means over its urban cells only, so "singapore=1" means the built up part of Singapore and not the sea around it.

After sampling, every channel is rescaled so its mean matches what the sliders asked for (`match_means`). The model decides where things go and the sliders decide how much. Without this the sliders only work if the training data happens to vary enough, which the synthetic test cities did not.

This is the layer that makes "from scratch" possible. Before, the context always came from real Singapore. Now the context itself is generated.

### 2. Water and green

Thresholded from the plan channels and traced into polygons (`city.py: plan_polygons`).

### 3. Streets

`streets.py`. One 512 m tile at a time. Each tile is a token sequence: for every line, its class (road class or rail type, times surface, elevated, underground or unknown), then its points as x, y tokens on a 1 m grid, then an end token. A grammar mask during sampling makes every output well formed.

The tile sees:

- the plan cells around it (1 km window)
- every line already generated in the 128 m ring around it, as context
- the ports, which are the exact points where those lines cross into this tile, with their class and heading

During training the visible neighbours are random: none, all of them, or the patterns that happen when generating left to right and bottom to top. All 8 rotations and flips are used. So the same model can start a city, continue one or fill a hole.

Tiles are generated in diagonal batches, so every tile can see its left and lower neighbours.

Ports are forced. They are sorted by side and position, training puts the lines that continue a port first in that order, and at generation the grammar makes each of those lines start with the port's class and exact point. The model only decides where the line goes next. A port that an earlier line ends on counts as used. The grammar also keeps enough of the token budget for the ports still waiting, ending a long line early if it has to, so one long line cannot use up the space the other ports need. Before this, an undertrained model continued only 14% of ports, and the tile seams broke. With forcing it is 100% by construction.

The other way a seam breaks is a new line that ends on the edge of a tile that is already done, where nothing continues it. That never happens in the training data, because every line end on a visible side is a port. So the grammar does not let a free line start or end within 4 m of a done side unless it is at a port. The line has to keep going instead.

Rail and elevated roads go through the same model. Their class token carries the vertical mode, and the plan has a rail channel. Rail is rare in the data, so it is the weakest part and the first thing to check in the audit.

### 4 and 5. Network and blocks

No learning, just geometry. Dead ends within 6 m of another line of the same mode are snapped onto it, and the vertex is added to both lines so the junction is exact. Then `urban_dataset.city_state.build_transport_graph` nodes everything into the usual graph.

Blocks are the polygons between surface roads and surface rail, shrunk by 3 m, minus water. Blocks over 384 m are split. This step does not invent structure. It reads off what the roads already decided.

### 6. Buildings

`buildings.py`. One block at a time, in the block's own frame, where the x axis is the block's long side. The sequence is the block's land use, then for every building: kind, x, y, width, depth, angle, height.

The block is described by:

- 48 points around its edge
- the plan values at the block
- how much of its edge faces major, secondary and local roads

Buildings that end up mostly outside the block, or overlapping an earlier one, are dropped (`place_buildings`).

## Multiple cities

Prepare each city with the existing pipeline (`python -m urban_dataset prepare-city --config configs/cities/<city>.yaml`), then:

```bash
python scripts/build_layer_data.py --city data/cities/singapore.gpkg --name singapore --output data/layers
python scripts/build_layer_data.py --city data/cities/<other>.gpkg --name <other> --output data/layers
```

Every folder in `data/layers` is one city. The checkpoints store the city names, so the model knows what it was trained on (`generate_city.py --list`). It also stores each city's average plan, which is the default for the sliders.

## Training

The three models are independent and can train at the same time on different GPUs:

```bash
python scripts/train_layer.py --layer plan --data data/layers --output runs/layers-plan --epochs 60 --steps 500 --batch-size 32
python scripts/train_layer.py --layer streets --data data/layers --output runs/layers-streets --epochs 40 --batch-size 32 --d 384 --layers 8 --heads 8 --max-len 1536
python scripts/train_layer.py --layer buildings --data data/layers --output runs/layers-buildings --epochs 40 --batch-size 64 --d 256 --layers 6 --heads 8 --max-len 1024
```

The validation split holds out whole 2 km squares. `--overfit N` trains and validates on N samples, which is a quick gate before a long run.

`meta.json` in each city folder has `street_tokens` and `building_tokens` (p50, p90, p99, max). Pick `--max-len` above the p99. Longer tiles and blocks are cut when training, and a cut street tile teaches the model to stop early.

Building the data for a 12 km test city with 14,000 road pieces and 53,000 buildings takes about 20 seconds on a CPU, so a whole real city should take minutes.

## Generating

```bash
python scripts/generate_city.py --plan runs/layers-plan/best.pt --streets runs/layers-streets/best.pt --buildings runs/layers-buildings/best.pt \
  --output out/city-001 --size 4096 4096 --style singapore=1 --slider roads=1.2 green=1.5 --seed 1
```

Each output folder has:

- `city.json`, the handoff file
- `plan.npy` and `plan.png`, the generated plan
- `city.png`, a top down picture
- `city.obj` and `city.mtl`, a quick 3D preview
- `audit.json`, the checks below

## Checks

`audit.py` measures the finished city:

- network: components, largest component share, dead ends per km, junction share, broken tile seams, road km per km2
- buildings: count, validity, coverage, overlapping pairs, median height, land use counts
- follows plan: the finished city is measured again with the same `plan_raster` code that made the training plans. Then for every channel it reports the planned mean, the realised mean and their correlation. This is how you test the sliders: ask for 1.3 x roads and see whether the realised road density moved by 1.3.

Run the same audit on a real city crop to get the target numbers. Realism is judged against those, plus the pictures.

## Where PCG fits

Nothing stops the model from writing an OBJ, and `city.obj` already is one. The reason not to stop there:

- An OBJ is just triangles. Unreal would not know which triangles are a road, a building or a park, so it could not place lane markings, choose facades, add trees or make collision and navmesh. All of that matters for drone training.
- The model has no data for 3D detail. OSM gives footprints, heights, classes and lines, not facades, roofs, kerbs or street furniture. Anything the model "made" at that level would be invented, not learned.
- PCG is where the asset library lives. A 30 m residential building on this footprint becomes a choice of modular facade, roof and props, with materials and LODs. Changing the assets should not need retraining.
- Variety for reinforcement learning is cheap in PCG: the same plan with different facades, trees, cars and lighting. Changing the plan itself is the model's job.
- Large worlds need streaming. PCG with World Partition builds cells near the camera. One giant OBJ cannot stream.

So the split is: the model decides what goes where (roads, rail, blocks, buildings, heights, land use, water, green), and PCG decides how it looks (meshes, materials, vegetation, props, collision). The JSON is the contract between the two.

## Handoff format (`city.json`)

```text
format: urban-layers-city, version: 0.1.0
units: metres, axes: x east, y north, z up, origin at the south west corner
size_m: [width, height]
seed, style (city weights), sliders, controls (requested plan means)
roads:     id, class (major | secondary | local), vertical (surface | elevated | underground | unknown), width_m, points [[x, y], ...]
rail:      id, class (rail | subway | light_rail | tram | monorail), vertical, points
nodes:     id, xy, mode, vertical, degree, type (endpoint | intersection | continuation | boundary_port)
edges:     id, from, to, mode, class, vertical, width_m, points   (the noded graph, for splines and junctions)
blocks:    id, landuse (none | residential | commercial | industrial | civic | green), polygon
buildings: id, block, kind, height_m, footprint (4 corners)
water:     polygon
green:     polygon
```

A PCG graph can read roads and rail as splines, junctions as points, and buildings as points or volumes with height and kind attributes. Water, green and blocks become surfaces. Elevated roads and rail should be raised and given piers. Underground ones can be skipped or drawn as portals.

## First check on two synthetic cities

Small models (d 128, 4 layers) trained on a CPU for about 20 minutes per layer, on a straight grid city and a curvy organic city. Each run generated a 2 km city from scratch, with no real context at all, and was compared with a 2 km crop of the synthetic city it learned from:

| | grid city | generated grid | curvy city | generated curvy | generated 50/50 mix |
|---|---|---|---|---|---|
| road pieces | 2 | 1 | 20 | 13 | 9 |
| largest piece | 0.99 | 1.00 | 0.87 | 0.89 | 0.90 |
| dead ends per km | 1.0 | 1.0 | 3.7 | 2.6 | 2.2 |
| road km per km2 | 13.2 | 12.8 | 11.2 | 9.2 | 9.4 |
| broken tile seams | | 0 | | 1 | 1 |
| buildings | 2229 | 1146 | 1314 | 859 | 888 |
| median height m | 60 | 45 | 12 | 12 | 10.5 |

A broken tile seam is a dead end within 4 m of an inside tile edge. Over six cities (two seeds each of grid, curvy and the mix) the side rule above took them from 62 to 3, and the largest piece got bigger in four of the six.

Road slider 0.6, 1.0 and 1.5 gave 10.6, 12.8 and 21.2 road km per km2, so the street layer follows the plan density. Buildings undercount mostly because the undertrained model's boxes overlap and get filtered. On real blocks the model asks for the right number (943 against 876, and 922 against 946), and about a quarter are dropped.

Two things these test cities cannot show. Both plans are almost the same everywhere, so the cell by cell correlation between plan and city is low (about 0.13 for roads) because there is little to follow. And in the mix, the plan asks for 30 m buildings but the building model makes 10 m ones, because in the test data the plan height and the city name always agree, so it never had to read height from the plan. Real cities vary inside themselves, so both should be checked in `audit.json` after the real runs.

## What is not done yet

- Only checked here on two synthetic cities on a CPU (a straight grid city and a curvy organic one). Singapore and other real cities need the GPU runs above.
- Buildings are oriented boxes. L shapes and courtyards would need a richer shape token, which is a later step.
- Style mixes are the least tested part. The street and building models only ever saw one city at a time and get a blend of the two city codes. Before the side rule a 50/50 mix had the most broken seams (16 and 19), feeding the models no city at all instead of the blend was no better, and the mix's buildings follow the low rise city.
- Metric Z for elevated and underground transport is not learned, because there is no data for it yet.
- Rail stations are not modelled.
