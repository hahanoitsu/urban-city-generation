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

### 2. Water, green and land use

Traced from the plan (`city.py`). The plan is upsampled to 16 m with a cubic spline first, so the edges are smooth curves and not the 128 m grid. Water is traced at the level that keeps the planned amount of water (`plan_polygons`). Land use comes from `landuse_regions`: every spot takes the use (residential, commercial, industrial, civic or green) with the biggest share in the plan there, and whatever no use covers is none. These regions decide the land use of the blocks, so the land use sliders act on the city directly.

### 3. Streets

`streets.py`. One 512 m tile at a time. Each tile is a token sequence: for every line, its class (road class or rail type, times surface, elevated, underground or unknown), then its points as x, y tokens on a 1 m grid, then an end token. A grammar mask during sampling makes every output well formed.

The lines are written in the order a network grows. First the lines that continue a port, then the network spreads: the next line is always one that starts on a line already written if there is one, major roads first, then secondary, then local. A line with nothing to start from begins a new piece. Two extra tokens say where lines meet: JOIN right after the class means the line starts on another line, and END_JOIN before the last point means it ends on another line. At generation, a joined end is snapped onto the nearest line of the same mode and level within 40 m, and an end without a join is left alone (only moved if it is within 3 m of a line), because real cul-de-sacs exist.

This matters because of how the first real run failed (see below). With lines in plain raster order, a junction only happens if the model copies an exact coordinate of an earlier line to the metre. It mostly missed, and the city fell apart into floating pieces. Now the model only has to say "this end meets a road" and get roughly close.

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

No learning, just geometry. Joined ends are snapped as above. When an end meets the end of another line, the two become one junction, and if that other end moves too, both move together, so chains of snaps do not leave gaps. The meeting point is added to the other line as a vertex so the junction is exact. Then `urban_dataset.city_state.build_transport_graph` nodes everything into the usual graph.

Blocks are the polygons between surface roads and surface rail, shrunk by 3 m, minus water, then cut along the land use regions (`landuse_blocks`). This step does not invent structure. It reads off what the roads and the plan already decided. Blocks bigger than 384 m are split into parts for the building model, but only inside the building step, so the output blocks keep their real shapes.

### 6. Buildings

`buildings.py`. One block at a time, in the block's own frame, where the x axis is the block's long side. The sequence is the block's land use, then for every building: kind, x, y, width, depth, angle, height.

The block is described by:

- 48 points around its edge
- the plan values at the block
- how much of its edge faces major, secondary and local roads

The land use token is not sampled. It is set to the block's land use from the plan, and the model draws buildings for that use.

The model puts some boxes a few metres off. A box that is less than 70% inside its block is moved the shortest way back in (at most 15 m), and only dropped if that does not work or it overlaps an earlier box (`place_buildings`). Real boxes pass this almost always, so it only fixes the model's aim.

Every block part gets 4 samples (`--building-samples`), and the one whose built area is closest to the planned coverage there is kept. One scale for the whole city is chosen so the total built area matches the plan, which is what makes the buildings slider work.

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

- network: components, lone lines (a piece that is a single line with two loose ends), largest component share, dead ends per km, junction share, broken tile seams, road km per km2
- buildings: count, validity, coverage, overlapping pairs, median height, land use counts
- follows plan: the finished city is measured again with the same `plan_raster` code that made the training plans. Then for every channel it reports the planned mean, the realised mean and their correlation. This is how you test the sliders: ask for 1.3 x roads and see whether the realised road density moved by 1.3.

`scripts/audit_real_city.py` cuts windows of the same size out of a real training city, mostly built up ones, and runs the same audit on them. Its `summary.json` gives the target numbers. Realism is judged against those, plus the pictures. Real data has some broken tile seams too, just from dead ends that happen to sit near a 512 m grid line.

```bash
python scripts/audit_real_city.py --city data/cities/singapore.gpkg --plan data/layers/singapore/plan.npz --output runs/layers-real --size 4096 --count 6
```

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

## First run on Singapore

Data from `singapore-v2.gpkg`: a 50 by 34 km box, 129,140 road and rail lines, 12,664 street tiles, 22,059 blocks with 130,789 of the 158,117 buildings inside them. Street tiles are 222 tokens at the median and 860 at p99. More than half the blocks have no buildings at all (parks, road islands, interchanges).

Training on the server took 51 minutes for the plan (validation 0.70 to 0.24), 28 minutes for streets (4.82 to 2.51, still falling at the end) and 38 minutes for buildings (best 2.02 around epoch 25, then it overfits, and `best.pt` keeps that epoch).

The plan looked like a real district layout, with dense blocks, a green core and water. The streets did not. In a 4 km city, 2,117 of the 4,711 road nodes were dead ends and the largest connected piece held only 11% of the network. 368 of its 544 pieces were a single short line floating on its own with two loose ends, and the median line was 44 m long. Buildings were reasonable on their own, but blocks only form between connected roads, so large leftover areas got cut into 384 m squares.

The cause was the line order. Lines were sorted by class and then by where they start, so nothing in a sequence said "start on a road that is already there". To make a junction the model had to repeat an exact coordinate from earlier in the sequence, an undertrained model sampled at temperature 1 mostly missed, and every miss became a floating piece. That is what the growth order and the join tokens in the streets section fix.

Checked on a test city built by growing roads off other roads, so it is full of T junctions and cul-de-sacs like the Singapore estates. The old and new street models were trained the same way on a CPU (d 128, 4 layers, 3,200 steps) and each made three 2 km cities from the same plans. Averages:

| | real test city | old line order | growth order and joins |
|---|---|---|---|
| road pieces | 6 | 112 | 45 |
| lone lines | 4 | 81 | 33 |
| largest piece | 0.98 | 0.22 | 0.75 |
| dead ends per km | 3.2 | 10.1 | 4.8 |
| road km per km2 | 13.3 | 11.0 | 11.5 |

The old order reproduces the Singapore failure on its own. The new one fixes most of it in the same training time, but it is not at the real numbers yet, and these are small models. Sampling at temperature 0.6 made both worse (shorter roads and about twice the pieces), so 1.0 stays the default.

## Second run on Singapore

Streets were retrained with the growth order and joins for 120 epochs (95 minutes, validation still slowly falling at 1.97). Real numbers come from `audit_real_city.py` on six 4 km windows, generated ones from 4 km cities with the same plan model as before:

| | real Singapore (median) | generated seed 1 | generated seed 2 | first run seed 1 |
|---|---|---|---|---|
| road pieces | 114 | 132 | 135 | 554 |
| lone lines | 89 | 90 | 91 | 368 |
| largest piece | 0.72 | 0.83 | 0.69 | 0.11 |
| dead ends per km | 3.2 | 4.1 | 3.9 | 12.7 |
| road km per km2 | 12.1 | 10.3 | 10.1 | 10.1 |
| building coverage | 0.175 | 0.065 | 0.060 | 0.057 |

Real Singapore has many pieces too. Bridges, tunnels and roads of unknown level are their own groups in `build_transport_graph`, and service roads get cut at the window edge. The roads now follow the plan closely (densities within about 10%, correlations 0.77 to 0.91), and the road slider works (1.4 x roads gave 1.37 x road density).

Two things were still wrong. The city looked blocky, because big areas with few roads became huge blocks that were cut into 384 m squares, and every square sampled its own land use. And buildings covered about half of what the plan asked for. On test city blocks the building model draws the right amount (1,420 boxes and 240,000 m2 against 1,421 and 236,000 real), but 38% were thrown away for sticking out of the block, because the model's aim is a few metres off. Real boxes almost never stick out.

The fixes are the land use regions, the nudging and the building samples described above. None of them needs retraining. On the test city the built area went from 41% of the plan to 99%.

## What is not done yet

- Only checked here on two synthetic cities on a CPU (a straight grid city and a curvy organic one). Singapore and other real cities need the GPU runs above.
- Buildings are oriented boxes. L shapes and courtyards would need a richer shape token, which is a later step.
- Style mixes are the least tested part. The street and building models only ever saw one city at a time and get a blend of the two city codes. Before the side rule a 50/50 mix had the most broken seams (16 and 19), feeding the models no city at all instead of the blend was no better, and the mix's buildings follow the low rise city.
- Metric Z for elevated and underground transport is not learned, because there is no data for it yet.
- Rail stations are not modelled.
