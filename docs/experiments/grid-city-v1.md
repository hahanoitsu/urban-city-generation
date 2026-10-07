# Grid city v1

## Why

The structured city denoiser could memorise 16 tiles but failed on held-out regions. On the 16 test regions it made about 32 road pieces per tile (real tiles have about 6), 41% isolated nodes and no valid buildings. 63% of its roads ran sideways (20% in the real tiles) and buildings piled up near the top.

The cause is slot identity. Nodes and buildings were sorted top to bottom, so slot k meant "the k-th thing from the top". On a new region the model falls back to that prior: edges join nodes of similar rank, which have similar height but any x, so they become horizontal stripes.

## Representation

Every slot now belongs to a place.

- The 512 m tile is split into cells. Nodes and buildings use a 16 x 16 grid (32 m cells), areas use 8 x 8 (64 m cells).
- Each cell owns a fixed number of slots: 10 node slots, 8 building slots, 8 area slots. Slots inside a cell are sorted by position.
- Roads are cut into segments of at most 28 m, so every segment joins nodes in the same or a neighbouring cell.
- Each node owns up to 4 edge slots. An edge slot points at one of the 90 node slots in the surrounding 3 x 3 cells, or at nothing. The pointer is a local category, not an index over the whole tile.
- Node positions are relative to the cell centre. Buildings and areas store a centre relative to the cell, a log size and a shape in their own unit frame. Road curves are stored relative to their segment.
- Presence is a masked category for every slot, so the model decides how many things exist in each cell. There are no count heads.
- Training uses the 8 rotations and flips of the tile, including the ports and the context offsets.

Decoding merges chains of segments with the same mode, class and vertical mode back into long edges, so the output json has the same schema as before and the existing audit works unchanged.

On 32 real tiles the encoding round trips with 99.5% median building overlap, 0.65 m 99th percentile road error, no rejected tiles, 6 dropped buildings out of 1,541 and 5 dropped areas out of 665.

## Run

Train, sample and audit in one go:

```bash
python scripts/train_grid_city.py --data DATA --output RUN --cache-dir CACHE
python scripts/sample_grid_city.py --data DATA --checkpoint RUN/best.pt --output RUN/gen-test --cache-dir CACHE --split test
python -m urban_analysis.structured_generation_audit --generations RUN/gen-test --targets RUN/gen-test/targets
```

The gate is the held-out test split: road pieces per tile, isolated nodes, valid buildings and the pictures.
