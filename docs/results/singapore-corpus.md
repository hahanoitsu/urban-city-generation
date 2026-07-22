# Singapore corpus check

The first multi-area corpus uses six non-overlapping rectangles selected to cover different parts of Singapore.

| Area | WGS84 bounds | Candidates | Accepted | Rejected |
|---|---|---:|---:|---:|
| central-mixed | `[103.7900, 1.2900, 103.8700, 1.3700]` | 56 | 56 | 0 |
| bukit-timah | `[103.7500, 1.3000, 103.7900, 1.3900]` | 27 | 27 | 0 |
| jurong-west | `[103.6900, 1.3150, 103.7500, 1.3850]` | 42 | 42 | 0 |
| north | `[103.7450, 1.4050, 103.8250, 1.4700]` | 48 | 48 | 0 |
| northeast | `[103.8700, 1.3750, 103.9500, 1.4450]` | 56 | 54 | 2 |
| east | `[103.9100, 1.3050, 103.9900, 1.3700]` | 48 | 47 | 1 |

The validation build produced 274 accepted tiles from 277 candidates. The combined audit found no invalid archives, exact duplicates, road-class overlaps or land-use-class overlaps.

With spatial groups of four tiles, the current deterministic split contains 216 training tiles, 36 validation tiles and 22 test tiles.

The source PBF used for this check had SHA-256:

```text
c3142990134efdefc5f356e5c87d46f780810cf5535f1a85c89f4e80cd2b450d
```

The rectangles are an initial sample rather than formal planning-area boundaries. Their purpose is to provide varied morphology while keeping the sampling method explicit and reproducible.
