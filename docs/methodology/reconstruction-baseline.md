# Reconstruction baseline

The first model tests whether the tile representation can be compressed without losing its main urban structure. It is not a city generator.

The encoder maps a twelve-channel 256 × 256 tile to a 32 × 32 latent map. The decoder predicts:

- road class;
- land-use class;
- water, building and rail masks;
- normalised building height;
- major, secondary and local road centre-lines.

Land-use loss is used only where OSM contains a mapped land-use polygon. Height loss is restricted to building pixels and weighted using the recorded confidence level.

The baseline uses ordinary residual convolution blocks and no encoder skip connections. This keeps the architecture simple and forces information through the bottleneck.

The main measurements are class IoU, building IoU, water and rail IoU, observed-height error and the reconstruction preview sheets.
