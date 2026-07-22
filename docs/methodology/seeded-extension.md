# Seeded map extension baseline

The extension task starts with one real tile and predicts the neighbouring tile. The original tile is treated as a fixed seed and is copied back unchanged after inference.

## Training examples

Adjacent tiles are found from their city, area, grid column and grid row in a normal dataset manifest. Each directed pair is rotated into the same layout:

```text
known seed | hidden continuation
```

This keeps every batch at 256 × 512 pixels even when the original continuation was north, south or west. A vertical flip is the only augmentation applied after pairing.

The model input has sixteen channels:

- twelve semantic channels with the hidden half set to zero;
- one known-region mask;
- three road-boundary guide channels for major, secondary and local roads.

The guide is made from road centre-lines that reach the edge of the seed. Those crossings are extended a short distance into the hidden region so the required road class and crossing position are explicit.

## Supervision

The output heads are the same as the reconstruction baseline. Losses and metrics are masked to the hidden half, so the network is not rewarded for copying the seed. Land-use and height confidence masks continue to control where supervision is reliable.

Validation also reports boundary road recall. A crossing is counted as matched when the predicted centre-line appears close to the required row within the first part of the generated tile.

## Inference

The `extend` command accepts a saved tile, checkpoint and direction. The tile is rotated into the canonical eastward layout, the continuation is predicted, and the result is rotated back. The saved output contains the generated tile, predicted road centre-lines and a combined seed-plus-extension tensor.

This is a deterministic conditional baseline. It establishes the data representation, boundary conditions and evaluation path before a probabilistic latent model is added.
