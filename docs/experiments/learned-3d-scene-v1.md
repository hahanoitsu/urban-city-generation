# Learned 3D scene v1

This is a small experiment, not a replacement architecture yet.

## Question

Can the existing vector city-state data be learned directly as 3D urban objects instead of collapsing the city to a mutually exclusive semantic raster?

The model receives no procedural road planner. Transport geometry is part of the learned target.

## Representation

Each tile is converted to an unordered set of object tokens.

Transport token:

- road or rail
- class
- vertical mode
- XYZ centre
- 3D direction
- length
- width

Building token:

- XYZ centre
- 2D orientation
- footprint length and width
- height
- footprint area hint

Padding is represented as a token type, so the model also learns how many objects should exist.

The first probe uses up to 512 tokens per 1 km tile. Token order is not meaningful and the Transformer has no positional embedding.

## Model

A Transformer denoiser learns the object set with direct clean-target prediction.

Training corrupts the entire object set with continuous noise. Half of updates use high-noise states so generation from pure noise is part of the actual training problem.

The existing 14 city-state morphology descriptors condition the model. They are measured from data and are not rules for where roads should go.

## 3D status

This probe is genuinely object/geometry based rather than RGB or semantic-pixel based. It predicts XYZ transport geometry and building volumes.

The current Singapore source still has limited metric vertical evidence. Surface, underground and elevated transport therefore uses the city-state vertical information currently available, including procedural fallback Z values where the source does not provide metric depth or elevation. Those values must not be presented as measured tunnel/deck heights.

## What the decoder is allowed to do

The decoder converts predicted objects into JSON, OBJ and a top-down preview. It does not choose road routes or place missing roads.

Later deterministic graph reconciliation may snap nearly coincident predicted endpoints and enforce narrow physical constraints. That is compilation, not urban planning.

## Tonight's gate

Run a short training experiment on the corrected Singapore city-state corpus.

Pass criteria for continuing this direction:

1. training and validation loss both fall;
2. pure-noise samples contain non-trivial road, rail and building objects;
3. generated XYZ values stay mostly inside sensible bounds;
4. transport is visibly more object-like than the raster baseline;
5. no hand-written road-layout rules are required.

Failure means revise the learned representation/model. It does not justify returning to a hand-authored city planner.

## Generalisation

Training only on Singapore cannot prove cross-city generalisation. The architecture is intentionally city-agnostic, but the next research gate must add morphologically contrasting cities and use city-held-out validation/test splits.
