# Semantic diffusion layout generation

The deterministic extension baselines are kept as negative results. They learned broad averages and did not reproduce sharp real continuations, even when the same four examples were used for both training and validation.

The active layout experiment follows the sequence described by CityGen more closely:

1. learn complete local semantic blocks with an unconditional diffusion model;
2. initialise an outpainting model from that block checkpoint;
3. preserve the known region during every denoising step;
4. refine boundaries and add heights in later stages.

CityGen's public repository currently contains the paper and project material but not its promised training code, model weights or dataset. This project therefore reproduces the published method with the maintained `UNet2DModel`, `DDPMScheduler` and `DDIMScheduler` implementations from Hugging Face Diffusers.

## Semantic field

The original twelve-channel tiles are reduced to one categorical field:

1. terrain;
2. vegetation;
3. building;
4. road;
5. rail;
6. water.

Road hierarchy remains available in the source tensors and in the three boundary-guide channels, but the first diffusion model generates one road class. This is intentional. The block generator first needs to learn coherent city structure; road hierarchy can then be propagated from boundary crossings or classified in a later stage.

The categorical field is one-hot encoded and scaled from `{0, 1}` to `{-1, 1}`. Height and land use are not included in this model. Height synthesis is a separate stage, following CityGen's decision to avoid inconsistent semantic and height fields during repeated expansion.

## Stage 1: local block generation

`SemanticBlockDataset` creates dense crops from every real tile. With the current settings, each 256 × 256 tile produces 25 overlapping 128 × 128 crops:

```text
crop size: 128 pixels
stride:     32 pixels
positions:   5 × 5 per tile
```

The current Singapore training split therefore produces about 5,400 block examples before rotations and flips. These crops remain inside their original train or validation manifest, so the data split is unchanged.

For a clean semantic field `x0`, a random timestep `t` and Gaussian noise `epsilon`, the scheduler creates `xt`. The U-Net predicts `epsilon` from `xt` and `t`. Mean squared error is calculated over the complete block.

The training code uses:

- Hugging Face Diffusers `UNet2DModel`;
- `DDPMScheduler` for the forward noising process;
- epsilon prediction;
- AdamW;
- exponential moving average model weights;
- `DDIMScheduler` for faster preview sampling.

The block generator must produce recognisable complete layouts before outpainting begins.

## Stage 2: masked outpainting

The outpainting model uses the same U-Net structure and output classes, but its input has four extra channels:

- one known-region mask;
- major-road boundary guide;
- secondary-road boundary guide;
- local-road boundary guide.

The block checkpoint initialises all matching tensors. The original semantic input weights are copied into the wider input convolution and the new mask and guide weights start at zero.

During training, the U-Net predicts diffusion noise over the full two-tile field, but loss is calculated only in the unknown half. During sampling, the correctly noised real seed is copied into the known region at every denoising step. The final clean semantic seed is copied back exactly after sampling.

This is the CityGen-style masked outpainting baseline. The three road guide channels are the project-specific addition used to test whether explicit geospatial boundary constraints improve road continuity.

## What is not included yet

This branch does not yet include:

- multi-scale boundary refinement;
- a road-topology repair graph;
- road-hierarchy prediction in the generated region;
- morphology or city-style conditioning;
- height synthesis;
- vector or Unreal Engine export.

Those stages should only be added after the block model learns a credible local distribution and the outpainting model produces visibly diverse continuations.

## References

- Jie Deng et al., *CityGen: Infinite and Controllable City Layout Generation*, arXiv:2312.01508 and CVPR Workshops 2025.
- Jonathan Ho, Ajay Jain and Pieter Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Jiaming Song, Chenlin Meng and Stefano Ermon, *Denoising Diffusion Implicit Models*, ICLR 2021.
- Hugging Face, *Diffusers: State-of-the-art diffusion models*.
- Yu Shang et al., *UrbanWorld: An Urban World Model for 3D City Generation*, arXiv:2407.11965. This is reserved for the later 3D stage.
