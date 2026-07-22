# Extension U-Net experiment

The first extension baseline reused the reconstruction autoencoder. It could learn broad land-use regions, but roads and buildings were blurred because every prediction passed through the same compressed bottleneck.

This experiment keeps the same adjacent-tile task and boundary guide, but changes the network used for extension.

## Network

`ExtensionUNet` uses encoder-to-decoder skip connections. The target half of the input is still zero, so the skips cannot copy the real continuation. They only preserve precise information from the seed tile, the known-region mask and the road guide.

The reconstruction experiment still uses the original autoencoder. The architecture is selected with `model.architecture` in the training YAML.

## Road supervision

The centre-line loss now combines weighted binary cross entropy, Dice and Tversky losses. Tversky gives a larger penalty to missed road pixels than to a small amount of extra prediction.

A separate boundary loss checks the explicit road guide. Validation also reports the average number of guide pixels for which the required road remains present, rather than only checking the first crossing.

## Memorisation check

`configs/extension-overfit.yaml` limits training to four directed pairs, disables augmentation and uses the training manifest for validation. This is not a research result. It is a diagnostic: the model should be able to memorise a few examples before it is trusted on the full corpus.

The normal experiment is in `configs/extension-unet.yaml`.
