# DeepSIM (trimmed)

Source: <https://github.com/eliahuhorwitz/DeepSIM> (ICCV 2021 Oral), based on
<https://github.com/NVIDIA/pix2pixHD>.

The following files are copied from the upstream repository:

```text
deepsim_models/networks.py
data/base_dataset.py
util/tps_warp.py
util/warp_image.py
LICENSE
```

They are used for the pix2pixHD generator/discriminator definition, the
VGG-19 perceptual feature extractor, and the TPS augmentation applied jointly
to edge/RGB pairs.  No functional changes were made.

The upstream licence is a research licence; see `LICENSE`.
