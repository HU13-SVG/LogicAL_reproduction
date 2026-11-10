# Third-party source

This directory contains trimmed copies of the upstream model definitions used
by the reproduction.  Only the modules required to build the networks and the
DeepSIM augmentation are included; training scripts, datasets, images and
upstream checkpoints are omitted.  The original licences are retained in each
subdirectory.

## DeepSIM / pix2pixHD

* `deepsim/` — model definitions and TPS augmentation from
  [DeepSIM](https://github.com/eliahuhorwitz/DeepSIM) (ICCV 2021 Oral), which
  is based on [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
* Included: `deepsim_models/networks.py`, `data/base_dataset.py`,
  `util/tps_warp.py`, `util/warp_image.py`.
* Licence: `deepsim/LICENSE` (SOFTWARE RESEARCH LICENSE, non-commercial
  research use).  The TPS warping file also retains its CellTool GPL-2.0
  header.

## PiDiNet

* `pidinet/` — model definitions from the official PiDiNet implementation
  (Su et al., "Pixel Difference Networks for Efficient Edge Detection").
* Included: `models/{__init__,config,convert_pidinet,hed_fps,ops,ops_theta,pidinet}.py`.
* Licence: `pidinet/LICENSE` (non-commercial research use; commercial use
  requires contacting the authors).

## Modifications

* DeepSIM: no functional changes; only the upstream files needed by
  `train_deepsim_unified.py` were copied.
* PiDiNet: no functional changes; only the model definitions were copied.

These directories are imported through `sys.path` by the scripts in `src/`.
