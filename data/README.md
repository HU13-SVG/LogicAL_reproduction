# Data

## MVTec LOCO AD

Download the dataset from
<https://www.mvtec.com/company/research/datasets/mvtec-loco> and arrange it as:

```text
data/mvtec_loco/
├── breakfast_box/
│   ├── train/good/*.png
│   ├── validation/good/*.png
│   └── test/{good,logical_anomalies,structural_anomalies}/*.png
├── juice_bottle/
├── pushpins/
├── screw_bag/
└── splicing_connectors/
```

Only `train/good` is used for edge extraction, edge-to-image training and
anomaly synthesis.  The localization test uses `test/good` and
`test/logical_anomalies`.  No test image is used for training, augmentation,
threshold selection or checkpoint selection.

The reproduction uses 1,778 normal training images:

| Category | Normal training images |
|---|---:|
| Breakfast box | 351 |
| Juice bottle | 335 |
| Pushpins | 372 |
| Screw bag | 360 |
| Splicing connectors | 360 |
| **Total** | **1,778** |

## CSAD component masks

`csad_component_masks/` contains 1,778 colour-coded component masks, one per
normal training image, named by the MVTec file stem
(`<category>/<stem>.png`).  They are the deterministic semantic-region
substitute for the paper's SAM + refinement step.

The masks are the `CSAD_LOCO_refined_masks_color` output of the CSAD
pipeline (Apache-2.0), generated from the normal MVTec LOCO training images.
For example, `breakfast_box/000.png` is byte-identical to the corresponding
CSAD output.  `SHA256SUMS` lists every released mask; the archive includes a
per-file hash manifest so the substitute can be audited.

The masks are **not** the paper's SAM + refinement output.  They are a
deterministic component-segmentation substitute, used only for normal
training images.  If you prefer the original paper's SAM-based regions,
replace this directory with your own masks in the same format and pass
`--mask-root` to the trainer.

The colour-coded format is: background `(0,0,0)`; each component has one
distinct colour.  The code selects one non-background colour with more than
150 pixels per mask.

MVTec LOCO images are not redistributed here and remain subject to the MVTec
licence.  The masks are derived annotations; redistributing or reusing them
does not grant any right to the underlying MVTec images.
