# Provenance and Reproduction Boundary

## Run environment

* Python 3.8 conda environment `logicAL`; PyTorch 2.0.1+cu117 and
  torchvision 0.15.2 (from `requirements.txt`).
* Four NVIDIA RTX A6000 GPUs; `DataParallel` for DeepSIM and the localization
  network.
* The released logs record the run-level facts:
  * `results/logs/08_deepsim_strict_train.log`:
    `DEEPSIM_V2_FROM_SCRATCH images=1778 batch=56 gpus=4 loss=GAN+FM+VGG`,
    300 epochs from random initialisation.
  * `results/logs/10_localization_strict_train.log`:
    `ONLINE_LOGICAL_FROM_SCRATCH train=1778 test=1136 gpus=4`, 300 epochs.
  * `results/logs/09_generate_strict_images.log`: 1,778 generated
    normal/anomaly examples for the qualitative bank.

The log file names retain the execution machine's working paths (for
example `/home/hello/program2/logicAL/...`).  They are kept verbatim as run
records; they are not portable paths.

## Code ↔ log boundary

The DeepSIM log header says `DEEPSIM_V2_FROM_SCRATCH`, while the released
`src/train_deepsim_unified.py` prints `DEEPSIM_UNIFIED_FROM_SCRATCH`.  The
released script is a **cleaned copy** of the run-time script (message text,
path handling and documentation edits); the training objective, schedule,
batch size and generator definitions are the ones documented in
[`PAPER_SETTINGS.md`](PAPER_SETTINGS.md).  This is stated rather than hidden
so that a reviewer is not surprised by the header difference.

No source hash was captured at run time for this archive.  The released
evidence is covered by `results/SHA256SUMS`; a reviewer can verify that the
score file, result trajectory and logs have not changed.

## What is and is not redistributed

| Item | Included? | Notes |
|---|---|---|
| Code (PiDiNet stage, DeepSIM training, online LAD, verification) | Yes | `src/`, `scripts/`, `tools/` |
| Official PiDiNet `table5_pidinet.pth` | Yes | `checkpoints/`; SHA-256 in `checkpoints/README.md` |
| CSAD component masks (1,778) | Yes | Derived annotations for normal training images; `data/csad_component_masks/SHA256SUMS` |
| MVTec LOCO images | No | MVTec licence; download from the official site |
| Trained DeepSIM generator | No | ≈2.25 GB per model; not retained |
| Trained localization checkpoint | No | Not retained; results and per-image scores are released |
| MVTec pixel ground truth | No | Needed only for sPRO, which is not reported |

## CSAD mask provenance

The masks are the `CSAD_LOCO_refined_masks_color` output of the CSAD
pipeline (Apache-2.0), generated from the normal MVTec LOCO training images.
For example, `data/csad_component_masks/breakfast_box/000.png` is
byte-identical to
`CSAD_LOCO_refined_masks_color/breakfast_box/000.png`.  The full per-file
manifest is `data/csad_component_masks/SHA256SUMS`.  The masks are
annotations, not MVTec images; MVTec's dataset licence still applies to the
underlying images.

## What a reviewer can verify without the dataset or checkpoints

* `python tools/verify_results.py` — recomputes the reported AUROC table from
  the released per-image scores (standard library only).
* `results/localization_strict/results.json` — all 30 evaluation points
  (epochs 10–300), including the final row.
* `results/logs/` — 300-epoch training logs for DeepSIM and localization.
* `results/SHA256SUMS` — integrity of the released evidence.

## What requires the dataset and a retraining run

* Re-executing PiDiNet edge extraction, DeepSIM training and online LAD
  training (`scripts/run_all.sh`).
* Any pixel-level metric.  The archive intentionally does not report sPRO;
  see [`METRICS.md`](METRICS.md).
