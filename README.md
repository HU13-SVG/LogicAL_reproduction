# LogicAL* — Paper-Guided Reproduction on MVTec LOCO

Paper-guided re-implementation of LogicAL (Zhao, 2024): edge manipulation, one
unified DeepSIM generator, and an OmniAL-style localization network.

Result: **75.03** mean image-level AUROC on the five MVTec LOCO logical-anomaly
test subsets (final epoch 300).  This archive reports **image-level AUROC
only**; the paper's pixel-level AU-sPRO, MVTecAD/VisA/MADsim experiments and
ablation are not reproduced.

## Result

| Category | Reproduced (logical) | Paper, all anomalies (Table 1) |
|---|---:|---:|
| Breakfast box | 78.03 | 85.4 |
| Juice bottle | 93.70 | 98.5 |
| Pushpins | 78.26 | 87.4 |
| Screw bag | 61.76 | 82.0 |
| Splicing connectors | 63.41 | 89.0 |
| **Mean** | **75.03** | **88.5** |

The paper's label-matched logical mean is **84.6** (Table 2).  The per-category
paper values above aggregate all anomaly types, so only the mean row is
comparable.  Test split: 1,136 images (`test/good` 575, `test/logical_anomalies`
561).  The final epoch is reported; no test-based checkpoint selection is used.

## Layout

```text
src/                    edge extraction, DeepSIM, synthesis, localization
scripts/run_all.sh      edges -> DeepSIM -> online LAD
data/csad_component_masks/   1,778 component masks (SHA256SUMS included)
checkpoints/            official PiDiNet table5_pidinet.pth
results/                results.json, scores_epoch_300.csv, logs, qualitative
docs/                   PAPER_SETTINGS, ASSUMPTIONS, METRICS, PROVENANCE, CODE_MAP
tools/verify_results.py stdlib-only AUROC recomputation
third_party/            vendored PiDiNet and DeepSIM sources
```

## Install and data

```bash
conda create -n logical python=3.8 -y
conda activate logical
pip install -r requirements.txt
```

Download MVTec LOCO and arrange `data/mvtec_loco/<category>/train/good`,
`validation/good`, and `test/{good,logical_anomalies,structural_anomalies}`.
The CSAD masks and the PiDiNet checkpoint are included.  Trained DeepSIM and
localization checkpoints are not included.

## Run

```bash
export MVTEC_LOCO_ROOT=/path/to/mvtec_loco
export WORK_DIR=/path/to/work
bash scripts/run_all.sh
```

The pipeline runs PiDiNet edge extraction, 300-epoch unified DeepSIM training,
then 300-epoch online edge manipulation + localization.  A full run needs a
CUDA GPU (the released run used four A6000s).

## Verify

```bash
python tools/verify_results.py
```

The checker uses only the Python standard library, recomputes every category
AUROC and the macro mean from `results/localization_strict/scores_epoch_300.csv`,
and prints `Result: PASS`.

## Scope and deviations

* CSAD component masks replace the paper's SAM + refinement.
* Localization batch is 12; the paper's `15(12)` notation is ambiguous and the
  most natural reading gives 15 at 256×256 for MVTec LOCO.
* The paper's second TPS warp of synthetic anomalies and the separate DiffNeck
  module are not implemented.
* Pixel-level AU-sPRO is not reported; no checkpoint, MVTec pixel ground truth
  or `defects_config.json` is redistributed.
* DeepSIM training uses the public-code default joint horizontal flip.

Details: [`docs/PAPER_SETTINGS.md`](docs/PAPER_SETTINGS.md),
[`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md),
[`docs/METRICS.md`](docs/METRICS.md),
[`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Licence

Research use.  Vendored DeepSIM and PiDiNet sources retain their upstream
licences; MVTec LOCO images remain subject to the MVTec licence.
