# Reviewer Guide

## Claim

This archive is a paper-guided re-implementation of the LogicAL logical-anomaly
synthesis module and the LAD pipeline (OmniAL-style localization network) on
MVTec LOCO AD.  It reports **image-level AUROC**; it is not a bit-exact
reproduction of the authors' code, which was not released, and it does not
report the paper's pixel-level AU-sPRO.

## Where to check paper adherence

* [`docs/PAPER_SETTINGS.md`](docs/PAPER_SETTINGS.md) separates settings stated
  by the paper, settings inherited from the public implementations the paper
  cites, and our choices.  Only the first category is a paper claim.
* [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) lists every choice the paper
  does not fix, including the CSAD-mask substitute for SAM+refinement and the
  localization batch-size deviation.
* [`docs/METRICS.md`](docs/METRICS.md) defines the reported image-level metric
  and explains why sPRO is not reported.
* [`docs/PROVENANCE.md`](docs/PROVENANCE.md) records the run environment, the
  code/log boundary, and what is and is not redistributed.
* The training and evaluation code is in `src/`:
  * `prepare_pidinet_edges.py` — PiDiNet, first-scale edges;
  * `train_deepsim_unified.py` — one five-category DeepSIM, 300 epochs;
  * `train_localization_online.py` — online Eq. (1) synthesis + 300-epoch LAD;
  * `logical_net.py` — 48/96/192/384 DCSA localization network.

## Result evidence

* `results/localization_strict/results.json` — all 30 evaluation points
  (epochs 10–300), including the final epoch.
* `results/localization_strict/scores_epoch_300.csv` — 1,136 per-image scores
  and labels.
* `results/localization_strict/summary.csv` — paper-vs-reproduction table.
* `results/logs/` — strict DeepSIM, generation and localization logs.
* `results/SHA256SUMS` — integrity of the evidence files.

Run:

```bash
python tools/verify_results.py
```

The checker uses only the Python standard library.  It recomputes every
category AUROC and the 75.03 macro mean from the released per-image scores
with an independent Mann–Whitney implementation.

## Numbers to read carefully

* Reproduced mean: **75.03%** on `test/logical_anomalies` vs `test/good`
  (1,136 images), final epoch 300.
* The paper's Table 2 reports a mean logical-anomaly AUROC of **84.6%**; this
  is the label-matched comparison.  The gap is **9.57 percentage points**.
* The paper's per-category values (85.4, 98.5, 87.4, 82.0, 89.0; mean 88.5)
  aggregate all anomaly types.  They are shown for context and are not
  label-matched to 75.03%.
* The final epoch is reported, not the best test epoch.  The trajectory shows
  epoch 290 at 75.26 and epoch 300 at 75.03; no test-based selection is used.
  The code computes the test trajectory every 10 epochs, but the reported row
  is the fixed final epoch and no checkpoint/hyperparameter is chosen from it;
  intermediate rows are diagnostic.
* `test/structural_anomalies` is excluded, so the metric matches the paper's
  logical column, not its all-anomaly column.

## Known deviations to inspect

1. **Localization batch size.**  The paper's "15(12)" notation is ambiguous;
   the most natural reading gives 15 at 256×256 for MVTec LOCO.  The released
   run used 12 (the script default).  This is disclosed in
   `docs/ASSUMPTIONS.md` and `docs/PAPER_SETTINGS.md`.
2. **Semantic regions.**  CSAD component masks replace SAM + refinement.
3. **Second TPS warp.**  The paper applies TPS to the synthetic anomaly edge
   map/image (Fig. 5e–f).  The released online trainer applies TPS only to the
   normal DeepSIM training pairs; the second warp is not implemented.
4. **DiffNeck.**  The paper describes OmniAL's DiffNeck module; the released
   network uses multi-scale bilinear/concat/ConvINAct fusion instead and does
   not implement a separate DiffNeck.  This is an architectural deviation, not
   a claim of exact OmniAL reproduction.
5. **JND map.**  The paper cites JND but gives no formula; the released code
   uses a 5×5 local-luminance/contrast map (documented in `ASSUMPTIONS.md`).
6. **Horizontal flip.**  The paper is silent; the run used the
   DeepSIM/pix2pixHD public default joint flip.  `--no-flip` is available.
7. **Pixel-level sPRO.**  Not reproduced; no checkpoint, MVTec ground truth or
   `defects_config.json` is redistributed and no substitute metric is claimed.

## Reproducibility boundary

Trained DeepSIM and localization checkpoints are not included.  The released
evidence is the per-image score file, the full 30-point trajectory and the
training logs.  Re-executing training requires MVTec LOCO and four GPUs (or a
longer single-GPU run); exact bitwise reproduction is not guaranteed.
