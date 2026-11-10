# Metrics: What Is Reported and What Is Not

## Scope

This archive covers the MVTec LOCO AD logical-anomaly subset only.  It does
not reproduce the paper's MVTecAD, VisA or MADsim experiments, the Table 1
all-anomaly comparison, the Table 3 ablation, or any pixel-level metric.  It
reports one result: image-level AUROC on `test/good` vs
`test/logical_anomalies`.

## Image-level AUROC (reported)

* **Test images.**  Official MVTec LOCO `test/good` (label 0) and
  `test/logical_anomalies` (label 1).  `test/structural_anomalies` is **not**
  included, so the metric is label-matched to the paper's Table 2 *logical*
  column, not to Table 1's all-anomaly column.
* **Counts.**  1,136 images: 575 good + 561 logical.  Per category:
  breakfast 102/83, juice 94/142, pushpins 138/91, screw bag 122/137,
  splicing 119/108 (good/logical).
* **Score.**  Mean of the top 1% of the sigmoid segmentation probabilities.
  The paper does not publish its pooling rule; this choice is recorded in
  [`ASSUMPTIONS.md`](ASSUMPTIONS.md).
* **Aggregation.**  Per-category AUROC with average ranks for ties, then an
  equal-weighted mean over the five categories.
* **Checkpoint.**  Final epoch 300, evaluated every 10 epochs; the complete
  30-point trajectory is in `results/localization_strict/results.json`.  No
  test label is used for training, early stopping or checkpoint selection.
  The released code computes test metrics every 10 epochs to record that
  trajectory; the reported row is the fixed final epoch, and no checkpoint or
  hyperparameter is chosen from any test row.  If a strict single-touch
  protocol is required, treat the intermediate rows as diagnostic only.
* **Reproduction.**  `python tools/verify_results.py` recomputes every value
  from `scores_epoch_300.csv` using only the Python standard library.

## Comparison with the paper

The paper reports a mean **logical** image AUROC of **84.6** (Table 2) and a
mean all-anomaly image AUROC of **88.5** (Table 1).  The reproduced 75.03 is
label-matched to the 84.6 logical value.  The paper's five category values
(85.4, 98.5, 87.4, 82.0, 89.0; mean 88.5) are all-anomaly values and are shown
in the README only as an indication of the per-category gap; they are not a
label-matched comparison.

## Pixel-level AU-sPRO (not reported)

The paper's second main metric is saturated per-region overlap (AU-sPRO at
FPR 5%).  This archive does **not** report it.  AU-sPRO is not a simple
pixel-AUROC: it needs

1. the trained localization checkpoint (not retained; DeepSIM and the
   localization network are not redistributed),
2. MVTec LOCO `ground_truth/` annotations for both logical and structural
   anomalies (not redistributed), and
3. the official `defects_config.json` saturation thresholds
   (not redistributed).

Without all three, an "sPRO" number would not be the paper's metric.  We
therefore omit it rather than substitute a different metric.  Reviewers who
download MVTec LOCO and retrain the released pipeline can run the official
MVTec LOCO evaluator on the resulting anomaly maps; the archive does not
claim the paper's 68.8 (logical) or 69.7 (overall) sPRO values.

## What the released numbers support

* The image-level ranking of logical anomalies versus good images on the
  official test split, with the released per-image scores and the released
  code.
* A documented reproduction gap of 9.6 percentage points to the paper's
  logical mean (75.03 vs 84.6), with the known deviations listed in
  [`PAPER_SETTINGS.md`](PAPER_SETTINGS.md) and
  [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

They do **not** support a claim that LogicAL's full LAD performance (including
pixel localization) has been reproduced.
