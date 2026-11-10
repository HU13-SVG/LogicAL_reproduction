# Released Results

## Files

| Path | Contents |
|---|---|
| `localization_strict/results.json` | All 30 evaluation points (epochs 10–300); per-category AUROC and macro mean at each point |
| `localization_strict/scores_epoch_300.csv` | Per-image scores/labels for the final epoch (1,136 test images) |
| `localization_strict/summary.csv` | Paper-vs-reproduction table |
| `logs/08_deepsim_strict_train.log` | Unified DeepSIM training log (`images=1778 batch=56 gpus=4`) |
| `logs/09_generate_strict_images.log` | Qualitative generation log |
| `logs/10_localization_strict_train.log` | Online LAD training log (`train=1778 test=1136 gpus=4`) |
| `qualitative/contact_sheet.jpg` | Five normal/generated/difference examples |
| `qualitative/samples/` | The five normal/generated pairs and difference maps |
| `qualitative/samples.csv` | Category, source/candidate files, operation, region type |
| `SHA256SUMS` | SHA-256 of every file in this directory tree |

`python tools/verify_results.py` recomputes the table from
`scores_epoch_300.csv` using only the Python standard library.  The final
epoch (300) is reported; the full `results.json` trajectory is included so
that the absence of test-based checkpoint selection can be inspected.  The
trajectory also shows that epoch 300 (75.03) is not the best test row
(epoch 290 is 75.26); the final row was reported without test-based
selection.

Exact final values:

| Category | AUROC (%) |
|---|---:|
| Breakfast box | 78.0298 |
| Juice bottle | 93.6994 |
| Pushpins | 78.2609 |
| Screw bag | 61.7626 |
| Splicing connectors | 63.4143 |
| **Mean** | **75.0334** |

## Run configuration

* Test split: official `test/good` (575 images) vs
  `test/logical_anomalies` (561 images); structural anomalies are excluded.
* Final epoch: 300; evaluation every 10 epochs; no test-based early stopping
  or checkpoint selection.
* Image score: mean of the top 1% segmentation probabilities.
* Localization batch: 12 (the paper's notation is ambiguous; see
  `../docs/PAPER_SETTINGS.md`).
* DeepSIM batch: 56; 300 epochs; four GPUs.
* Seeds: 42 (training); 20260929 (qualitative generation).

Paths in `scores_epoch_300.csv` are the execution machine's absolute paths;
they are kept verbatim as run records and are not portable.
