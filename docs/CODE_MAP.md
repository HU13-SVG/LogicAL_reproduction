# Code Map

| File | Role |
|---|---|
| `src/common.py` | Category list, test dataset, AUROC, SSIM/focal/edge losses, JND map, evaluation and CSV writer |
| `src/logical_net.py` | `ConvINAct`, `DCSA`, `LogicALNet` (OmniAL-style localization network) |
| `src/synthesis.py` | Region sampling (`choose_semantic`, `arbitrary_mask`), candidate-edge augmentation, Eq. (1) `manipulate` |
| `src/prepare_pidinet_edges.py` | Runs the official PiDiNet checkpoint over all normal images and saves first-scale edges |
| `src/train_deepsim_unified.py` | Trains one five-category DeepSIM generator from random initialisation |
| `src/train_localization_online.py` | **LAD pipeline**: online anomaly synthesis + localization training + evaluation |
| `src/generate_qualitative.py` | Optional: one anomaly per normal image, for qualitative examples |
| `tools/verify_results.py` | Recomputes the reported AUROC table from the released scores (standard library only) |
| `scripts/run_all.sh` | Runs the three stages in order |
| `docs/METRICS.md` | Defines the reported image-level metric; explains why sPRO is not reported |
| `docs/PROVENANCE.md` | Run environment, code/log boundary, redistributed and omitted assets |

## Data flow

```text
MVTec LOCO train/good
        │
        ├─ prepare_pidinet_edges.py ──► work/pidinet_edges/<category>/<stem>.png
        │                                      │
        │                                      ▼
        │                          train_deepsim_unified.py
        │                                      │
        │                                      ▼
        │                          work/deepsim_unified/latest_net_G.pth
        │                                      │
        └─ CSAD component masks ───────────────┤
                                               ▼
                              train_localization_online.py
                                ├─ online Eq. (1) synthesis
                                ├─ frozen DeepSIM generates normal/anomaly images
                                ├─ SSIM difference target
                                └─ localization training + top-1% AUROC
                                               │
                                               ▼
                               work/localization/{results.json, scores_epoch_*.csv}
```

## Reproducing the reported table

The released evidence is in `results/localization_strict/`:

* `results.json` — all 30 evaluation points (epochs 10–300), including the
  final per-category AUROC and macro mean;
* `scores_epoch_300.csv` — per-image scores and labels used for the final row;
* `summary.csv` — the paper-vs-reproduction table.

`python tools/verify_results.py` independently recomputes the AUROC values from
the CSV and checks them against the stored `results.json`.
