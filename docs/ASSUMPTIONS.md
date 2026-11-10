# Implementation Choices Not Fixed by the LogicAL Paper

The LogicAL authors did not release their localization implementation at the
time of our experiments.  The following choices instantiate the paper's
qualitative description.  They are recorded so that the reproduction can be
audited and so that paper-specified settings are not confused with assumptions.

## Semantic regions

* **Paper:** SAM produces rough semantic regions, followed by refinement
  (background removal, small-region grouping, overlap merging).
* **This repository:** precomputed **CSAD component masks** for MVTec LOCO
  (`data/csad_component_masks/`, 1,778 PNGs).  Each mask is a colour-coded
  component map; one non-background colour is chosen at random per sample.
  The semantic-vs-arbitrary 50/50 split and all subsequent edge operations are
  unchanged.  The official SAM refinement parameters are unavailable.
* **Consequence:** the semantic region distribution is not identical to the
  authors' SAM+refinement distribution.

## Edge representation

PiDiNet stores edge probability with 1 = edge.  DeepSIM/pix2pixHD expects
black edges on a white background, so the released code stores
`edge = 1 − PiDiNet_probability` (1 = background, 0 = edge) and applies
Eq. (1) in that representation.  Removing an edge therefore sets the selected
pixels to 1 (background), replacing copies the candidate's inverted values,
and merging uses `clip(A(E_i)+E_j−1, 0, 1)`, which is the union of edges in
the inverted representation.  Both the online trainer and the qualitative
generator use the same convention.

## Region and augmentation ranges

* Arbitrary masks: three rectangles or ellipses, widths/heights 20–70 px,
  positions in [10, 210].
* Morphological closing kernel: random choice from {3, 5, 7}.
* Minimum accepted region: 64 px; if the selected mask is empty or smaller,
  an arbitrary mask is used.
* Candidate-edge augmentation: random horizontal flip, rotation ±15°, scale
  [0.85, 1.15], translation ±16 px.  If the actual changed region is smaller
  than 32 px, the full mask is used as the target.
* If an operation changes fewer than 32 pixels, `synthesis.manipulate` keeps
  the modified edge map and uses the full selected mask as the target; the
  online trainer and the qualitative generator share this function.

These ranges instantiate the paper's qualitative description; the paper does
not publish numeric values.

## DeepSIM augmentation and horizontal flip

The LogicAL paper does not state whether its DeepSIM augmentation flips
images.  The released DeepSIM/pix2pixHD training code uses a joint random
horizontal flip by default (`no_flip=False` in `train_deepsim_unified.py`).
**The released run used that public-code default**, because it is the
augmentation implemented by the DeepSIM code the paper cites.  The no-flip
variant is available as `--no-flip`; it requires retraining the generator and
the localization network.  This is an augmentation choice, not a test-time
setting, and it does not use test data.

## TPS warp of synthetic anomalies

The paper says that, besides the DeepSIM training-time TPS augmentation, the
synthetic anomaly edge map and image "can be further augmented by applying TPS
warping" (Fig. 5e–f).  The released online trainer does **not** apply this
second TPS to the synthesized anomaly edge map or image; it applies TPS only
to the normal edge/RGB pairs while training DeepSIM.  This is a declared
deviation.  Adding the second TPS would require retraining both the generator
and the localization network, so the released numbers correspond to the
no-second-TPS run.

## Localization batch size

The paper writes "batch size of 15(12) images having size of
256x256(256x320) for MVTecAD, MVTecLOCO and MADsim(VisA)".  The most natural
reading is 15 at 256×256 for MVTecAD, MVTecLOCO and MADsim, and 12 at 256×320
for VisA.  The released run used **batch 12 at 256×256 for MVTec LOCO**, the
default of `train_localization_online.py`.  We keep the run as produced and
record the deviation here instead of silently describing 12 as a paper
setting.  The difference may contribute to the gap to the paper's 84.6%
logical AUROC.

## Localization architecture details

* **Paper:** the localization module follows OmniAL and consists of
  reconstruction and localization sub-networks with DCSA blocks and a
  **DiffNeck** module; it reconstructs the RGB image, the JND map and the edge
  map (Sec. 3.2).
* **This repository:** `src/logical_net.py` implements a four-stage encoder
  (48/96/192/384), DCSA blocks, bilinear upsampling + concatenation + ConvINAct
  fusion, a five-channel reconstruction head (RGB, JND, edge) and a
  segmentation head.  **DiffNeck is not implemented as a separate module.**
  The exact OmniAL DiffNeck definition was not available to us; the released
  network uses the multi-scale skip fusion above instead.  This is a known
  architectural deviation from the paper's description, not a claim of exact
  OmniAL reproduction.
* **JND map:** the paper cites the JND definition but does not state the
  formula.  The released code uses a 5×5 local-luminance mean and absolute
  local contrast, `clamp(0.6·local + 1.8·contrast, 0, 1)`, following the
  JND/OmniAL public implementation.  This is our implementation choice.
* Encoder widths, instance normalisation, activation and the segmentation
  head are our instantiation of the paper's description; the paper does not
  publish layer-level details.

## Localization loss and schedule details

* Edge loss: η = 0.3, β = 1.1.
* Focal segmentation loss: γ = 2, positive/negative weights 0.75/0.25.
* LR milestones: epochs 200 and 260; decay factor 0.1.
* All loss terms have unit weight (no auxiliary weighting).
* The paper specifies MSE, SSIM, the annotator-robust edge loss, a focal
  segmentation loss and a multi-step schedule, but not these numeric details.

## Image-level score

The paper derives an image-level decision from the pixel localization map but
does not specify the pooling rule.  We use the mean of the top 1% of
segmentation probabilities, which is robust to a single noisy maximum pixel.
This exact rule is used for the reported 75.03%.

## Segmentation target

The target is the SSIM-map difference between the DeepSIM image generated from
the normal edge map and the image generated from the modified edge map.  The
SSIM implementation uses an 11×11 average-pooling window and the standard
stabilising constants 0.01² and 0.03².  The paper describes an SSIM-refined
pseudo ground truth but does not publish the exact map computation.

## Pixel-level AU-sPRO

The paper also reports pixel-level AU-sPRO (Table 1 and Table 2).  This
archive does **not** reproduce that metric.  AU-sPRO requires the trained
localization checkpoint, the per-defect saturation configuration distributed
with MVTec LOCO, and the `defects_config.json` files; the checkpoint was not
retained and the reproduction does not redistribute MVTec annotations.  The
released evidence is therefore image-level AUROC only.  We do not report,
estimate or claim the paper's sPRO values.  See [`METRICS.md`](METRICS.md).

## Checkpoints and determinism

* DeepSIM and the localization network are trained from random initialisation.
* The DeepSIM generator is trained for 300 epochs; the localization network
  for 300 epochs.  Initialisation is controlled by seed 42, but exact bitwise
  reproduction is not guaranteed across GPU models, CUDA/cuDNN versions, or
  different numbers of GPUs (the original run used four A6000s with
  `DataParallel`).
* The localization network is evaluated every 10 epochs and the final epoch
  is reported.  The released `results.json` contains the full trajectory so
  that the absence of test-based checkpoint selection can be checked.
* Trained checkpoints are not included; they are large and can be regenerated
  by `scripts/run_all.sh`.  The released per-image scores, the full result
  trajectory and the training logs are the evidence for this archive; the
  score file is covered by `results/SHA256SUMS`.
* The DeepSIM log header (`DEEPSIM_V2_FROM_SCRATCH`) identifies the run-time
  version of the training script.  The released `train_deepsim_unified.py`
  is a cleaned copy (message text and portability edits); the algorithm and
  defaults are the ones documented here.  This boundary is recorded in
  [`PROVENANCE.md`](PROVENANCE.md).
