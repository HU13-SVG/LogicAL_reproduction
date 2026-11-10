# Paper Settings, Public-Code Defaults, and Our Choices

The LogicAL paper does not publish its localization source or a complete
hyper-parameter table.  To keep the audit honest, this file separates three
kinds of settings:

* **Paper** — stated explicitly in Zhao (2024).
* **Public code** — inherited from an implementation the paper cites
  (PiDiNet, DeepSIM/pix2pixHD, OmniAL).  These are not paper claims, but they
  are reproducible from the cited public code.
* **Ours** — our choice where neither source fixes the value.  These are also
  listed in [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

Only rows marked **Paper** should be read as paper adherence.  The released
numbers were produced with the batch-size deviation noted at the end of this
file.

| Stage | Source | Setting | Implementation |
|---|---|---|---|
| Edge detector | Paper + ours | Paper: pre-trained PiDiNet; four scales extracted; first scale used for generation and localization.  Ours: the network is frozen. | `src/prepare_pidinet_edges.py` |
| Edge detector configuration | Public code | Carv4 backbone, side-output aggregation, dilated convolutions, `table5_pidinet.pth` | `models.pidinet(config="carv4", sa=True, dil=True)` |
| Edge input | Paper + ours | Paper: first-scale edge map used.  Ours: 256×256 input, ImageNet normalisation, 8-bit probability map | bilinear resize, `(x-mean)/std`, `net(x)[0]`, `uint8(255·e)` |
| Edge-to-image model | Paper + ours | Paper: one unified DeepSIM/pix2pixHD conditional GAN for the dataset.  Ours: the single model covers all five MVTec LOCO categories. | `networks.define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance", [])` |
| Generator/discriminator architecture | Public code | Global generator + one local enhancer; 64 base channels; 4 downsampling stages; 9 residual blocks; 3-layer two-scale PatchGAN; instance normalisation | `train_deepsim_unified.py`: `define_G(...)`, `define_D(6, 64, 3, ...)` |
| Generator training | Paper | 300 epochs; batch 56; 256×256 | `train_deepsim_unified.py`: `--epochs 300 --batch-size 56` |
| Generator initialisation | Public code | random initialisation | `networks.define_G(...)` without a pretrained load |
| Generator optimisation | Public code | Adam 2×10⁻⁴, β=(0.5, 0.999); constant LR for 150 epochs, then linear decay to 0 | same file, `lr = 2e-4 if epoch <= 150 else 2e-4*(300-epoch)/150` |
| Generator objective | Public code | LSGAN + 5 × discriminator feature matching + 10 × VGG-19 perceptual; VGG layer weights (1/32, 1/16, 1/8, 1/4, 1) | same file, `gan_loss`, `5*fm`, `10*perceptual` |
| TPS augmentation (DeepSIM training) | Paper | Random 3×3 control-point TPS warp applied jointly to edge map and RGB while training the generator | `Pairs` calls DeepSIM `get_params` / `get_transform` |
| TPS amount and seed | Ours | p = 0.99; seed 42 | `tps_percent=.99`, `tps_points_per_dim=3` |
| TPS augmentation (synthetic anomalies) | Paper — **not implemented** | Paper says the synthesis anomaly edge map and image "can be further augmented by applying TPS warping" (Fig. 5e–f).  The released online trainer does not apply this second TPS; it applies TPS only to the normal DeepSIM training pairs.  This is a declared deviation. | — |
| Training images | Dataset | MVTec LOCO `train/good` only; no test images; 1,778 images in five categories | `OnlineTrain` / `GeneratePairs`: `train/good` only |
| Semantic regions | Paper + ours | Paper: SAM + refinement (background removal, small-region grouping, overlap merging).  Ours: precomputed CSAD component masks as a deterministic substitute | `synthesis.choose_semantic`, `data/csad_component_masks/` |
| Arbitrary regions | Paper + ours | Paper: three regions with different aspect ratios/shapes.  Ours: rectangles/ellipses, widths/heights 20–70 px, positions in [10, 210] | `synthesis.arbitrary_mask` |
| Region post-processing | Ours | Morphological closing with kernel 3/5/7; reject regions < 64 px | `train_localization_online.py` |
| Candidate edge | Paper + ours | Paper: candidate edge from the same category, random augmentation A(E_i).  Ours: horizontal flip, rotation ±15°, scale [0.85, 1.15], translation ±16 px | `synthesis.augment_edge` |
| Edge manipulation (Eq. 1) | Paper | Remove / replace / merge, equal probability: `M+(1−M)E`; `M·A(E_i)+(1−M)E_j`; `M·(A(E_i)+E_j−1)+(1−M)E_j` | `synthesis.manipulate`; `op = random.randrange(3)` |
| Anomaly count | Ours | One online anomaly per normal image per epoch | `OnlineTrain` |
| Resolution | Paper | 256×256 synthesis | all paths |
| Qualitative generation | Ours | One anomaly per normal image; seed 20260929 + image index | `src/generate_qualitative.py` |
| Localization architecture | Paper + public code | Paper: OmniAL-style reconstruction/localization network with DCSA and DiffNeck, reconstructing RGB, JND and edge maps.  Public code: four encoder stages 48/96/192/384.  **Deviation: DiffNeck is not a separate module in the released network** (see `ASSUMPTIONS.md`) | `src/logical_net.py` |
| Localization training | Paper + ours | Paper: 300 epochs; Adam 10⁻⁴; multi-step schedule; 256×256.  Ours: batch 12 (see deviation); milestones [200, 260]; γ = 0.1 | `train_localization_online.py` |
| Localization losses | Paper + ours | Paper: MSE + SSIM on RGB/JND; annotator-robust edge loss; focal segmentation loss.  Ours: η = 0.3, β = 1.1; focal γ = 2, pos/neg 0.75/0.25; unit loss weights | `src/common.py`, `train_localization_online.py` |
| Segmentation target | Paper + ours | Paper: SSIM-based difference between the generated normal and generated anomaly images.  Ours: 11×11 SSIM window with constants 0.01²/0.03², then clamp to [0,1] | `common.ssim_ground_truth` |
| Image-level score | Ours | Mean of the top 1% segmentation probabilities | `common.evaluate` |
| Test split | Paper | Table 2 separates logical and structural anomalies; we report the logical subset (`test/good` vs `test/logical_anomalies`) | `common.LogicALTest` |
| Checkpoint selection | Ours | Final epoch 300; evaluate every 10 epochs; no test-based early stopping | `train_localization_online.py` |
| Seeds | Ours | 42 for training; 20260929 for qualitative generation | both training scripts |

## Known deviations and non-reproduced items

1. **Localization batch size.**  The paper writes "batch size of 15(12) images
   having size of 256x256(256x320) for MVTecAD, MVTecLOCO and MADsim(VisA)".
   The most natural reading is 15 images at 256×256 for MVTecAD, MVTecLOCO and
   MADsim, and 12 images at 256×320 for VisA.  The released run used **12
   images at 256×256 for MVTec LOCO** (the script default).  We retain the run
   as produced and disclose this deviation; it may contribute to the gap to
   the paper.  See [`ASSUMPTIONS.md`](ASSUMPTIONS.md).
2. **Semantic regions.**  CSAD component masks replace SAM + refinement.  The
   masks are derived annotations for normal training images and are released
   with a per-file hash manifest; they are not the authors' SAM output.
3. **DeepSIM horizontal flip.**  The LogicAL paper does not state whether the
   DeepSIM augmentation flips.  We used the DeepSIM/pix2pixHD public default
   (`no_flip=False`, joint random horizontal flip).  The flag `--no-flip`
   reproduces the no-flip variant but requires retraining.
4. **Second TPS warp.**  The paper applies TPS to the synthetic anomaly edge
   map/image (Fig. 5e–f) in addition to the DeepSIM training-time TPS.  The
   released online trainer does not implement that second warp.
5. **DiffNeck.**  The paper describes OmniAL's DiffNeck module; the released
   network uses multi-scale bilinear/concat/ConvINAct fusion and does not
   implement a separate DiffNeck.
6. **Pixel-level AU-sPRO is not reproduced.**  The archive reports image-level
   AUROC only.  The paper's pixel-level AU-sPRO requires the trained
   localization checkpoint and the official per-defect saturation
   configuration; neither is redistributed here.  We do not claim the paper's
   sPRO numbers.  See [`METRICS.md`](METRICS.md).
7. **Trained checkpoints.**  The DeepSIM generator and localization network
   are not included (they are large and were not retained); `results.json`,
   the per-image scores and the training logs are the released evidence.
