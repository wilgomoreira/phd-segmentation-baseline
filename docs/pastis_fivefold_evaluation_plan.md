# PASTIS five-fold baseline evaluation plan

This plan covers DeepLabV3+ ResNet-50 (scratch) and SegFormer-B1 (NVIDIA
MiT-B1 encoder, `nvlabs_reference` 256-channel decoder). It does not authorize
training, calibration fitting, test evaluation, or new experiments. The full
metric specification is maintained in
[`evaluation_protocol.md`](evaluation_protocol.md); this document sets the
partition order and the reporting record.

## Partition order and permitted uses

1. **Training folds:** fit model weights and training-only normalization.
2. **Checkpoint-selection partition:** this was the training-time mIoU
   checkpoint-selection subset. Use it now only for implementation checks
   (for example tensor shapes, metric implementation sanity checks, and
   reproducibility checks). Do not treat its scores as final fold results,
   fit a calibrator on it, or use it to revise the evaluation protocol.
3. **Calibration partition:** after the protocol is frozen, fit any approved
   calibrator here. Record the calibration method, fitted parameters,
   calibration sample count, and checkpoint hash. Do not use this partition to
   choose architectures, checkpoints, thresholds, bin counts, or metric
   definitions. Calibration-partition scores are fit-set diagnostics, not
   independent generalization results.
4. **Test partition:** leave unopened until model/checkpoint selection,
   calibrator choice, metric definitions, binning, and implementation are
   frozen. Then run one final evaluation per prespecified model and fold.
   Report raw probabilities and, if a calibrator was approved and fit, the
   separately identified calibrated predictions. Do not feed test outcomes
   back into model, calibrator, or protocol choices.

No calibration or test partition has been evaluated and no post-hoc
calibration has been applied for this plan.

## Metric groups

All segmentation and probability metrics use the same valid-pixel mask. Pool
valid pixels across a complete partition before computing metrics; never
average batch-level or image-level scores. The definitions, bin edges,
positive classes, score directions, tie behavior, and degenerate cases are in
[`evaluation_protocol.md`](evaluation_protocol.md).

### Segmentation

- mIoU and per-class IoU from one global confusion matrix.
- Overall Accuracy (OA): total correct valid pixels divided by all valid
  pixels.
- Macro F1: proposed unweighted class mean over IDs 0–18, with an F1 of zero
  when a class has a zero denominator. This choice needs approval before
  implementation; the report must include per-class support and F1 values.

For per-class IoU, retain a 19-entry vector in class-ID order. A class with no
target or predicted pixels has undefined IoU (`null`); a class with no target
pixels but at least one false positive has IoU 0. Do not encode absent classes
as zero in fold aggregation. Aggregate each class over folds where its IoU is
defined and retain that per-class `n`.

### Probabilistic calibration

- **ECE:** top-label expected calibration error with 15 fixed equal-width
  confidence bins over `[0,1]`, weighted by each bin's share of valid pixels.
- **Multiclass Brier:** sum squared probability error over all 19 classes,
  then average over valid pixels.
- **Multiclass NLL:** categorical negative log likelihood averaged over valid
  pixels, with no clipping or epsilon.
- **Classwise ECE:** proposed one-vs-rest ECE for each class using the class
  probability and binary class-membership target, 15 fixed equal-width bins;
  report the 19 class ECE values and their unweighted macro mean. This is
  distinct from top-label ECE and needs approval before implementation.

Do not fit temperature scaling or another calibrator until the calibration
partition and calibration method have been explicitly approved. If calibration
is later fitted, store its parameters and evaluate its effect only on the
reserved test partition after the complete protocol is frozen.

### Error detection and selective prediction

The positive event is an incorrect pixel prediction. Use top-label uncertainty
`u = 1 - max_k p_k`; larger values mean stronger predicted error. AUROC and
AUPRC are computed over valid pixels only. Group tied uncertainty values at
the same threshold so results do not depend on pixel order. The proposed AUPRC
is non-interpolated average precision (step-wise increase in recall times
precision at that threshold); confirm this choice before implementation,
since trapezoidal PR area is a different statistic.

AURC accepts pixels from lowest to highest uncertainty. At coverage `k/N`,
risk is the error fraction among the first `k` accepted pixels. Use the
right-step mean over all `k=1..N`; ties use expected risk under a uniform
permutation within each equal-score group. Lower AURC is better. AURC is
undefined if the partition has no valid pixels.

### Future teacher-student analysis

This block is a future analysis plan, not part of the current baseline queue.

- Report student mIoU and signed delta mIoU against the same architecture's
  hard-label-supervised student: `student_mIoU - hard_label_student_mIoU`, in
  percentage points. Both must use identical fold, test pixels, seed,
  architecture, and training settings apart from the approved supervision
  target. Approve the teacher/student training comparison before launching it.
- Stratify student metrics by the teacher's top-label confidence. Proposed
  strata are ten fixed equal-width bins over `[0,1]`, with 1 in the final bin;
  use `[b/10,(b+1)/10)` intervals and include 1 in the final bin; record each
  bin's pixel count and interval. Within each bin, compute student mIoU, OA,
  Macro F1, ECE, Brier, and NLL using the same valid pixels and definitions as
  above; report per-class IoU where defined. Teacher confidence is the
  teacher's maximum class probability, while correctness is measured against
  ground truth for the student. Confirm this binning before implementation.
  Do not choose strata after inspecting test results.
- For boundary analysis, derive semantic boundaries from valid ground-truth
  labels by marking both pixels of every 8-neighbor pair with different valid
  classes. Proposed distance strata are distance `<=1`, `>1 and <=4`, and
  `>4` pixels from that boundary set, using Euclidean pixel distance. Ignore
  void/invalid pixels as both boundary sources and scored pixels. Report
  pixel count, error rate (`1 - OA`), mIoU, ECE, Brier, and NLL in each band;
  classwise IoU may be null where a band lacks a class union. Connectivity and
  band widths need approval before implementation.

## Aggregation and reproducibility

For each architecture, evaluation variant, partition, and metric, report the
arithmetic mean and sample standard deviation (`ddof=1`) of the five fold
results. Also report `n`; for per-class IoU, `n` may be below five when a class
is absent from a fold. Fold variation is descriptive, not an independence-
based confidence interval, because the official folds are not geographically
independent.

Record architecture/profile, fold, seed, checkpoint SHA-256, dataset and
normalization config paths, valid-pixel count, metric protocol version,
software versions, and calibration state with each result. Use integer
confusion/bin counts and float64 metric sums; use deterministic score sorting
and group ties before threshold/ranking calculations. Store ECE and classwise
ECE bin counts, mean confidence, and empirical accuracy as audit data.

The machine-readable record is
[`../configs/manifests/pastis_fivefold_baselines.json`](../configs/manifests/pastis_fivefold_baselines.json).
It distinguishes checkpoint-selection implementation checks, calibration-fit
records, raw test results, calibrated test results, and future teacher-student
results. Leave all calibration/test and teacher-student values empty until
their permitted stage.

## Frozen core and provisional extensions

The core baseline definitions are frozen: Macro F1 averages all 19 valid
classes with zero for zero-denominator classes; Classwise ECE is one-vs-rest
with 15 fixed equal-width bins and an unweighted mean over classes; AUPRC is
tie-grouped Average Precision; AUROC uses incorrect pixels as positive and
`1 - max probability` as score; AURC uses low-to-high acceptance, right-step
integration, and expected risk within ties; Brier sums squared class errors
without class-count division; NLL uses stable `log_softmax` on logits and no
probability clipping. OA and per-class IoU use the definitions in the formal
protocol.

Teacher-confidence strata and spatial boundary-distance bands remain
provisional and outside the core baseline metrics. If the future
teacher-student analysis proceeds, approve its confidence binning and boundary
connectivity/distance strata before implementation. This approval is not a
prerequisite for implementing or reporting the frozen baseline metrics.
