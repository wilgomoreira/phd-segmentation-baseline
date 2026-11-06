# PASTIS evaluation protocol

The implementation pools every valid pixel passed through
`SegmentationMetricAccumulator.update()` and computes metrics once across the
accumulated partition. Batches can have different sizes and arrive in any
order. No averaging of per-batch or per-image metrics is performed.

Inputs are logits or normalized probabilities shaped `[N, C, ...]` and integer
target masks shaped `[N, ...]`. `input_type` is explicit. The configured
`ignore_index` and each update's valid-pixel mask are combined. A valid pixel
must have a label in `[0, C)`. All metrics use the resulting same pixel set.
For PASTIS, class IDs 0–18 are scored and void label 19 is ignored; explicit
invalid pixels are excluded even if their stored target falls in 0–18.
Evaluation detaches tensors and runs under inference mode; no autograd graph is
retained.

## Shared definitions and segmentation metrics

Let `p_i` be softmax of logits or the supplied probability vector. Predicted
class is `argmax_k p_ik`; confidence is `c_i = max_k p_ik`; uncertainty is
`u_i = 1 - c_i`; and classification error is
`e_i = 1[argmax_k p_ik != y_i]`. Larger `u_i` means less confidence. All
uncertainty metrics use this same score and 0/1 error event.

### mIoU (higher is better)

One global `C x C` confusion matrix is accumulated. For each class,
`IoU_k = TP_k / (TP_k + FP_k + FN_k)`. mIoU is the unweighted mean over classes
with nonzero union. A class absent from both target and predictions has
undefined IoU and is excluded; a class absent from targets but predicted at
least once has IoU zero.

Report the complete per-class IoU vector in class-ID order (0–18), alongside
target support, predicted count, and union. Represent zero-union classes as
undefined (`null`), not zero. The mIoU denominator includes only classes with
nonzero union.

### Overall Accuracy (higher is better)

Overall Accuracy (OA) is `sum_k TP_k / n`, where `n` is the number of valid
pixels. This is pooled pixel accuracy over the complete partition, not an
average of image or batch accuracies.

### Macro F1 (higher is better; frozen)

For class `k`, `F1_k = 2 TP_k / (2 TP_k + FP_k + FN_k)`. Use the approved unweighted mean over all 19 valid semantic classes, assigning
zero when the denominator is zero. This differs from mIoU's exclusion of zero-union classes.
Retain the per-class vector and support so the aggregation is auditable. The
class inclusion and zero-denominator convention are frozen; do not silently
substitute a library default.

## Probabilistic calibration metrics

### ECE and adaptive top-label ECE (lower is better)

For top-label confidence and correctness `a_i = 1[pred_i = y_i]`,
`ECE = sum_b (n_b/n) * |mean(a_b) - mean(c_b)|`. ECE uses 15 equal-width bins
on `[0,1]`; confidence 1 belongs to the final bin. Per-bin counts, confidence
sums, and correctness sums are accumulated exactly (up to floating-point sum
rounding), so ECE does not retain per-pixel records.
For reproducible assignment use `b = min(floor(15*c_i), 14)`, so bins are
left-inclusive/right-exclusive except that the final bin includes 1.

The reported `AdaECE` is **adaptive top-label ECE**, not the canonical
all-class Adaptive Calibration Error (ACE). Canonical ACE bins probabilities
for each class and aggregates class/bin calibration gaps; this implementation
uses only the predicted top-label confidence and correctness. Nixon et al.
discuss ACE and other calibration variants; results must use the explicit name
and definition here rather than implying canonical ACE.

AdaECE uses up to 15 approximately equal-mass bins. Equal-confidence pixels
are indivisible: confidence tie groups are sorted by value, and each group is
assigned to the bin containing its midpoint rank, `floor(midpoint_rank * B/n)`
(clipped to the final bin). This rule is invariant to pixel and batch order.
It may yield fewer than 15 nonempty bins when there are fewer distinct
confidence values, and a large tie group can make bin sizes depart substantially
from equal mass. AdaECE retains confidence and correctness per valid pixel for
global sorting.

### Classwise ECE (lower is better; frozen)

For each semantic class `k`, treat `p_ik` as confidence and `1[y_i=k]` as the
binary outcome over every valid pixel. Use the same 15 equal-width bins on
`[0,1]`, with 1 included in the last bin. Define
`ECE_k = sum_b (n_kb/n) * |mean(1[y_i=k])_b - mean(p_ik)_b|`.
Assign class probability `p_ik` with `b = min(floor(15*p_ik), 14)`.
Classwise ECE is the unweighted mean of the 19 `ECE_k` values.
Unlike top-label ECE, every class probability contributes for every pixel.
Report all per-class values and bin counts. This one-vs-rest fixed-bin
definition and unweighted class aggregation are frozen. Call this Classwise
ECE; do not conflate it with top-label ECE or adaptive top-label AdaECE.

### NLL and multiclass Brier (lower is better)

Categorical NLL is `-(1/n) sum_i log(p_i,y_i)`. Logits use log-softmax. Given
probabilities, `log(0)` is negative infinity and therefore a zero target
probability contributes positive-infinite NLL. No epsilon or clipping is
applied. The multiclass Brier score is
`(1/n) sum_i sum_k (p_ik - 1[y_i=k])^2`: sum over classes, then mean over valid
pixels. NLL and Brier are accumulated as float64 running sums; the confusion
matrix uses integer counts.

## Error detection and selective prediction

The positive event is an incorrect pixel (`e_i=1`). The error-detection score
is top-label uncertainty `u_i = 1 - max_k p_ik`; larger values mean more likely
incorrect. Compute using only the same valid pixels as the segmentation
metrics. Group exact score ties before threshold calculations so results are
invariant to pixel order.

### AUROC and FPR95 for pixel error detection

The positive class is an erroneous pixel (`e_i=1`); larger `u_i` is a more
positive detection score. AUROC is trapezoidal area under the empirical ROC,
with equal-score thresholds grouped. FPR95 is the smallest empirical FPR at a
discrete threshold with TPR at least 0.95, with no interpolation. Both are
undefined if the accumulated set has no errors or no correct pixels. The
aggregate returns `None` for those metrics; standalone AUROC/FPR functions
raise `ValueError`.

### AUPRC for pixel error detection (higher is better; frozen)

Use non-interpolated average precision. Process unique
uncertainty thresholds in descending order, add all pixels tied at a threshold
as one group, and compute
`AP = sum_j (Recall_j - Recall_(j-1)) * Precision_j`, starting from recall
zero. This groups ties and does not depend on pixel order. It differs from
trapezoidal area under linearly interpolated precision-recall points. AUPRC is
undefined when there are no incorrect pixels. If every valid pixel is
incorrect, AP is 1.

### AURC and E-AURC (lower is better)

Pixels are accepted in ascending uncertainty. At coverage `k/n`, selective
risk is the 0/1 error fraction among the first `k` accepted pixels. AURC uses
the right-step empirical integral `(1/n) sum_{k=1}^n risk(k/n)`.

For tied uncertainty scores, the convention is expected risk under a uniform
random permutation within each equal-score group. For group size `m` containing
`q` errors, the expected number of errors contributed by its first `j`
accepted pixels is `j*q/m`. This makes AURC invariant to input order and
corresponds to averaging the right-step AURC over all within-tie orderings.
The optimal ranking accepts all correct pixels before errors. E-AURC is
observed AURC minus this optimal AURC, with no further normalization. The
oracle is evaluated with the same right-step convention.

AURC is undefined for an empty valid-pixel set. A dataset with no errors has
AURC zero; a dataset with all pixels in error has AURC one.

### AUSE (lower is better)

Sparsification removes pixels in descending uncertainty. At each removal
count, error is the 0/1 error fraction among retained pixels; the oracle removes
errors first. Curves are integrated by trapezoids at every integer removal
fraction `r/n` on `[0,1]`; the all-removed endpoint is zero for both curves.
Within an equal-uncertainty group, expected retained error is computed under a
uniform random permutation of that group, making AUSE invariant to input
order. At least two valid pixels are required.

## Streaming API and computational requirements

Use `SegmentationMetricAccumulator(num_classes, input_type=..., storage=...)`.
Call `update(scores, target, valid_mask=...)` for every batch, `compute()` when
the full evaluation set has been seen, and `reset()` before a new dataset.
Repeated `compute()` calls do not change state. `compute_metrics(...)` is a
one-batch convenience wrapper around this accumulator; it cannot combine
separate calls.

The confusion matrix and fixed-bin ECE require `O(C^2 + B)` state. NLL and
Brier require constant-size sums. Exact adaptive ECE and ranking metrics
require retaining one 17-byte record per valid pixel (float64 confidence,
float64 uncertainty, uint8 error): `O(P)` storage. `storage="memory"` keeps
batch record arrays in RAM; concatenating for `compute()` can temporarily
require another record-sized allocation. `storage="disk"` writes records to a
temporary memory-mapped file (about `17P` bytes on disk). Exact sorting still
requires `O(P)` index/work arrays in RAM and `O(P log P)` time; disk storage
reduces retained score storage but does not make sorting fully out-of-core.
Temporary disk space must be available, and is removed with the accumulator.
The implementation uses PyTorch and NumPy, CPU only; no GPU or dataset files
are needed for metrics computation.

For reproducibility, pool integer confusion/bin counts and float64 sums across
the complete partition. Ranking metrics must group equal scores before
integration. Store protocol version, valid-pixel count, software versions, and
per-class/per-bin diagnostics with each result.

## Validation and edge cases

Inputs must be finite. Probability vectors must lie in `[0,1]` and sum to one.
The sum tolerance is `max(1e-6, 2 * machine_epsilon(input_dtype))` for both
relative and absolute tolerance. Half and bfloat16 inputs are promoted to
float32 before normalization/logarithms; higher precision inputs retain their
dtype. Empty valid updates are rejected. Invalid labels,
shape mismatches, and nonpositive bin counts raise errors. A zero target
probability gives infinite NLL by definition. AUROC/FPR95 degeneracy and
single-pixel AUSE are handled explicitly as described above. Floating-point
accumulators can exhibit negligible summation-order differences; integer
confusion and bin counts are exact.

For classwise ECE, bins with no pixels contribute zero for that class/bin; a
class with no positive targets is still defined because its negative examples
and predicted probabilities contribute. Macro F1 follows its explicitly
approved zero-denominator rule. For error-detection metrics, AUROC is
undefined unless both correct and incorrect pixels occur; AUPRC is undefined
when no incorrect pixel occurs. Return `null` in aggregate outputs for an
undefined metric and retain the reason. One-class edge cases must not be
silently converted to a favorable score, except the defined all-error AUPRC=1
case.

## Partition use and evaluation order

- Training folds fit model weights and training-only normalization.
- The checkpoint-selection partition is reserved for implementation checks
  only at evaluation time. Training-time checkpoint selection has already
  used its prescribed mIoU; do not use these pixels to fit calibrators or make
  final fold-performance claims.
- The calibration partition is reserved for fitting approved calibrators
  after the protocol is frozen. Scores on calibrator-fit pixels are fit-set
  diagnostics, not independent generalization results.
- The test partition remains unopened until architecture/checkpoint selection,
  calibrator, metric definitions, binning, and implementation are frozen. Use
  it for final raw-probability evaluation and, if approved, a separately
  identified calibrated evaluation.

No calibration or test partition has been evaluated and no calibrator has
been fitted under this protocol.

## References and scope differences

- Guo et al., “On Calibration of Modern Neural Networks,” ICML 2017:
  <https://proceedings.mlr.press/v70/guo17a.html> (top-label fixed-bin ECE).
- Nixon et al., “Measuring Calibration in Deep Learning,” CVPR Workshops 2019:
  <https://arxiv.org/abs/1904.01685> (ACE and calibration binning choices;
  canonical ACE is all-class, unlike this project's adaptive top-label variant).
- Brier, “Verification of forecasts expressed in terms of probability,”
  Monthly Weather Review 1950:
  <https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2>.
- Geifman and El-Yaniv, “Selective Classification for Deep Neural Networks,”
  NeurIPS 2017: <https://papers.nips.cc/paper/7073-selective-classification-for-deep-neural-networks> (selective prediction context; this implementation fixes
  the discrete right-step and tie conventions above).
- Ilg et al., “Uncertainty Estimates and Multi-Hypotheses Networks for Optical
  Flow,” ECCV 2018: <https://arxiv.org/abs/1802.07095> (sparsification and
  oracle ranking; the pixelwise error and integration choices here are explicit
  adaptations for semantic segmentation).

## Implementation status and provisional future analyses

The frozen core metrics are mIoU, overall accuracy, Macro F1, per-class IoU,
top-label ECE, multiclass Brier, multiclass NLL, Classwise ECE, AUROC, AUPRC,
and AURC. Report ECE as top-label ECE and AdaECE as adaptive top-label ECE, not
canonical ACE. Report the tie-grouped Average Precision convention and the
tie-averaged AURC right-step integration.

Teacher-confidence strata and spatial boundary-distance bands remain
provisional future teacher-student analyses; they are outside the frozen core
baseline metric protocol and must not affect baseline evaluation choices.
