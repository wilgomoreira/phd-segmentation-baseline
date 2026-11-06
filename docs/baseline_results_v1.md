# Frozen Baseline v1.0 results

This document records the uncalibrated supervised baseline summarized in
`PhD_01_Segmentation_Baselines.pdf`. The evaluation uses frozen checkpoints and
held-out test partitions. Test results were not used for checkpoint selection,
hyperparameter tuning, or any fitting step. No post-hoc calibration or
distillation was performed.

## Datasets and evaluation units

- **PASTIS:** Sentinel-2, ten-band single-date adaptation for semantic crop
  segmentation. Report the arithmetic mean and sample standard deviation over
  the five official patch-disjoint folds. The folds remain spatially
  correlated, so they are not independent geographic replicates.
- **LandCover.ai v1:** aerial RGB land-cover segmentation. The test partition
  contains six geographically held-out scenes. Segmentation metrics are pooled
  over valid pixels; spatial dependence means pixels are not independent
  replicates.

The PASTIS and LandCover.ai protocols differ, so cross-dataset score comparisons
are descriptive only.

## Checkpoint selection

Selection used validation mIoU only, with no test data. The selected setups were
DeepLabV3+ ResNet-50 from scratch with fixed learning rate on PASTIS, and
SegFormer-B1 fine-tuned from ImageNet initialization with fixed learning rate.
Both architectures used the fine-tuned, fixed-learning-rate setup on
LandCover.ai. The factorial pilot used a single seed; the final PASTIS
five-fold result reports fold-level variability.

## Held-out segmentation and reliability metrics

Values are mean ± sample standard deviation for PASTIS and pixel-pooled values
for LandCover.ai. Lower is better for ECE, AdaECE, NLL, Brier, AURC, E-AURC,
AUSE, and FPR95; higher is better for the remaining metrics.

| Dataset | Model | mIoU | Accuracy | Macro F1 | ECE | AdaECE | NLL | Brier | AUROC | AURC | E-AURC / AUSE | FPR95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PASTIS | DeepLabV3+ ResNet-50 | 0.2746 ± 0.0024 | 0.6885 ± 0.0059 | 0.3800 ± 0.0032 | 0.0206 ± 0.0062 | 0.0209 ± 0.0060 | 0.9741 ± 0.0179 | 0.4378 ± 0.0050 | 0.7610 ± 0.0064 | 0.1465 ± 0.0029 | 0.0919 ± 0.0029 | 0.6636 ± 0.0208 |
| PASTIS | SegFormer-B1 | 0.2894 ± 0.0060 | 0.7046 ± 0.0055 | 0.3993 ± 0.0091 | 0.0951 ± 0.0216 | 0.0951 ± 0.0217 | 1.0368 ± 0.0370 | 0.4349 ± 0.0059 | 0.7638 ± 0.0034 | 0.1394 ± 0.0018 | 0.0907 ± 0.0009 | 0.6807 ± 0.0117 |
| LandCover.ai | DeepLabV3+ ResNet-50 | 0.7357 | 0.9309 | 0.8389 | 0.0284 | 0.0284 | 0.2177 | 0.1076 | 0.8877 | 0.0122 | 0.0098 | 0.4289 |
| LandCover.ai | SegFormer-B1 | 0.7386 | 0.9200 | 0.8425 | 0.0290 | 0.0289 | 0.2452 | 0.1239 | 0.8834 | 0.0151 | 0.0118 | 0.3803 |

The complete machine-readable table is
[`../results/baseline_metrics.csv`](../results/baseline_metrics.csv). LandCover.ai
pixel-pooled ranking metrics are deterministic histogram approximations using
262,144 bins; other metric aggregation details and definitions are described
in the frozen evaluation record. AUSE uses the project-specific residual
pixel-error curve definition and should be interpreted with that definition.

## Interpretation

SegFormer-B1 has the higher PASTIS segmentation scores and the higher
LandCover.ai mIoU and macro F1 in this evaluation. DeepLabV3+ has higher
LandCover.ai pixel accuracy. Reliability metrics do not follow a single model
ranking across datasets: for example, PASTIS SegFormer has higher mIoU but
substantially higher ECE than DeepLabV3+. These are descriptive results from
the specified splits, not evidence of geographic generalization beyond them.
