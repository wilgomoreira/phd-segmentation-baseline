"""Metrics for multiclass semantic segmentation."""

from .metrics import (
    adaptive_ece,
    auprc_error_detection,
    auroc_error_detection,
    brier_score,
    classwise_ece,
    compute_metrics,
    ece,
    expected_aurc,
    fpr_at_tpr,
    mean_iou,
    macro_f1,
    multiclass_nll,
    overall_accuracy,
    sparsification_ause,
    SegmentationMetricAccumulator,
)

__all__ = [
    "adaptive_ece",
    "auprc_error_detection",
    "auroc_error_detection",
    "brier_score",
    "classwise_ece",
    "compute_metrics",
    "ece",
    "expected_aurc",
    "fpr_at_tpr",
    "mean_iou",
    "macro_f1",
    "multiclass_nll",
    "overall_accuracy",
    "sparsification_ause",
    "SegmentationMetricAccumulator",
]
