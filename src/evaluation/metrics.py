"""Pixel-level segmentation metrics implemented with PyTorch tensors.

Inputs use logits/probabilities shaped ``[N, C, ...]`` and integer targets
shaped ``[N, ...]``. See ``docs/evaluation_protocol.md`` for definitions and
conventions. Metrics return Python floats except ``mean_iou`` which also
returns per-class IoUs when requested.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import numpy as np
from torch import Tensor
import tempfile
from pathlib import Path

InputType = Literal["logits", "probabilities"]


def _validate_input(
    scores: Tensor,
    target: Tensor,
    num_classes: int,
    input_type: InputType,
    ignore_index: int | None,
    valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(scores, Tensor) or not isinstance(target, Tensor):
        raise TypeError("scores and target must be torch.Tensor instances")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be an integer")
    if scores.ndim < 3:
        raise ValueError("scores must have shape [N, C, ...] with at least one spatial dimension")
    if target.ndim != scores.ndim - 1 or scores.shape[0] != target.shape[0]:
        raise ValueError("target must have shape [N, ...] matching scores [N, C, ...]")
    if scores.shape[1] != num_classes or num_classes < 2:
        raise ValueError("num_classes must be >= 2 and match scores.shape[1]")
    if tuple(scores.shape[2:]) != tuple(target.shape[1:]):
        raise ValueError("spatial dimensions of scores and target must match")
    if input_type not in ("logits", "probabilities"):
        raise ValueError("input_type must be 'logits' or 'probabilities'")
    if not scores.is_floating_point():
        raise TypeError("scores must have floating point dtype")
    if target.is_floating_point() or target.is_complex():
        raise TypeError("target must contain integer class indices")

    target = target.to(device=scores.device, dtype=torch.long)
    if valid_mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        if not isinstance(valid_mask, Tensor):
            raise TypeError("valid_mask must be a torch.Tensor")
        if tuple(valid_mask.shape) != tuple(target.shape):
            raise ValueError("valid_mask must have the same shape as target")
        valid = valid_mask.to(device=scores.device, dtype=torch.bool)
    if ignore_index is not None:
        valid &= target != ignore_index
    if not bool(valid.any()):
        raise ValueError("no valid pixels remain after applying masks")
    invalid_labels = valid & ((target < 0) | (target >= num_classes))
    if bool(invalid_labels.any()):
        raise ValueError("valid target labels must be in [0, num_classes)")

    flat_scores = scores.movedim(1, -1).reshape(-1, num_classes)[valid.reshape(-1)]
    flat_target = target.reshape(-1)[valid.reshape(-1)]
    if not bool(torch.isfinite(flat_scores).all()):
        raise ValueError("valid scores must be finite")
    work_dtype = torch.float32 if flat_scores.dtype in (torch.float16, torch.bfloat16) else flat_scores.dtype
    flat_scores = flat_scores.to(work_dtype)
    if input_type == "logits":
        probs = torch.softmax(flat_scores, dim=1)
        log_probs = torch.log_softmax(flat_scores, dim=1)
    else:
        if bool((flat_scores < 0).any()) or bool((flat_scores > 1).any()):
            raise ValueError("probabilities must be in [0, 1]")
        tolerance = max(1e-6, 2 * torch.finfo(scores.dtype).eps)
        if not torch.allclose(flat_scores.sum(dim=1), torch.ones(flat_scores.shape[0], device=scores.device, dtype=work_dtype), rtol=tolerance, atol=tolerance):
            raise ValueError("each probability vector must sum to 1")
        probs = flat_scores
        log_probs = torch.log(probs)
    return probs, log_probs, flat_target


def _as_flat_probabilities(
    scores: Tensor,
    target: Tensor,
    num_classes: int,
    input_type: InputType,
    ignore_index: int | None,
    valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    return _validate_input(scores, target, num_classes, input_type, ignore_index, valid_mask)


def _bin_error(confidence: Tensor, correct: Tensor, bin_count: int, adaptive: bool) -> float:
    if not isinstance(bin_count, int) or isinstance(bin_count, bool) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")
    if not isinstance(confidence, Tensor) or not isinstance(correct, Tensor):
        raise TypeError("confidence and correct must be torch.Tensor instances")
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have identical shapes")
    if correct.dtype != torch.bool:
        raise TypeError("correct must be a boolean tensor")
    confidence = confidence.reshape(-1)
    correct = correct.to(device=confidence.device, dtype=torch.bool).reshape(-1)
    if not confidence.is_floating_point() or not bool(torch.isfinite(confidence).all()):
        raise ValueError("confidence must contain finite floating point values")
    if bool(((confidence < 0) | (confidence > 1)).any()):
        raise ValueError("confidence values must be in [0, 1]")
    n = confidence.numel()
    if n == 0:
        raise ValueError("cannot calculate calibration on an empty sample")
    if adaptive:
        # Keep equal-confidence observations together. Assign each confidence
        # tie block by its midpoint rank, so permuting pixels cannot alter bins.
        ordered = torch.argsort(confidence, stable=True)
        _, group_ids, group_sizes = torch.unique_consecutive(
            confidence[ordered], return_inverse=True, return_counts=True
        )
        starts = torch.cumsum(group_sizes, 0) - group_sizes
        midpoint = starts + group_sizes.to(starts.dtype) / 2
        group_bins = torch.clamp((midpoint * bin_count / n).to(torch.long), max=bin_count - 1)
        item_bins = group_bins[group_ids]
        sorted_correct = correct[ordered].to(confidence.dtype)
        counts = torch.bincount(item_bins, minlength=bin_count).to(confidence.dtype)
        confidence_sums = torch.bincount(item_bins, weights=confidence[ordered], minlength=bin_count)
        correct_sums = torch.bincount(item_bins, weights=sorted_correct, minlength=bin_count)
        nonempty = counts > 0
        gaps = torch.abs(correct_sums[nonempty] / counts[nonempty] - confidence_sums[nonempty] / counts[nonempty])
        return float((gaps * counts[nonempty] / n).sum().item())
    else:
        # Equal-width bins on [0, 1]; confidence == 1 belongs to the final bin.
        ids = torch.clamp((confidence * bin_count).to(torch.long), max=bin_count - 1)
        chunks = tuple(torch.nonzero(ids == i, as_tuple=False).flatten() for i in range(bin_count))
    error = confidence.new_zeros(())
    for inds in chunks:
        if inds.numel():
            gap = torch.abs(correct[inds].to(confidence.dtype).mean() - confidence[inds].mean())
            error += (inds.numel() / n) * gap
    return float(error.item())


def ece(confidence: Tensor, correct: Tensor, bin_count: int = 15) -> float:
    """Top-label expected calibration error using equal-width confidence bins."""
    return _bin_error(confidence, correct, bin_count, adaptive=False)


def _classwise_ece_from_flat_probs(probs: Tensor, target: Tensor, bin_count: int) -> list[float]:
    """One-vs-rest fixed-bin ECE for every class in a flat probability matrix."""
    if not isinstance(bin_count, int) or isinstance(bin_count, bool) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")
    if probs.ndim != 2 or target.ndim != 1 or probs.shape[0] != target.numel():
        raise ValueError("probs must be [N, C] and target must be [N]")
    if probs.shape[0] == 0:
        raise ValueError("cannot calculate classwise ECE on an empty sample")
    n, classes = probs.shape
    values: list[float] = []
    for class_id in range(classes):
        confidence = probs[:, class_id].to(torch.float64)
        ids = torch.clamp((confidence * bin_count).to(torch.long), max=bin_count - 1)
        counts = torch.bincount(ids, minlength=bin_count).to(torch.float64)
        confidence_sums = torch.bincount(ids, weights=confidence, minlength=bin_count)
        positive_sums = torch.bincount(
            ids, weights=(target == class_id).to(torch.float64), minlength=bin_count,
        )
        nonempty = counts > 0
        gaps = (positive_sums[nonempty] / counts[nonempty]
                - confidence_sums[nonempty] / counts[nonempty]).abs()
        values.append(float((gaps * counts[nonempty] / n).sum().item()))
    return values


def classwise_ece(
    scores: Tensor,
    target: Tensor,
    num_classes: int,
    *,
    input_type: InputType,
    ignore_index: int | None = None,
    valid_mask: Tensor | None = None,
    bin_count: int = 15,
    return_per_class: bool = False,
) -> float | tuple[float, list[float]]:
    """Macro one-vs-rest ECE over all classes using fixed equal-width bins."""
    probs, _, truth = _validate_input(
        scores, target, num_classes, input_type, ignore_index, valid_mask,
    )
    per_class = _classwise_ece_from_flat_probs(probs, truth, bin_count)
    result = sum(per_class) / num_classes
    return (result, per_class) if return_per_class else result


def adaptive_ece(confidence: Tensor, correct: Tensor, bin_count: int = 15) -> float:
    """Top-label adaptive ECE using equal-mass bins, merging small samples."""
    return _bin_error(confidence, correct, bin_count, adaptive=True)


def mean_iou(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    valid_mask: Tensor | None = None,
    *,
    return_per_class: bool = False,
) -> float | tuple[float, list[float | None]]:
    """Mean IoU; classes with zero union are excluded from the macro mean."""
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch.Tensor instances")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if prediction.is_floating_point() or prediction.is_complex():
        raise TypeError("prediction must contain integer class indices")
    if target.is_floating_point() or target.is_complex():
        raise TypeError("target must contain integer class indices")
    if valid_mask is not None and not isinstance(valid_mask, Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    pred = prediction.to(dtype=torch.long)
    truth = target.to(device=pred.device, dtype=torch.long)
    valid = torch.ones_like(truth, dtype=torch.bool)
    if valid_mask is not None:
        if valid_mask.shape != truth.shape:
            raise ValueError("valid_mask must have the same shape as target")
        valid &= valid_mask.to(device=pred.device, dtype=torch.bool)
    if ignore_index is not None:
        valid &= truth != ignore_index
    if not bool(valid.any()):
        raise ValueError("no valid pixels remain after applying masks")
    p, t = pred[valid], truth[valid]
    if bool(((p < 0) | (p >= num_classes) | (t < 0) | (t >= num_classes)).any()):
        raise ValueError("valid prediction and target labels must be in [0, num_classes)")
    cm = torch.bincount(t * num_classes + p, minlength=num_classes**2).reshape(num_classes, num_classes)
    tp = cm.diag().to(torch.float64)
    union = cm.sum(0).to(torch.float64) + cm.sum(1).to(torch.float64) - tp
    present = union > 0
    if not bool(present.any()):
        raise ValueError("mIoU is undefined because every class has zero union")
    class_values = torch.full((num_classes,), float("nan"), dtype=torch.float64, device=cm.device)
    class_values[present] = tp[present] / union[present]
    result = float(class_values[present].mean().item())
    if return_per_class:
        return result, [float(x) if bool(present[i]) else None for i, x in enumerate(class_values)]
    return result


def _confusion_for_labels(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int | None,
    valid_mask: Tensor | None,
) -> Tensor:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be torch.Tensor instances")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2")
    if prediction.is_floating_point() or prediction.is_complex():
        raise TypeError("prediction must contain integer class indices")
    if target.is_floating_point() or target.is_complex():
        raise TypeError("target must contain integer class indices")
    pred = prediction.to(torch.long)
    truth = target.to(device=pred.device, dtype=torch.long)
    valid = torch.ones_like(truth, dtype=torch.bool)
    if valid_mask is not None:
        if not isinstance(valid_mask, Tensor) or valid_mask.shape != truth.shape:
            raise ValueError("valid_mask must be a tensor with the same shape as target")
        valid &= valid_mask.to(device=pred.device, dtype=torch.bool)
    if ignore_index is not None:
        valid &= truth != ignore_index
    if not bool(valid.any()):
        raise ValueError("no valid pixels remain after applying masks")
    pred, truth = pred[valid], truth[valid]
    if bool(((pred < 0) | (pred >= num_classes) | (truth < 0) | (truth >= num_classes)).any()):
        raise ValueError("valid prediction and target labels must be in [0, num_classes)")
    return torch.bincount(
        truth * num_classes + pred, minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def overall_accuracy(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    valid_mask: Tensor | None = None,
) -> float:
    """Pooled pixel accuracy over valid, non-ignored pixels."""
    cm = _confusion_for_labels(prediction, target, num_classes, ignore_index, valid_mask)
    return float(cm.diag().sum().to(torch.float64).div(cm.sum()).item())


def macro_f1(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    valid_mask: Tensor | None = None,
    *,
    return_per_class: bool = False,
) -> float | tuple[float, list[float]]:
    """Unweighted F1 over every class; zero-denominator classes receive zero."""
    cm = _confusion_for_labels(prediction, target, num_classes, ignore_index, valid_mask).to(torch.float64)
    tp = cm.diag()
    denominator = 2 * tp + cm.sum(0) - tp + cm.sum(1) - tp
    per_class = torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 0.0)
    result = float(per_class.mean().item())
    values = [float(value) for value in per_class]
    return (result, values) if return_per_class else result


def _prediction_errors(probs: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    confidence, prediction = probs.max(dim=1)
    error = prediction != target
    uncertainty = 1.0 - confidence
    return uncertainty, error


def _binary_roc(scores: Tensor, positive: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(scores, Tensor) or not isinstance(positive, Tensor):
        raise TypeError("scores and positive labels must be torch.Tensor instances")
    if scores.ndim != positive.ndim or scores.shape != positive.shape:
        raise ValueError("scores and positive labels must have identical shapes")
    if positive.dtype != torch.bool:
        raise TypeError("positive labels must be boolean")
    if scores.device != positive.device:
        raise ValueError("scores and positive labels must be on the same device")
    if not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite floating point values")
    if bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError("uncertainty scores must be in [0, 1]")
    positive = positive.to(torch.bool)
    positives = int(positive.sum().item())
    negatives = positive.numel() - positives
    if positives == 0 or negatives == 0:
        raise ValueError("binary ranking metric is undefined unless both positive and negative cases occur")
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_pos = positive[order].to(torch.float64)
    tps = torch.cat((torch.zeros(1, dtype=torch.float64, device=scores.device), sorted_pos.cumsum(0)))
    fps = torch.cat((torch.zeros(1, dtype=torch.float64, device=scores.device), (~positive[order]).to(torch.float64).cumsum(0)))
    # Collapse equal-score thresholds so the ROC is invariant to within-tie ordering.
    sorted_scores = scores[order]
    ends = torch.ones_like(sorted_scores, dtype=torch.bool)
    if sorted_scores.numel() > 1:
        ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    idx = torch.nonzero(ends, as_tuple=False).flatten() + 1
    idx = torch.cat((torch.zeros(1, dtype=torch.long, device=scores.device), idx))
    tpr = tps[idx] / positives
    fpr = fps[idx] / negatives
    return fpr, tpr


def auroc_error_detection(uncertainty: Tensor, error: Tensor) -> float:
    """AUROC for error detection; error is positive and larger uncertainty is positive."""
    fpr, tpr = _binary_roc(uncertainty.reshape(-1), error.reshape(-1))
    return float(torch.trapezoid(tpr, fpr).item())


def auprc_error_detection(uncertainty: Tensor, error: Tensor) -> float:
    """Tie-grouped non-interpolated average precision; incorrect pixels are positive."""
    if not isinstance(uncertainty, Tensor) or not isinstance(error, Tensor):
        raise TypeError("uncertainty and error must be torch.Tensor instances")
    if uncertainty.shape != error.shape or uncertainty.device != error.device:
        raise ValueError("uncertainty and error must have identical shapes and devices")
    if error.dtype != torch.bool:
        raise TypeError("error must be a boolean tensor")
    scores = uncertainty.reshape(-1)
    positive = error.reshape(-1)
    if scores.numel() == 0:
        raise ValueError("AUPRC is undefined for an empty sample")
    if not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("uncertainty must contain finite floating point values")
    if bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError("uncertainty values must be in [0, 1]")
    positives = int(positive.sum().item())
    if positives == 0:
        raise ValueError("AUPRC is undefined when there are no incorrect pixels")
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_scores = scores[order]
    ordered_positive = positive[order].to(torch.float64)
    group_ends = torch.ones_like(ordered_scores, dtype=torch.bool)
    if ordered_scores.numel() > 1:
        group_ends[:-1] = ordered_scores[:-1] != ordered_scores[1:]
    ends = torch.nonzero(group_ends, as_tuple=False).flatten() + 1
    cumulative_tp = torch.cat((
        torch.zeros(1, dtype=torch.float64, device=scores.device),
        ordered_positive.cumsum(0)[ends - 1],
    ))
    cumulative_fp = torch.cat((
        torch.zeros(1, dtype=torch.float64, device=scores.device),
        (1 - ordered_positive).cumsum(0)[ends - 1],
    ))
    recall = cumulative_tp / positives
    precision = cumulative_tp / (cumulative_tp + cumulative_fp).clamp_min(1)
    return float(((recall[1:] - recall[:-1]) * precision[1:]).sum().item())


def fpr_at_tpr(uncertainty: Tensor, error: Tensor, tpr_target: float = 0.95) -> float:
    """Smallest empirical FPR at or above requested TPR; errors are positive."""
    if not 0 < tpr_target <= 1:
        raise ValueError("tpr_target must be in (0, 1]")
    fpr, tpr = _binary_roc(uncertainty.reshape(-1), error.reshape(-1))
    qualifying = torch.nonzero(tpr >= tpr_target, as_tuple=False).flatten()
    return float(fpr[qualifying[0]].item())


def _tie_averaged_aurc(uncertainty: Tensor, error: Tensor) -> Tensor:
    """Expected right-step AURC over uniform random permutations within ties."""
    order = torch.argsort(uncertainty, stable=True)
    u = uncertainty[order]
    e = error[order].to(torch.float64)
    n = e.numel()
    _, group_ids, counts = torch.unique_consecutive(u, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    prior_n = starts[group_ids].to(torch.float64)
    within = torch.arange(n, device=e.device, dtype=torch.float64) - prior_n + 1
    group_errors = torch.bincount(group_ids, weights=e, minlength=counts.numel())
    prior_errors = torch.cumsum(group_errors, 0) - group_errors
    expected_cumulative_errors = prior_errors[group_ids] + within * group_errors[group_ids] / counts[group_ids]
    return (expected_cumulative_errors / (prior_n + within)).mean()


def expected_aurc(uncertainty: Tensor, error: Tensor) -> tuple[float, float]:
    """Return (AURC, E-AURC) via ranked pixel acceptance and oracle ranking.

    AURC uses the empirical right-step sum of selective risks at coverages k/n.
    E-AURC subtracts the best attainable AURC for the same error labels.
    """
    if not isinstance(uncertainty, Tensor) or not isinstance(error, Tensor):
        raise TypeError("uncertainty and error must be torch.Tensor instances")
    if error.dtype != torch.bool:
        raise TypeError("error must be a boolean tensor")
    if uncertainty.device != error.device:
        raise ValueError("uncertainty and error must be on the same device")
    u, e = uncertainty.reshape(-1), error.reshape(-1)
    if u.numel() == 0 or u.numel() != e.numel():
        raise ValueError("uncertainty and error must be non-empty and have the same number of pixels")
    if not u.is_floating_point() or not bool(torch.isfinite(u).all()):
        raise ValueError("uncertainty must contain finite floating point values")
    if bool(((u < 0) | (u > 1)).any()):
        raise ValueError("uncertainty values must be in [0, 1]")
    aurc = _tie_averaged_aurc(u, e)
    ordered_error = torch.sort(e.to(torch.float64)).values
    oracle_risk = ordered_error.cumsum(0) / torch.arange(1, e.numel() + 1, dtype=torch.float64, device=e.device)
    excess = aurc - oracle_risk.mean()
    return float(aurc.item()), float(excess.item())


def sparsification_ause(uncertainty: Tensor, error: Tensor) -> float:
    """AUSE: normalized area between residual error curves after removal.

    Removes highest-uncertainty pixels first; oracle removes errors first.
    Error is the fraction of erroneous pixels among retained pixels. Curves are
    evaluated at every integer removal count and integrated by trapezoids.
    """
    if not isinstance(uncertainty, Tensor) or not isinstance(error, Tensor):
        raise TypeError("uncertainty and error must be torch.Tensor instances")
    if error.dtype != torch.bool:
        raise TypeError("error must be a boolean tensor")
    if uncertainty.device != error.device:
        raise ValueError("uncertainty and error must be on the same device")
    u, e = uncertainty.reshape(-1), error.reshape(-1)
    if u.numel() == 0 or u.numel() != e.numel():
        raise ValueError("uncertainty and error must be non-empty and have the same number of pixels")
    if not u.is_floating_point() or not bool(torch.isfinite(u).all()):
        raise ValueError("uncertainty must contain finite floating point values")
    if bool(((u < 0) | (u > 1)).any()):
        raise ValueError("uncertainty values must be in [0, 1]")
    n = e.numel()
    if n == 1:
        raise ValueError("AUSE requires at least two pixels to define an area")
    order = torch.argsort(u, descending=True, stable=True)
    sorted_u, sorted_e = u[order], e[order].to(torch.float64)
    _, group_ids, counts = torch.unique_consecutive(sorted_u, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    prior_n = starts[group_ids].to(torch.float64)
    within = torch.arange(n, device=e.device, dtype=torch.float64) - prior_n + 1
    group_errors = torch.bincount(group_ids, weights=sorted_e, minlength=counts.numel())
    prior_errors = torch.cumsum(group_errors, 0) - group_errors
    total_errors = e.to(torch.float64).sum()
    remaining = n - prior_n - within
    retained_errors = total_errors - prior_errors[group_ids] - within * group_errors[group_ids] / counts[group_ids]
    after_each_removal = torch.where(remaining > 0, retained_errors / torch.clamp(remaining, min=1), 0.0)
    observed_curve = torch.cat((total_errors.reshape(1) / n, after_each_removal))
    removals = torch.arange(n, dtype=torch.float64, device=e.device)
    oracle_curve = torch.cat((torch.clamp(total_errors - removals, min=0) / (n - removals), torch.zeros(1, dtype=torch.float64, device=e.device)))
    gap = observed_curve - oracle_curve
    return float((torch.trapezoid(gap, dx=1.0 / n)).item())


def multiclass_nll(
    scores: Tensor,
    target: Tensor,
    *,
    input_type: InputType = "logits",
) -> float:
    """Mean categorical NLL; logits use stable log_softmax and probabilities are unclipped."""
    if scores.ndim != 2 or target.ndim != 1 or scores.shape[0] != target.numel():
        raise ValueError("scores must be [N, C] and target must be [N]")
    if target.is_floating_point() or target.is_complex():
        raise TypeError("target must contain integer class indices")
    if input_type == "logits":
        if not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
            raise ValueError("logits must be finite floating point values")
        log_probs = torch.log_softmax(scores, dim=1)
    elif input_type == "probabilities":
        if not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
            raise ValueError("probabilities must be finite floating point values")
        if bool(((scores < 0) | (scores > 1)).any()):
            raise ValueError("probabilities must be in [0, 1]")
        tolerance = max(1e-6, 2 * torch.finfo(scores.dtype).eps)
        if not torch.allclose(
            scores.sum(1), torch.ones(scores.shape[0], device=scores.device, dtype=scores.dtype),
            rtol=tolerance, atol=tolerance,
        ):
            raise ValueError("each probability vector must sum to 1")
        log_probs = torch.log(scores)
    else:
        raise ValueError("input_type must be 'logits' or 'probabilities'")
    target = target.to(device=scores.device, dtype=torch.long)
    if bool(((target < 0) | (target >= scores.shape[1])).any()):
        raise ValueError("target labels must be in [0, num_classes)")
    return float((-log_probs.gather(1, target[:, None]).to(torch.float64).mean()).item())


def brier_score(probs: Tensor, target: Tensor) -> float:
    """Mean multiclass Brier score (sum over classes, then mean over pixels)."""
    one_hot = torch.nn.functional.one_hot(target.to(torch.long), num_classes=probs.shape[1]).to(torch.float64)
    return float(((probs.to(torch.float64) - one_hot).square().sum(dim=1)).mean().item())


_RECORD_DTYPE = np.dtype([("confidence", "<f8"), ("uncertainty", "<f8"), ("error", "u1")])


class SegmentationMetricAccumulator:
    """Exact dataset-level pixel metrics with RAM or disk-backed rank storage.

    Confusion matrix, fixed-bin ECE, NLL, and Brier use constant-size running
    sufficient statistics. Exact adaptive ECE and ranking metrics retain one
    compact record per valid pixel. ``storage='disk'`` writes these records to
    a temporary binary memmap; sorting still needs O(P) indices in RAM.
    Updates detach inputs and run without autograd tracking.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        input_type: InputType,
        ignore_index: int | None = None,
        ece_bins: int = 15,
        adaptive_ece_bins: int = 15,
        storage: Literal["memory", "disk"] = "memory",
        temp_dir: str | None = None,
    ) -> None:
        if not isinstance(num_classes, int) or isinstance(num_classes, bool):
            raise TypeError("num_classes must be an integer")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if input_type not in ("logits", "probabilities"):
            raise ValueError("input_type must be 'logits' or 'probabilities'")
        if any(not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0 for bins in (ece_bins, adaptive_ece_bins)):
            raise ValueError("calibration bin counts must be positive")
        if storage not in ("memory", "disk"):
            raise ValueError("storage must be 'memory' or 'disk'")
        self.num_classes = num_classes
        self.input_type = input_type
        self.ignore_index = ignore_index
        self.ece_bins = ece_bins
        self.adaptive_ece_bins = adaptive_ece_bins
        self.storage = storage
        self._temp = tempfile.TemporaryDirectory(dir=temp_dir) if storage == "disk" else None
        self._record_path = Path(self._temp.name) / "pixel_records.bin" if self._temp else None
        self.reset()

    def reset(self) -> None:
        """Discard accumulated data and return the accumulator to its empty state."""
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        self.ece_count = np.zeros(self.ece_bins, dtype=np.int64)
        self.ece_confidence_sum = np.zeros(self.ece_bins, dtype=np.float64)
        self.ece_correct_sum = np.zeros(self.ece_bins, dtype=np.float64)
        self.class_ece_count = np.zeros((self.num_classes, self.ece_bins), dtype=np.int64)
        self.class_ece_probability_sum = np.zeros((self.num_classes, self.ece_bins), dtype=np.float64)
        self.class_ece_positive_sum = np.zeros((self.num_classes, self.ece_bins), dtype=np.float64)
        self.nll_sum = 0.0
        self.brier_sum = 0.0
        self.pixel_count = 0
        self._chunks: list[np.ndarray] = []
        if self._record_path is not None:
            self._record_path.write_bytes(b"")

    def update(
        self,
        scores: Tensor,
        target: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> None:
        """Add one batch; all batch pixels contribute to the same dataset totals."""
        if not isinstance(scores, Tensor) or not isinstance(target, Tensor):
            raise TypeError("scores and target must be torch.Tensor instances")
        if valid_mask is not None and not isinstance(valid_mask, Tensor):
            raise TypeError("valid_mask must be a torch.Tensor")
        with torch.inference_mode():
            probs, log_probs, truth = _validate_input(
                scores.detach(), target.detach(), self.num_classes, self.input_type,
                self.ignore_index, valid_mask.detach() if valid_mask is not None else None,
            )
            prediction = probs.argmax(1)
            confidence = probs.max(1).values.to(torch.float64)
            uncertainty = 1.0 - confidence
            error = prediction != truth
            self.confusion += torch.bincount(
                (truth * self.num_classes + prediction).cpu(),
                minlength=self.num_classes**2,
            ).reshape(self.num_classes, self.num_classes)
            self.nll_sum += float((-log_probs.gather(1, truth[:, None]).to(torch.float64).sum()).item())
            one_hot = torch.nn.functional.one_hot(truth, num_classes=self.num_classes).to(torch.float64)
            self.brier_sum += float((probs.to(torch.float64) - one_hot).square().sum().item())
            conf_cpu = confidence.cpu().numpy()
            error_cpu = error.cpu().numpy()
            bin_ids = np.minimum((conf_cpu * self.ece_bins).astype(np.int64), self.ece_bins - 1)
            self.ece_count += np.bincount(bin_ids, minlength=self.ece_bins)
            self.ece_confidence_sum += np.bincount(bin_ids, weights=conf_cpu, minlength=self.ece_bins)
            self.ece_correct_sum += np.bincount(bin_ids, weights=(~error_cpu).astype(np.float64), minlength=self.ece_bins)
            probability_cpu = probs.cpu().numpy()
            truth_cpu = truth.cpu().numpy()
            for class_id in range(self.num_classes):
                class_probability = probability_cpu[:, class_id]
                class_bins = np.minimum(
                    (class_probability * self.ece_bins).astype(np.int64), self.ece_bins - 1,
                )
                self.class_ece_count[class_id] += np.bincount(
                    class_bins, minlength=self.ece_bins,
                )
                self.class_ece_probability_sum[class_id] += np.bincount(
                    class_bins, weights=class_probability.astype(np.float64), minlength=self.ece_bins,
                )
                self.class_ece_positive_sum[class_id] += np.bincount(
                    class_bins, weights=(truth_cpu == class_id).astype(np.float64), minlength=self.ece_bins,
                )
            records = np.empty(conf_cpu.size, dtype=_RECORD_DTYPE)
            records["confidence"] = conf_cpu
            records["uncertainty"] = uncertainty.cpu().numpy()
            records["error"] = error_cpu.astype(np.uint8)
            if self.storage == "memory":
                self._chunks.append(records)
            else:
                with self._record_path.open("ab") as output:
                    records.tofile(output)
            self.pixel_count += int(truth.numel())

    def _records(self) -> np.ndarray:
        if self.storage == "memory":
            return np.concatenate(self._chunks) if self._chunks else np.empty(0, dtype=_RECORD_DTYPE)
        if self.pixel_count == 0:
            return np.empty(0, dtype=_RECORD_DTYPE)
        return np.memmap(self._record_path, dtype=_RECORD_DTYPE, mode="r", shape=(self.pixel_count,))

    def compute(self) -> dict[str, Any]:
        """Return current global metrics without changing accumulated state."""
        n = self.pixel_count
        if n == 0:
            raise ValueError("cannot compute metrics before adding valid pixels")
        cm = self.confusion.to(torch.float64)
        tp = cm.diag()
        target_count = cm.sum(1)
        predicted_count = cm.sum(0)
        union = cm.sum(0) + cm.sum(1) - tp
        present = union > 0
        miou = float((tp[present] / union[present]).mean().item())
        per_class_iou = [float(tp[i] / union[i]) if bool(present[i]) else None
                         for i in range(self.num_classes)]
        undefined_reasons: dict[str, str] = {
            f"per_class_IoU_class_{i}": "class has zero target and prediction union"
            for i in range(self.num_classes) if not bool(present[i])
        }
        f1_denominator = target_count + predicted_count
        per_class_f1 = torch.where(
            f1_denominator > 0, 2 * tp / f1_denominator.clamp_min(1), 0.0,
        )
        macro_f1_value = float(per_class_f1.mean().item())
        accuracy = float(tp.sum().div(n).item())
        ece_gaps = np.abs(
            self.ece_correct_sum[self.ece_count > 0] / self.ece_count[self.ece_count > 0]
            - self.ece_confidence_sum[self.ece_count > 0] / self.ece_count[self.ece_count > 0]
        )
        ece_value = float(np.sum(ece_gaps * self.ece_count[self.ece_count > 0]) / n)
        ece_bin_diagnostics = []
        for bin_id, count in enumerate(self.ece_count):
            if count:
                mean_confidence = float(self.ece_confidence_sum[bin_id] / count)
                mean_accuracy = float(self.ece_correct_sum[bin_id] / count)
                gap = abs(mean_accuracy - mean_confidence)
                contribution = float(count / n * gap)
            else:
                mean_confidence = mean_accuracy = gap = None
                contribution = 0.0
            ece_bin_diagnostics.append({
                "bin": bin_id,
                "lower": bin_id / self.ece_bins,
                "upper": (bin_id + 1) / self.ece_bins,
                "count": int(count),
                "mean_confidence": mean_confidence,
                "mean_accuracy": mean_accuracy,
                "absolute_gap": gap,
                "ece_contribution": contribution,
            })
        classwise_values: list[float] = []
        classwise_bin_diagnostics: list[list[dict[str, Any]]] = []
        for class_id in range(self.num_classes):
            counts = self.class_ece_count[class_id]
            nonempty = counts > 0
            gaps = np.abs(
                self.class_ece_positive_sum[class_id, nonempty] / counts[nonempty]
                - self.class_ece_probability_sum[class_id, nonempty] / counts[nonempty]
            )
            classwise_values.append(float(np.sum(gaps * counts[nonempty] / n)))
            class_bins: list[dict[str, Any]] = []
            for bin_id, count in enumerate(counts):
                if count:
                    mean_probability = float(self.class_ece_probability_sum[class_id, bin_id] / count)
                    empirical_rate = float(self.class_ece_positive_sum[class_id, bin_id] / count)
                    gap = abs(empirical_rate - mean_probability)
                    contribution = float(count / n * gap)
                else:
                    mean_probability = empirical_rate = gap = None
                    contribution = 0.0
                class_bins.append({
                    "bin": bin_id,
                    "lower": bin_id / self.ece_bins,
                    "upper": (bin_id + 1) / self.ece_bins,
                    "count": int(count),
                    "mean_probability": mean_probability,
                    "empirical_rate": empirical_rate,
                    "absolute_gap": gap,
                    "ece_contribution": contribution,
                })
            classwise_bin_diagnostics.append(class_bins)
        classwise_ece_value = float(np.mean(classwise_values))
        records = self._records()
        conf = np.asarray(records["confidence"])
        unc = np.asarray(records["uncertainty"])
        err = np.asarray(records["error"], dtype=bool)
        adaece = _adaptive_ece_numpy(conf, ~err, self.adaptive_ece_bins)
        if err.any() and (~err).any():
            auroc, fpr95 = _roc_numpy(unc, err)
        else:
            auroc = fpr95 = None
            undefined_reasons["AUROC"] = "requires at least one incorrect and one correct valid pixel"
            undefined_reasons["FPR95"] = "requires at least one incorrect and one correct valid pixel"
        auprc = _auprc_numpy(unc, err) if err.any() else None
        if auprc is None:
            undefined_reasons["AUPRC"] = "requires at least one incorrect valid pixel"
        aurc = _aurc_numpy(unc, err)
        eaurc = aurc - _aurc_numpy(err.astype(np.float64), err)
        if n >= 2:
            ause = _ause_numpy(unc, err)
        else:
            ause = None
            undefined_reasons["AUSE"] = "requires at least two valid pixels"
        return {
            "mIoU": miou,
            "OverallAccuracy": accuracy,
            "MacroF1": macro_f1_value,
            "per_class_IoU": per_class_iou,
            "per_class_F1": [float(value) for value in per_class_f1],
            "per_class_support": [int(value) for value in target_count],
            "per_class_predicted_count": [int(value) for value in predicted_count],
            "per_class_union": [int(value) for value in union],
            "ECE": ece_value,
            "ECE_bin_diagnostics": ece_bin_diagnostics,
            "ClasswiseECE": classwise_ece_value,
            "ClasswiseECE_per_class": classwise_values,
            "ClasswiseECE_bin_diagnostics": classwise_bin_diagnostics,
            "AdaECE": adaece,
            "NLL": self.nll_sum / n,
            "Brier": self.brier_sum / n,
            "AUROC": auroc,
            "AUPRC": auprc,
            "AURC": aurc,
            "E-AURC": eaurc,
            "AUSE": ause,
            "FPR95": fpr95,
            "undefined_metric_reasons": undefined_reasons,
        }


def _adaptive_ece_numpy(confidence: np.ndarray, correct: np.ndarray, bins: int) -> float:
    order = np.argsort(confidence, kind="stable")
    values, first, counts = np.unique(confidence[order], return_index=True, return_counts=True)
    del values
    n = confidence.size
    midpoint = np.cumsum(counts) - counts / 2
    group_bin = np.minimum((midpoint * bins / n).astype(np.int64), bins - 1)
    sorted_correct = correct[order].astype(np.float64)
    result = 0.0
    for bin_id in np.unique(group_bin):
        groups = np.flatnonzero(group_bin == bin_id)
        starts = first[groups]
        ends = starts + counts[groups]
        count = int(np.sum(counts[groups]))
        confidence_group_sums = np.add.reduceat(confidence[order], first)
        mean_conf = float(np.sum(confidence_group_sums[groups]))
        mean_correct = float(np.sum([sorted_correct[a:b].sum() for a, b in zip(starts, ends)]))
        result += count / n * abs(mean_correct / count - mean_conf / count)
    return result


def _aurc_numpy(uncertainty: np.ndarray, error: np.ndarray) -> float:
    order = np.argsort(uncertainty, kind="stable")
    sorted_scores = uncertainty[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    counts = np.diff(np.r_[starts, error.size])
    ordered_error = error[order].astype(np.float64)
    group_error = np.add.reduceat(ordered_error, starts)
    group_ids = np.repeat(np.arange(counts.size), counts)
    within = np.arange(error.size) - np.repeat(starts, counts) + 1
    prior_n = np.repeat(starts, counts)
    prior_error = np.repeat(np.cumsum(group_error) - group_error, counts)
    tie_error = np.repeat(group_error, counts)
    tie_size = np.repeat(counts, counts)
    risk = (prior_error + within * tie_error / tie_size) / (prior_n + within)
    return float(risk.mean())


def _ause_numpy(uncertainty: np.ndarray, error: np.ndarray) -> float:
    n = error.size
    if n < 2:
        raise ValueError("AUSE requires at least two pixels to define an area")
    order = np.argsort(-uncertainty, kind="stable")
    sorted_scores = uncertainty[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    counts = np.diff(np.r_[starts, n])
    sorted_error = error[order].astype(np.float64)
    group_error = np.add.reduceat(sorted_error, starts)
    within = np.arange(n) - np.repeat(starts, counts) + 1
    prior_n = np.repeat(starts, counts)
    prior_error = np.repeat(np.cumsum(group_error) - group_error, counts)
    tie_error = np.repeat(group_error, counts)
    tie_size = np.repeat(counts, counts)
    remaining = n - prior_n - within
    retained = error.sum() - prior_error - within * tie_error / tie_size
    after = np.divide(retained, remaining, out=np.zeros(n, dtype=np.float64), where=remaining > 0)
    observed = np.concatenate(([error.mean()], after))
    r = np.arange(n, dtype=np.float64)
    oracle = np.concatenate((np.maximum(error.sum() - r, 0) / (n - r), [0.0]))
    gap = observed - oracle
    return float(np.sum((gap[:-1] + gap[1:]) * (0.5 / n)))


def _auprc_numpy(uncertainty: np.ndarray, error: np.ndarray) -> float:
    """Tie-grouped non-interpolated average precision for error detection."""
    positives = int(error.sum())
    if positives == 0:
        raise ValueError("AUPRC is undefined when there are no incorrect pixels")
    order = np.argsort(-uncertainty, kind="stable")
    scores = uncertainty[order]
    positive = error[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), scores.size - 1]
    true_positives = np.cumsum(positive)[ends]
    false_positives = np.cumsum(1 - positive)[ends]
    previous_true_positives = np.r_[0, true_positives[:-1]]
    recall_increment = (true_positives - previous_true_positives) / positives
    precision = true_positives / (true_positives + false_positives)
    return float(np.sum(recall_increment * precision))


def _roc_numpy(scores: np.ndarray, positive: np.ndarray) -> tuple[float, float]:
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_positive = positive[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), positive.size - 1]
    tps = np.r_[0, np.cumsum(sorted_positive)[ends]] / n_pos
    fps = np.r_[0, np.cumsum(1 - sorted_positive)[ends]] / n_neg
    auroc = float(np.sum((tps[:-1] + tps[1:]) * np.diff(fps) * 0.5))
    fpr95 = float(fps[np.flatnonzero(tps >= .95)[0]])
    return auroc, fpr95


@torch.inference_mode()
def compute_metrics(
    scores: Tensor,
    target: Tensor,
    num_classes: int,
    *,
    input_type: InputType,
    ignore_index: int | None = None,
    valid_mask: Tensor | None = None,
    ece_bins: int = 15,
    adaptive_ece_bins: int = 15,
) -> dict[str, Any]:
    """Convenience one-batch wrapper around the dataset-level accumulator."""
    accumulator = SegmentationMetricAccumulator(
        num_classes,
        input_type=input_type,
        ignore_index=ignore_index,
        ece_bins=ece_bins,
        adaptive_ece_bins=adaptive_ece_bins,
    )
    accumulator.update(scores, target, valid_mask=valid_mask)
    return accumulator.compute()
