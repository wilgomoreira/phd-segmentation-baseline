import math

import pytest
import torch

from src.evaluation.metrics import (
    adaptive_ece,
    auroc_error_detection,
    brier_score,
    compute_metrics,
    ece,
    expected_aurc,
    fpr_at_tpr,
    mean_iou,
    multiclass_nll,
    sparsification_ause,
    SegmentationMetricAccumulator,
)


def test_perfect_predictions_and_absent_classes():
    target = torch.tensor([[[0, 1], [1, 0]]])
    pred = target.clone()
    score = torch.full((1, 4, 2, 2), -8.0)
    score.scatter_(1, target.unsqueeze(1), 8.0)
    value, per_class = mean_iou(pred, target, 4, return_per_class=True)
    assert value == pytest.approx(1.0)
    assert per_class[:2] == [1.0, 1.0]
    assert per_class[2:] == [None, None]
    metrics = compute_metrics(score, target, 4, input_type="logits")
    assert metrics["mIoU"] == pytest.approx(1.0)
    assert metrics["NLL"] < 1e-5
    assert metrics["Brier"] < 1e-8
    assert metrics["AUROC"] is None and metrics["FPR95"] is None
    with pytest.raises(ValueError, match="both positive and negative"):
        auroc_error_detection(torch.zeros(4), torch.zeros(4, dtype=torch.bool))


def test_completely_incorrect_predictions_and_uncertainty_ranking():
    target = torch.tensor([0, 0, 1, 1])
    pred = 1 - target
    assert mean_iou(pred, target, 2) == 0.0
    error = torch.ones(4, dtype=torch.bool)
    with pytest.raises(ValueError, match="both positive and negative"):
        fpr_at_tpr(torch.ones(4), error)

    # Correct pixels are confidently right; errors are confidently wrong.
    y = torch.tensor([0, 1, 0, 1])
    p = torch.tensor([[.99, .01], [.01, .99], [.40, .60], [.65, .35]])
    u = 1 - p.max(dim=1).values
    err = p.argmax(1) != y
    assert auroc_error_detection(u, err) == 1.0
    assert fpr_at_tpr(u, err) == 0.0
    aurc, excess = expected_aurc(u, err)
    assert aurc == pytest.approx(5 / 24)
    assert excess == pytest.approx(0.0)
    assert sparsification_ause(u, err) == pytest.approx(0.0)


def test_uniform_probabilities_calibration_nll_and_brier():
    p = torch.full((4, 3), 1 / 3)
    target = torch.tensor([0, 1, 2, 0])
    confidence = p.max(1).values
    correct = p.argmax(1) == target
    assert ece(confidence, correct, bin_count=3) == pytest.approx(1 / 6)
    assert adaptive_ece(confidence, correct, bin_count=2) == pytest.approx(1 / 6)
    assert multiclass_nll(torch.log(p), target) == pytest.approx(math.log(3))
    assert brier_score(p, target) == pytest.approx(2 / 3)


def test_ignore_index_and_valid_mask_are_shared():
    target = torch.tensor([[[0, 1, 255]]])
    probs = torch.tensor([[[[.9, .2, 0.0]], [[.1, .8, 1.0]]]])
    metrics = compute_metrics(
        probs,
        target,
        2,
        input_type="probabilities",
        ignore_index=255,
        valid_mask=torch.tensor([[[True, True, False]]]),
    )
    assert metrics["mIoU"] == pytest.approx(1.0)
    assert metrics["NLL"] == pytest.approx((-math.log(.9) - math.log(.8)) / 2)


def test_batches_and_class_counts():
    torch.manual_seed(41)
    for batch, classes in [(1, 2), (3, 5)]:
        target = torch.randint(0, classes, (batch, 2, 3))
        logits = torch.randn(batch, classes, 2, 3)
        result = mean_iou(logits.argmax(1), target, classes)
        assert 0 <= result <= 1
        calibrated = compute_metrics(logits, target, classes, input_type="logits")
        assert set(calibrated) == {"mIoU", "ECE", "AdaECE", "NLL", "Brier", "AUROC", "AURC", "E-AURC", "AUSE", "FPR95"}


def test_constant_uncertainty_and_roc_tie_grouping():
    u = torch.ones(4)
    error = torch.tensor([False, True, False, True])
    assert auroc_error_detection(u, error) == pytest.approx(0.5)
    assert fpr_at_tpr(u, error) == pytest.approx(1.0)
    aurc, excess = expected_aurc(u, error)
    assert aurc >= 0
    assert excess >= 0
    assert 0 <= sparsification_ause(u, error) <= 1


def test_explicit_validation_and_infinite_nll_for_zero_probability():
    with pytest.raises(ValueError, match="sum to 1"):
        compute_metrics(torch.tensor([[[.3]], [[.3]]]).permute(1, 0, 2), torch.tensor([[0]]), 2, input_type="probabilities")
    assert math.isinf(multiclass_nll(torch.tensor([[-math.inf, 0.0]]), torch.tensor([0])))
    with pytest.raises(ValueError, match="positive integer"):
        ece(torch.tensor([.5]), torch.tensor([True]), bin_count=0)


def test_streaming_matches_whole_dataset_across_batch_sizes_and_orders():
    torch.manual_seed(2026)
    scores = torch.tensor(
        [[[[3., 0., 0., 2.], [0., 2., 0., 1.]],
          [[0., 3., 2., 0.], [2., 0., 1., 0.]],
          [[0., 0., 0., 0.], [0., 0., 3., 2.]]],
         [[[0., 1., 3., 0.], [2., 0., 0., 0.]],
          [[2., 0., 0., 3.], [0., 3., 2., 0.]],
          [[0., 2., 0., 0.], [0., 0., 0., 3.]]]]
    )
    target = torch.tensor([[[0, 1, 2, 0], [1, 1, 2, 0]], [[1, 2, 0, 1], [0, 1, 1, 2]]])
    target[0, 0, 3] = 255
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[1, 1, 0] = False
    whole = compute_metrics(scores, target, 3, input_type="logits", ignore_index=255, valid_mask=valid)
    streamed = SegmentationMetricAccumulator(3, input_type="logits", ignore_index=255)
    streamed.update(scores[:1], target[:1], valid_mask=valid[:1])
    streamed.update(scores[1:], target[1:], valid_mask=valid[1:])
    first = streamed.compute()
    second = streamed.compute()
    for key in whole:
        assert first[key] == pytest.approx(whole[key]) if whole[key] is not None else first[key] is None
        assert second[key] == first[key]

    reordered = SegmentationMetricAccumulator(3, input_type="logits", ignore_index=255)
    reordered.update(scores[1:], target[1:], valid_mask=valid[1:])
    reordered.update(scores[:1], target[:1], valid_mask=valid[:1])
    reverse_batch = reordered.compute()
    for key in first:
        assert reverse_batch[key] == pytest.approx(first[key]) if first[key] is not None else reverse_batch[key] is None


def test_adaptive_ece_ties_are_order_invariant_and_kept_together():
    confidence = torch.tensor([.8, .8, .8, .8, .6, .6])
    correct = torch.tensor([True, False, False, True, True, False])
    value = adaptive_ece(confidence, correct, bin_count=3)
    perm = torch.tensor([4, 2, 0, 5, 1, 3])
    assert adaptive_ece(confidence[perm], correct[perm], bin_count=3) == pytest.approx(value)
    # Two tied-confidence groups produce two indivisible adaptive bins.
    assert value == pytest.approx((4 / 6) * abs(.5 - .8) + (2 / 6) * abs(.5 - .6))


def test_tied_uncertainty_aurc_and_ause_are_order_invariant():
    uncertainty = torch.ones(4)
    error = torch.tensor([False, True, False, True])
    expected_aurc_value, expected_excess = expected_aurc(uncertainty, error)
    expected_ause_value = sparsification_ause(uncertainty, error)
    assert expected_aurc_value == pytest.approx(.5)
    assert expected_excess == pytest.approx(7 / 24)
    assert expected_ause_value == pytest.approx(7 / 24)
    permutation = torch.tensor([1, 3, 0, 2])
    assert expected_aurc(uncertainty[permutation], error[permutation])[0] == pytest.approx(expected_aurc_value)
    assert sparsification_ause(uncertainty[permutation], error[permutation]) == pytest.approx(expected_ause_value)


def test_nontrivial_fpr95_and_ause_reference_values():
    # At TPR=.95, one of 20 correct pixels has crossed the threshold: FPR=.05.
    uncertainty = torch.tensor([.95] + [.90] * 19 + [.80] + [.10] * 19)
    error = torch.tensor([False] + [True] * 19 + [True] + [False] * 19)
    assert auroc_error_detection(uncertainty, error) == pytest.approx(.95)
    assert fpr_at_tpr(uncertainty, error) == pytest.approx(.05)

    # Manual sparsification curves yield trapezoidal area 11/24.
    u = torch.tensor([.4, .3, .2, .1])
    e = torch.tensor([False, True, False, True])
    assert sparsification_ause(u, e) == pytest.approx(11 / 24)


def test_disk_storage_reset_degenerate_and_autograd():
    target = torch.tensor([[[0, 1], [0, 1]]])
    logits = torch.tensor([[[[2., 0.], [2., 0.]], [[0., 2.], [0., 2.]]]], requires_grad=True)
    acc = SegmentationMetricAccumulator(2, input_type="logits", storage="disk")
    acc.update(logits, target)
    computed = acc.compute()
    in_memory = SegmentationMetricAccumulator(2, input_type="logits", storage="memory")
    in_memory.update(logits, target)
    assert in_memory.compute() == computed
    assert computed["AUROC"] is None and computed["FPR95"] is None
    assert acc.compute() == computed
    acc.reset()
    with pytest.raises(ValueError, match="before adding"):
        acc.compute()
    acc.update(logits, target)
    assert acc.compute() == computed


def test_single_pixel_ause_degeneracy_is_explicit():
    target = torch.tensor([[[0]]])
    logits = torch.tensor([[[[1.]], [[0.]]]])
    acc = SegmentationMetricAccumulator(2, input_type="logits")
    acc.update(logits, target)
    with pytest.raises(ValueError, match="at least two pixels"):
        acc.compute()


def test_absent_ground_truth_class_with_false_positive_prediction():
    pred = torch.tensor([0, 2, 0, 0])
    target = torch.tensor([0, 0, 0, 0])
    value, per_class = mean_iou(pred, target, 3, return_per_class=True)
    assert value == pytest.approx(.375)
    assert per_class == [pytest.approx(.75), None, 0.0]


def test_direct_metric_validation_rejects_float_class_ids_and_bad_scores():
    with pytest.raises(TypeError, match="integer class indices"):
        mean_iou(torch.tensor([0.0, 1.0]), torch.tensor([0, 1]), 2)
    with pytest.raises(ValueError, match="finite floating"):
        auroc_error_detection(torch.tensor([.1, float("nan")]), torch.tensor([False, True]))


def test_half_precision_logits_are_promoted_for_finite_nll():
    target = torch.tensor([[[1, 1]]])
    logits = torch.tensor([[[[1000., -1000.]], [[-1000., 1000.]]]], dtype=torch.float16)
    acc = SegmentationMetricAccumulator(2, input_type="logits")
    acc.update(logits, target)
    assert acc.compute()["NLL"] == pytest.approx(1000.0)
