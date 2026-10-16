"""Batch collation for the project's segmentation sample interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


_TENSOR_FIELDS = ("image", "mask", "valid_mask")


def collate_segmentation_batch(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack segmentation tensors and retain per-sample metadata as a list.

    Metadata can contain values such as ``datetime.date`` that PyTorch's
    default collator cannot batch. Keeping each metadata mapping intact also
    preserves identifiers, folds, and dataset-specific fields for callers.
    Additional sample fields are retained as per-sample lists.
    """
    if not samples:
        raise ValueError("cannot collate an empty segmentation batch")
    if any(not isinstance(sample, Mapping) for sample in samples):
        raise TypeError("each segmentation sample must be a mapping")

    keys = set(samples[0])
    required = {*_TENSOR_FIELDS, "metadata"}
    if not required.issubset(keys):
        raise KeyError(f"segmentation sample is missing required fields: {sorted(required - keys)}")
    if any(set(sample) != keys for sample in samples[1:]):
        raise ValueError("segmentation samples in a batch must have matching fields")

    batch: dict[str, Any] = {}
    for field in _TENSOR_FIELDS:
        values = [sample[field] for sample in samples]
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"segmentation sample field {field!r} must contain tensors")
        batch[field] = torch.stack(values, dim=0)

    batch["metadata"] = [sample["metadata"] for sample in samples]
    for field in keys - required:
        batch[field] = [sample[field] for sample in samples]
    return batch
