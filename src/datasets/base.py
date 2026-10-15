"""Common array-manifest segmentation dataset interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import Normalization, _spatial_shape, channel_first, load_array, resolve_relative, tile_windows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read project-defined JSON Lines manifests without inferring file names."""
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest {path} contains no records")
    return rows


class ArrayManifestDataset(Dataset):
    """Load samples described by explicit image/mask paths and axis layouts.

    Each item is a dictionary with ``image`` [C,H,W], ``mask`` [H,W],
    ``valid_mask`` [H,W], and metadata retaining the manifest fields.
    """

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        *,
        base_dir: str | Path,
        normalization: Normalization | None = None,
        ignore_index: int | None = None,
        split: str | None = None,
        tile_size: int | tuple[int, int] | None = None,
        tile_stride: int | tuple[int, int] | None = None,
        temporal: bool = False,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.normalization = normalization or Normalization()
        self.ignore_index = ignore_index
        self.temporal = temporal
        source_records = list(records)
        if split is not None:
            source_records = [r for r in source_records if r.get("split") == split]
        if not source_records:
            raise ValueError("no manifest records selected")
        self.records: list[tuple[dict[str, Any], tuple[int, int, int, int] | None]] = []
        for record in source_records:
            if "image" not in record or "mask" not in record:
                raise ValueError("each record must explicitly provide image and mask paths")
            layout = record.get("image_layout", "TCHW" if temporal else None)
            if layout is None:
                raise ValueError("image_layout must be specified; channel ordering is never inferred")
            image_path = resolve_relative(self.base_dir, record["image"])
            image = load_array(image_path, key=record.get("image_key"), mmap=True)
            height, width = _spatial_shape(image, layout, temporal=temporal)
            if tile_size is None:
                self.records.append((record, None))
            else:
                windows = tile_windows(height, width, tile_size, tile_stride)
                if not windows:
                    raise ValueError(f"image {image_path} is smaller than configured tile size")
                self.records.extend((record, window) for window in windows)
        if not self.records:
            raise ValueError("no samples or full-size tiles were produced")

    def _load_image(self, record: dict[str, Any]) -> np.ndarray:
        image = load_array(resolve_relative(self.base_dir, record["image"]), key=record.get("image_key"), mmap=True)
        return channel_first(image, record["image_layout"])

    def _load_mask(self, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        mask = load_array(resolve_relative(self.base_dir, record["mask"]), key=record.get("mask_key"), mmap=True)
        if mask.ndim == 3 and 1 in mask.shape:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError("semantic mask must be two-dimensional or have one singleton channel")
        if not np.issubdtype(mask.dtype, np.integer):
            raise TypeError("semantic mask must contain integer class IDs")
        valid = np.ones(mask.shape, dtype=bool)
        if self.ignore_index is not None:
            valid &= mask != self.ignore_index
        valid_path = record.get("valid_mask")
        if valid_path is not None:
            explicit = load_array(resolve_relative(self.base_dir, valid_path), key=record.get("valid_mask_key"), mmap=True)
            explicit = np.squeeze(explicit)
            if explicit.shape != mask.shape:
                raise ValueError("explicit valid mask must match semantic mask dimensions")
            valid &= explicit.astype(bool)
        return mask, valid

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, window = self.records[index]
        image = self._load_image(record)
        mask, valid = self._load_mask(record)
        if image.shape[-2:] != mask.shape:
            raise ValueError(f"image/mask spatial shape mismatch for {record['image']}")
        metadata = dict(record.get("metadata", {}))
        # Preserve identifiers, coordinates, CRS tags, and user metadata while
        # omitting only the fields used to locate/read the arrays themselves.
        path_fields = {
            "image", "mask", "valid_mask", "image_key", "mask_key",
            "valid_mask_key", "image_layout", "mask_layout", "metadata",
        }
        for key, value in record.items():
            if key not in path_fields:
                metadata.setdefault(key, value)
        if window is not None:
            y0, x0, y1, x1 = window
            image = image[:, y0:y1, x0:x1]
            mask = mask[y0:y1, x0:x1]
            valid = valid[y0:y1, x0:x1]
            metadata["window_yx_yx"] = window
        image = self.normalization.apply(image)
        return {
            "image": torch.from_numpy(image.copy()),
            "mask": torch.from_numpy(np.asarray(mask, dtype=np.int64).copy()),
            "valid_mask": torch.from_numpy(np.asarray(valid, dtype=bool).copy()),
            "metadata": metadata,
        }
