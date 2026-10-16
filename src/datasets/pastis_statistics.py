"""Training-only statistics for a prespecified PASTIS single-date partition."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

from .pastis import DateSelection, _metadata_dates, select_observation_index
from .preprocessing import load_array, resolve_relative


def fit_pastis_training_statistics(
    root: str | Path,
    training_patch_ids: Iterable[int],
    *,
    date_selection: DateSelection,
    training_folds: Iterable[int],
    output_path: str | Path,
    overwrite: bool = False,
) -> dict:
    """Fit pixel-weighted per-band population statistics from training patches.

    Only the selected S2 plane is read; annotation arrays are never opened.
    The destination is written atomically and is protected against accidental
    replacement unless ``overwrite`` is explicitly requested.
    """
    root = Path(root)
    output_path = Path(output_path)
    patch_ids = sorted(int(value) for value in training_patch_ids)
    fold_set = {int(value) for value in training_folds}
    if len(fold_set) != 3 or not fold_set.issubset({1, 2, 3, 4, 5}):
        raise ValueError("statistics fitting requires exactly three official training folds")
    if not patch_ids or len(set(patch_ids)) != len(patch_ids):
        raise ValueError("training patch IDs must be nonempty and unique")
    metadata_path = root / "metadata.geojson"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = {}
    for feature in metadata.get("features", []):
        properties = feature.get("properties", {})
        patch_id = int(properties["ID_PATCH"])
        if patch_id in rows:
            raise ValueError(f"duplicate metadata patch ID {patch_id}")
        rows[patch_id] = properties
    if set(patch_ids) - rows.keys():
        raise ValueError("training patch IDs are absent from PASTIS metadata")

    sums = np.zeros(10, dtype=np.float64)
    square_sums = np.zeros(10, dtype=np.float64)
    pixel_count = 0
    for patch_id in patch_ids:
        properties = rows[patch_id]
        if int(properties["Fold"]) not in fold_set:
            raise ValueError(f"patch {patch_id} is outside the declared training folds")
        dates = _metadata_dates(properties["dates-S2"])
        array = load_array(resolve_relative(root, f"DATA_S2/S2_{patch_id}.npy"), mmap=True)
        if array.ndim != 4 or array.shape[1] != 10 or array.shape[0] != len(dates):
            raise ValueError(f"unexpected S2 shape/date count for training patch {patch_id}")
        image = np.asarray(array[select_observation_index(dates, date_selection)], dtype=np.float64)
        if image.shape[0] != 10 or not np.isfinite(image).all():
            raise ValueError(f"invalid selected S2 image for training patch {patch_id}")
        sums += image.sum(axis=(1, 2), dtype=np.float64)
        square_sums += np.square(image).sum(axis=(1, 2), dtype=np.float64)
        pixel_count += int(image.shape[1] * image.shape[2])

    mean = sums / pixel_count
    variance = np.maximum(square_sums / pixel_count - mean * mean, 0.0)
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("training data produced invalid per-channel statistics")
    id_digest = hashlib.sha256(",".join(map(str, patch_ids)).encode()).hexdigest()
    result = {
        "schema_version": 1,
        "dataset": "PASTIS",
        "method": "pixel-weighted population mean/std (ddof=0) over selected training S2 planes",
        "date_selection": {"strategy": date_selection.strategy, "value": date_selection.value},
        "channels": 10,
        "training_patch_count": len(patch_ids),
        "training_folds": sorted(fold_set),
        "training_patch_ids_sha256": id_digest,
        "pixel_count_per_channel": pixel_count,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace existing statistics file: {output_path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to replace existing statistics file: {output_path}")
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return result
