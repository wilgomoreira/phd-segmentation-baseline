"""Partition checks for the leakage-controlled five-fold PASTIS protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OFFICIAL_FOLDS = {1, 2, 3, 4, 5}


def plan_pastis_partitions(metadata_path: str | Path, protocol: dict[str, Any]) -> dict[str, list[int]]:
    """Resolve patch IDs from official folds and disjoint validation tile groups.

    This reads metadata only. It never opens S2 or annotation arrays, including
    for the test fold.
    """
    train_folds = protocol.get("train_folds")
    validation_fold = protocol.get("validation_fold")
    test_fold = protocol.get("test_fold")
    if not isinstance(train_folds, list) or len(train_folds) != 3:
        raise ValueError("each run must declare exactly three training folds")
    fold_roles = [*train_folds, validation_fold, test_fold]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in fold_roles):
        raise ValueError("all fold assignments must be integers")
    if len(set(fold_roles)) != 5 or set(fold_roles) != OFFICIAL_FOLDS:
        raise ValueError("train, validation, and test roles must partition official folds 1–5")
    if protocol.get("validation_folds") != [validation_fold] or protocol.get("test_folds") != [test_fold]:
        raise ValueError("validation_folds/test_folds must agree with the singular run roles")
    if (protocol.get("channels") != 10 or protocol.get("num_classes") != 19
            or protocol.get("ignore_index") != 19
            or protocol.get("models") != ["deeplabv3plus", "segformer_b1"]):
        raise ValueError("protocol must use 10 S2 bands, classes 0–18, ignored class 19, and both approved models")
    date_cfg = protocol.get("single_date") or {}
    if date_cfg != {"strategy": "nearest_date", "value": "2019-05-15"}:
        raise ValueError("protocol must use the approved nearest-date selection at 2019-05-15")
    selection_tiles = protocol.get("checkpoint_selection_tiles")
    calibration_tiles = protocol.get("calibration_tiles")
    if (not isinstance(selection_tiles, list) or not selection_tiles or
            not isinstance(calibration_tiles, list) or not calibration_tiles):
        raise ValueError("checkpoint-selection and calibration tile lists are required")
    if set(selection_tiles) & set(calibration_tiles):
        raise ValueError("checkpoint-selection and calibration tiles must be disjoint")

    source = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    records: list[tuple[int, int, str]] = []
    seen: set[int] = set()
    for feature in source.get("features", []):
        properties = feature.get("properties", {})
        patch_id, fold = int(properties["ID_PATCH"]), int(properties["Fold"])
        tile = properties.get("TILE")
        if patch_id in seen:
            raise ValueError(f"duplicate PASTIS patch ID {patch_id}")
        if fold not in OFFICIAL_FOLDS or not isinstance(tile, str) or not tile:
            raise ValueError(f"invalid fold or tile metadata for patch {patch_id}")
        seen.add(patch_id)
        records.append((patch_id, fold, tile))
    if not records:
        raise ValueError("PASTIS metadata has no patch records")

    validation_tiles_present = {tile for _, fold, tile in records if fold == validation_fold}
    declared_validation_tiles = set(selection_tiles) | set(calibration_tiles)
    if validation_tiles_present != declared_validation_tiles:
        raise ValueError(
            "validation tile lists must cover exactly the tiles represented in the validation fold"
        )
    partitions = {
        "train": sorted(pid for pid, fold, _ in records if fold in set(train_folds)),
        "checkpoint_selection": sorted(
            pid for pid, fold, tile in records
            if fold == validation_fold and tile in set(selection_tiles)
        ),
        "calibration": sorted(
            pid for pid, fold, tile in records
            if fold == validation_fold and tile in set(calibration_tiles)
        ),
        "test": sorted(pid for pid, fold, _ in records if fold == test_fold),
    }
    id_sets = {key: set(value) for key, value in partitions.items()}
    keys = list(id_sets)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            if id_sets[left] & id_sets[right]:
                raise ValueError(f"sample IDs overlap between {left} and {right}")
    if set.union(*id_sets.values()) != seen:
        raise ValueError("partition IDs do not cover the metadata patch IDs exactly")
    return partitions
