import json
from datetime import date

import numpy as np
import torch

from src.datasets import collate_segmentation_batch
from src.training.config import build_loaders


def test_collate_segmentation_batch_stacks_tensors_and_preserves_metadata():
    acquisition_dates = [date(2020, 1, 1), date(2020, 1, 5)]
    samples = [
        {
            "image": torch.full((10, 3, 4), float(patch_id)),
            "mask": torch.full((3, 4), patch_id % 3, dtype=torch.long),
            "valid_mask": torch.ones((3, 4), dtype=torch.bool),
            "metadata": {
                "patch_id": patch_id,
                "fold": fold,
                "dates": acquisition_dates,
            },
        }
        for patch_id, fold in ((101, 1), (102, 2))
    ]

    batch = collate_segmentation_batch(samples)

    assert batch["image"].shape == (2, 10, 3, 4)
    assert batch["mask"].shape == (2, 3, 4)
    assert batch["valid_mask"].shape == (2, 3, 4)
    assert batch["metadata"] == [sample["metadata"] for sample in samples]
    assert batch["metadata"][0]["dates"] == acquisition_dates
    assert all(isinstance(value, date) for value in batch["metadata"][0]["dates"])
    assert [item["patch_id"] for item in batch["metadata"]] == [101, 102]
    assert [item["fold"] for item in batch["metadata"]] == [1, 2]


def _write_pastis_fixture(root):
    (root / "DATA_S2").mkdir()
    (root / "ANNOTATIONS").mkdir()
    features = []
    for patch_id, fold in ((101, 1), (102, 2), (103, 3), (104, 4)):
        image = np.full((2, 10, 3, 4), patch_id, dtype=np.uint16)
        target = np.zeros((3, 3, 4), dtype=np.uint8)
        np.save(root / "DATA_S2" / f"S2_{patch_id}.npy", image)
        np.save(root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy", target)
        features.append({
            "type": "Feature",
            "properties": {
                "ID_PATCH": patch_id,
                "Fold": fold,
                "dates-S2": ["20200101", "20200105"],
            },
            "geometry": None,
        })
    (root / "metadata.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def test_configured_loaders_collate_pastis_datetime_metadata(tmp_path):
    _write_pastis_fixture(tmp_path)
    dataset_config = {
        "dataset": "pastis",
        "root": str(tmp_path),
        "train_folds": [1, 2],
        "validation_folds": [3, 4],
        "single_date": {"strategy": "index", "value": 0},
        "normalization": {"scale": None, "mean": None, "std": None},
    }

    train_loader, validation_loader = build_loaders(
        dataset_config,
        batch_size=2,
        num_workers=0,
        seed=17,
        pin_memory=False,
    )

    for loader, expected_folds in ((train_loader, {1, 2}), (validation_loader, {3, 4})):
        batch = next(iter(loader))
        assert batch["image"].shape == (2, 10, 3, 4)
        assert batch["mask"].shape == (2, 3, 4)
        assert batch["valid_mask"].shape == (2, 3, 4)
        assert {item["fold"] for item in batch["metadata"]} == expected_folds
        assert all(isinstance(item["dates"][0], date) for item in batch["metadata"])
