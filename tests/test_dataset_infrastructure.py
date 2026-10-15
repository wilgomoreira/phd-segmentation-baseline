import json

import numpy as np
import pytest
import torch

from src.datasets import (
    DateSelection,
    FiveBillionPixelsDataset,
    Normalization,
    PastisSingleDateDataset,
    assign_spatial_grid_splits,
    tile_windows,
)


def _write_pastis_fixture(root):
    (root / "DATA_S2").mkdir()
    (root / "ANNOTATIONS").mkdir()
    features = []
    for patch_id, fold in [(101, 1), (102, 2)]:
        # Date planes have distinct constant values, all ten channels retained.
        image = np.stack(
            [np.full((10, 3, 4), 11, dtype=np.uint16), np.full((10, 3, 4), 55, dtype=np.uint16)]
        )
        np.save(root / "DATA_S2" / f"S2_{patch_id}.npy", image)
        target = np.full((1, 3, 4), patch_id % 3, np.uint8)
        target[0, 0, 0] = 1
        np.save(root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy", target)
        features.append({
            "type": "Feature",
            "properties": {
                "ID_PATCH": patch_id,
                "Fold": fold,
                "dates-S2": {"0": "20200105", "1": "20200101"},
                "tile_id": "tile-synthetic",
            },
            "geometry": None,
        })
    (root / "metadata.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


def test_pastis_prespecified_single_date_and_metadata(tmp_path):
    _write_pastis_fixture(tmp_path)
    ds = PastisSingleDateDataset(
        tmp_path,
        date_selection=DateSelection("nearest_date", "2020-01-03"),
        folds=[2],
        normalization=Normalization(scale=0.1),
    )
    sample = ds[0]
    assert sample["image"].shape == (10, 3, 4)
    assert sample["image"].dtype == torch.float32
    assert torch.allclose(sample["image"], torch.full((10, 3, 4), 1.1))
    assert sample["mask"].shape == sample["valid_mask"].shape == (3, 4)
    assert sample["valid_mask"].all()
    assert sample["metadata"]["patch_id"] == 102
    assert sample["metadata"]["fold"] == 2
    # Equidistant date selection is deterministic and chooses the earlier date.
    assert sample["metadata"]["selected_acquisition_date"] == "2020-01-01"
    assert sample["metadata"]["selected_acquisition_index"] == 0
    assert sample["metadata"]["tile_id"] == "tile-synthetic"


def test_pastis_tiling_and_invalid_date_index(tmp_path):
    _write_pastis_fixture(tmp_path)
    tiled = PastisSingleDateDataset(
        tmp_path,
        date_selection=DateSelection("index", 1),
        tile_size=2,
        tile_stride=2,
        ignore_index=1,
    )
    assert len(tiled) == 8
    first = tiled[0]
    assert first["image"].shape == (10, 2, 2)
    assert first["metadata"]["window_yx_yx"] == (0, 0, 2, 2)
    assert not first["valid_mask"].all()
    with pytest.raises(IndexError, match="out of range"):
        PastisSingleDateDataset(tmp_path, date_selection=DateSelection("index", 99))


def test_five_billion_manifest_preserves_four_channels_and_validity(tmp_path):
    image = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(5, 6, 4)
    mask = np.arange(30, dtype=np.uint8).reshape(5, 6) % 3
    valid = np.ones((5, 6), dtype=np.uint8)
    valid[0, 0] = 0
    np.save(tmp_path / "scene.npy", image)
    np.save(tmp_path / "labels.npy", mask)
    np.save(tmp_path / "valid.npy", valid)
    row = {
        "image": "scene.npy",
        "mask": "labels.npy",
        "valid_mask": "valid.npy",
        "image_layout": "HWC",
        "image_id": "image-A",
        "geographic_id": "region-4",
        "x": 123.0,
        "y": 456.0,
        "split": "train",
        "metadata": {"source": "synthetic"},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ds = FiveBillionPixelsDataset(manifest, split="train", ignore_index=2, normalization=Normalization(scale=0.5))
    sample = ds[0]
    assert sample["image"].shape == (4, 5, 6)
    assert sample["image"][3, 0, 0].item() == pytest.approx(float(image[0, 0, 3]) * 0.5)
    assert sample["mask"].shape == (5, 6)
    assert not sample["valid_mask"][0, 0]
    assert not sample["valid_mask"][0, 2]
    assert sample["metadata"]["image_id"] == "image-A"
    assert sample["metadata"]["geographic_id"] == "region-4"
    assert sample["metadata"]["split"] == "train"
    assert sample["metadata"]["x"] == 123.0 and sample["metadata"]["y"] == 456.0
    with pytest.raises(ValueError, match="four channels"):
        bad = np.zeros((3, 5, 6), dtype=np.uint8)
        np.save(tmp_path / "bad.npy", bad)
        bad_row = {**row, "image": "bad.npy", "image_layout": "CHW"}
        bad_manifest = tmp_path / "bad.jsonl"
        bad_manifest.write_text(json.dumps(bad_row), encoding="utf-8")
        FiveBillionPixelsDataset(bad_manifest)


def test_spatial_grid_splits_are_reproducible_and_guarded():
    rows = [{"x": float(i * 100), "y": 0.0, "image_id": f"i{i}"} for i in range(48)]
    fractions = {"train": .6, "validation": .2, "test": .2}
    a = assign_spatial_grid_splits(rows, block_size=100, fractions=fractions, seed=23, guard_cells=1)
    b = assign_spatial_grid_splits(rows, block_size=100, fractions=fractions, seed=23, guard_cells=1)
    assert [r["split"] for r in a] == [r["split"] for r in b]
    assert [r["spatial_block_id"] for r in a] == [r["spatial_block_id"] for r in b]
    assigned = {int(r["spatial_block_id"].split(":")[0]): r["split"] for r in a if r["split"] != "excluded"}
    for cell, split in assigned.items():
        for other, other_split in assigned.items():
            if abs(cell - other) <= 1:
                assert split == other_split
    assert {r["split"] for r in a} >= {"train", "validation", "test", "excluded"}
    with pytest.raises(ValueError, match="overwrite existing split"):
        assign_spatial_grid_splits([{**rows[0], "split": "official_test"}], block_size=100, fractions=fractions, seed=23)


def test_normalization_validation_and_tile_windows():
    assert tile_windows(5, 6, 3, 3) == [(0, 0, 3, 3), (0, 3, 3, 6), (2, 0, 5, 3), (2, 3, 5, 6)]
    image = np.ones((4, 2, 2), dtype=np.uint16)
    normalized = Normalization(mean=(1, 1, 1, 1), std=(2, 2, 2, 2)).apply(image)
    assert normalized.shape == (4, 2, 2)
    assert np.all(normalized == 0)
    with pytest.raises(ValueError, match="positive"):
        Normalization(mean=(0, 0), std=(1, 0)).apply(np.zeros((2, 1, 1)))
