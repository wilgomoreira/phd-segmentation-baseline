import json

import numpy as np
import pytest

from src.datasets import DateSelection
from src.datasets.pastis_statistics import fit_pastis_training_statistics
from src.training.config import build_calibration_dataset, build_loaders
from src.training.pastis_protocol import plan_pastis_partitions


TILES = ["t30uxv", "t31tfj", "t31tfm", "t32ulu"]


def _write_protocol_fixture(root):
    (root / "DATA_S2").mkdir()
    (root / "ANNOTATIONS").mkdir()
    features = []
    for fold in range(1, 6):
        for tile_index, tile in enumerate(TILES):
            patch_id = fold * 100 + tile_index
            first = 10 + fold
            second = 20 + fold
            image = np.stack([
                np.full((10, 2, 2), first, dtype=np.uint16),
                np.full((10, 2, 2), second, dtype=np.uint16),
            ])
            np.save(root / "DATA_S2" / f"S2_{patch_id}.npy", image)
            target = np.zeros((3, 2, 2), dtype=np.uint8)
            np.save(root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy", target)
            features.append({
                "type": "Feature",
                "properties": {
                    "ID_PATCH": patch_id,
                    "Fold": fold,
                    "TILE": tile,
                    "dates-S2": {"0": "20190515", "1": "20190520"},
                },
                "geometry": None,
            })
    (root / "metadata.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8",
    )


def _protocol(root, stats_file):
    return {
        "dataset": "pastis",
        "root": str(root),
        "channels": 10,
        "num_classes": 19,
        "ignore_index": 19,
        "models": ["deeplabv3plus", "segformer_b1"],
        "train_folds": [3, 4, 5],
        "validation_fold": 2,
        "validation_folds": [2],
        "test_fold": 1,
        "test_folds": [1],
        "checkpoint_selection_tiles": [TILES[0], TILES[1]],
        "calibration_tiles": [TILES[2], TILES[3]],
        "single_date": {"strategy": "nearest_date", "value": "2019-05-15"},
        "normalization": {"scale": None, "statistics_file": str(stats_file)},
    }


def test_protocol_partitions_are_unique_disjoint_and_spatially_grouped(tmp_path):
    _write_protocol_fixture(tmp_path)
    partitions = plan_pastis_partitions(tmp_path / "metadata.geojson", _protocol(tmp_path, "unused"))
    assert {name: len(ids) for name, ids in partitions.items()} == {
        "train": 12, "checkpoint_selection": 2, "calibration": 2, "test": 4,
    }
    sets = [set(ids) for ids in partitions.values()]
    assert all(not left & right for i, left in enumerate(sets) for right in sets[i + 1:])


def test_protocol_rejects_fold_overlap_and_incomplete_tile_groups(tmp_path):
    _write_protocol_fixture(tmp_path)
    bad_roles = _protocol(tmp_path, "unused")
    bad_roles["test_fold"] = 2
    bad_roles["test_folds"] = [2]
    with pytest.raises(ValueError, match="partition official folds"):
        plan_pastis_partitions(tmp_path / "metadata.geojson", bad_roles)
    bad_tiles = _protocol(tmp_path, "unused")
    bad_tiles["calibration_tiles"] = [TILES[2]]
    with pytest.raises(ValueError, match="cover exactly"):
        plan_pastis_partitions(tmp_path / "metadata.geojson", bad_tiles)


def test_training_statistics_use_only_training_ids_and_selected_date(tmp_path, monkeypatch):
    _write_protocol_fixture(tmp_path)
    config = _protocol(tmp_path, tmp_path / "stats.json")
    partitions = plan_pastis_partitions(tmp_path / "metadata.geojson", config)
    import src.datasets.pastis_statistics as statistics_module

    read_paths = []
    original_load = statistics_module.load_array

    def tracked_load(path, **kwargs):
        read_paths.append(str(path))
        return original_load(path, **kwargs)

    monkeypatch.setattr(statistics_module, "load_array", tracked_load)
    result = fit_pastis_training_statistics(
        tmp_path,
        partitions["train"],
        date_selection=DateSelection("nearest_date", "2019-05-15"),
        training_folds=config["train_folds"],
        output_path=tmp_path / "stats.json",
    )
    expected = np.mean([13, 14, 15])
    assert np.allclose(result["mean"], expected)
    assert np.allclose(result["std"], np.std([13, 14, 15], ddof=0))
    assert result["training_patch_count"] == 12
    assert len(read_paths) == 12
    assert all("DATA_S2" in path for path in read_paths)
    assert json.loads((tmp_path / "stats.json").read_text())["training_folds"] == [3, 4, 5]
    with pytest.raises(FileExistsError):
        fit_pastis_training_statistics(
            tmp_path, partitions["train"],
            date_selection=DateSelection("nearest_date", "2019-05-15"),
            training_folds=config["train_folds"], output_path=tmp_path / "stats.json",
        )


def test_training_loaders_exclude_calibration_and_test_samples(tmp_path):
    _write_protocol_fixture(tmp_path)
    config = _protocol(tmp_path, tmp_path / "stats.json")
    fit_pastis_training_statistics(
        tmp_path, plan_pastis_partitions(tmp_path / "metadata.geojson", config)["train"],
        date_selection=DateSelection("nearest_date", "2019-05-15"),
        training_folds=config["train_folds"], output_path=tmp_path / "stats.json",
    )
    train, selection = build_loaders(
        config, batch_size=2, num_workers=0, seed=4, pin_memory=False,
    )
    train_ids = {row[0]["patch_id"] for row in train.dataset.records}
    selection_ids = {row[0]["patch_id"] for row in selection.dataset.records}
    partitions = plan_pastis_partitions(tmp_path / "metadata.geojson", config)
    assert train_ids == set(partitions["train"])
    assert selection_ids == set(partitions["checkpoint_selection"])
    assert train.dataset.normalization.mean == selection.dataset.normalization.mean
    assert train.dataset.normalization.std == selection.dataset.normalization.std
    assert not selection_ids & set(partitions["calibration"])
    assert not train_ids & set(partitions["test"])
    assert not selection_ids & set(partitions["test"])
    calibration = build_calibration_dataset(config)
    calibration_ids = {record[0]["patch_id"] for record in calibration.records}
    assert calibration_ids == set(partitions["calibration"])
    assert not calibration_ids & set(partitions["test"])
