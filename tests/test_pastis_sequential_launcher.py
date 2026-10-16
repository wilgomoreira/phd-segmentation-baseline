"""Filesystem-only safety tests for the prepared sequential PASTIS launcher."""

import json

import pytest
import torch

from scripts.run_pastis_baselines import (
    RUNS,
    UnsafeRunState,
    _find_resume,
    _initialize_root,
    _read_state,
    _write_state,
    build_run_config,
    run_sequence,
)


def test_launcher_builds_all_ten_approved_model_fold_configs_without_training(tmp_path):
    assert len(RUNS) == 10
    result = run_sequence(31415, tmp_path / "not-created", dry_run=True)
    assert len(result) == 10
    assert not (tmp_path / "not-created").exists()
    for model_name, run_number in RUNS:
        root = tmp_path / model_name / f"run_{run_number:02d}"
        config = build_run_config(model_name, run_number, 31415, root)
        assert config["seed"] == 31415
        assert config["epochs"] == 100
        assert config["data"]["batch_size"] == 16
        assert config["early_stopping"] == {
            "enabled": True, "patience": 15, "min_delta": 0.001,
        }
        assert config["optimizer"] == {
            "name": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.0001,
        }
        assert config["scheduler"] == {"name": "none", "parameters": {}}
        assert config["loss"]["name"] == "CrossEntropyLoss"
        assert config["loss"]["ignore_index"] == 19
        assert config["num_classes"] == 19
        assert config["resume"] is None


def test_launcher_explicitly_selects_only_deeplab_run_01(tmp_path, capsys):
    result = run_sequence(
        2401, tmp_path / "not-created", dry_run=True,
        model_name="deeplabv3plus_resnet50", run_number=1,
    )
    output = capsys.readouterr().out
    assert result == ["planned"]
    assert output.count("PLAN ") == 1
    assert "deeplabv3plus_resnet50_run_01" in output
    assert "run_02" not in output
    assert "segformer_b1" not in output
    assert "seed=2401" in output
    assert not (tmp_path / "not-created").exists()


def test_single_run_selection_requires_both_model_and_run(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        run_sequence(
            2401, tmp_path / "not-created", dry_run=True,
            model_name="deeplabv3plus_resnet50",
        )
    assert not (tmp_path / "not-created").exists()


def _write_incomplete_run(root, config, *, latest_epoch=3):
    _initialize_root(root, config, "deeplabv3plus_resnet50", 1)
    child = root / "run_20261016T000000Z_test"
    child.mkdir()
    early = {"best_miou": 0.4, "epochs_without_improvement": 1, "stopped": False}
    latest = {
        "model": {},
        "optimizer": {},
        "scheduler": None,
        "epoch": latest_epoch,
        "best_validation_miou": 0.4,
        "early_stopping": early,
        "configuration": {"training": config},
        "rng_states": {},
    }
    best = {
        "model": {}, "optimizer": {}, "scheduler": None, "epoch": 2,
        "best_validation_miou": 0.4, "configuration": {"training": config},
        "rng_states": {},
    }
    torch.save(latest, child / "latest.pt")
    torch.save(best, child / "best.pt")
    resolved = {
        "training": config,
        "resolved_training": config,
        "dataset": {},
        "model": {},
    }
    (child / "resolved_configuration.json").write_text(json.dumps(resolved), encoding="utf-8")
    (child / "run_metadata.json").write_text("{}", encoding="utf-8")
    metrics = [
        {"epoch": epoch, "validation": {"mIoU": 0.4}}
        for epoch in range(latest_epoch + 1)
    ]
    (child / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in metrics), encoding="utf-8",
    )
    return child


def test_interrupted_run_resumes_only_from_verified_latest(tmp_path):
    root = tmp_path / "run"
    config = build_run_config("deeplabv3plus_resnet50", 1, 31415, root)
    child = _write_incomplete_run(root, config)
    resume = _find_resume(root, config)
    assert resume is not None and not resume["complete"]
    assert resume["latest"] == child / "latest.pt"
    assert resume["epoch"] == 3


def test_launcher_refuses_unowned_or_inconsistent_output(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    config = build_run_config("deeplabv3plus_resnet50", 1, 31415, root)
    with pytest.raises(UnsafeRunState, match="without launcher ownership"):
        _read_state(root, config, "deeplabv3plus_resnet50", 1)

    root = tmp_path / "bad_run"
    config = build_run_config("deeplabv3plus_resnet50", 1, 31415, root)
    child = _write_incomplete_run(root, config)
    # Simulate a crash after best.pt replacement but before latest.pt replacement.
    torch.save({
        "model": {}, "optimizer": {}, "scheduler": None, "epoch": 4,
        "best_validation_miou": 0.5, "configuration": {"training": config},
        "rng_states": {},
    }, child / "best.pt")
    with pytest.raises(UnsafeRunState, match="inconsistent"):
        _find_resume(root, config)


def test_launcher_skips_only_complete_run_with_matching_provenance(tmp_path, monkeypatch):
    from scripts import run_pastis_baselines as launcher

    root = tmp_path / "deeplabv3plus_resnet50" / "run_01"
    config = build_run_config("deeplabv3plus_resnet50", 1, 31415, root)
    _initialize_root(root, config, "deeplabv3plus_resnet50", 1)
    child = root / "run_20261016T000000Z_test"
    child.mkdir()
    checkpoint = {
        "model": {},
        "optimizer": {},
        "scheduler": None,
        "epoch": 99,
        "best_validation_miou": 0.4,
        "configuration": {"training": config},
        "rng_states": {},
        "early_stopping": {
            "best_miou": 0.4,
            "epochs_without_improvement": 0,
            "stopped": False,
        },
    }
    torch.save(checkpoint, child / "latest.pt")
    torch.save({
        "model": {}, "optimizer": {}, "scheduler": None, "epoch": 90,
        "best_validation_miou": 0.4, "configuration": {"training": config},
        "rng_states": {},
    }, child / "best.pt")
    resolved = {
        "training": config,
        "resolved_training": config,
        "dataset": {},
        "model": {},
    }
    (child / "resolved_configuration.json").write_text(json.dumps(resolved), encoding="utf-8")
    (child / "run_metadata.json").write_text("{}", encoding="utf-8")
    rows = [
        {"epoch": epoch, "validation": {"mIoU": 0.4}}
        for epoch in range(100)
    ]
    (child / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    state = _read_state(root, config, "deeplabv3plus_resnet50", 1)
    _write_state(root, state, "completed", latest=str(child / "latest.pt"), epoch=99)

    def unexpected_training(*args, **kwargs):
        raise AssertionError("verified complete run must be skipped without invoking training")

    monkeypatch.setattr(launcher.subprocess, "run", unexpected_training)
    assert launcher._run_one(
        "deeplabv3plus_resnet50", 1, 31415, tmp_path, dry_run=False,
    ) == "skipped"
