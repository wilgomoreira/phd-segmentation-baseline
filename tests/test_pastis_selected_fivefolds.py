"""Regression tests for selected five-fold queue completion verification."""

import hashlib

import pytest

from scripts import run_pastis_selected_fivefolds as queue
from scripts.run_pastis_baselines import UnsafeRunState


def _summary():
    return {
        "best_epoch": 38,
        "best_checkpoint_selection_miou": 0.25206162374543073,
        "final_epoch": 53,
        "elapsed_seconds": 1561.8763068779954,
    }


def test_completion_summary_allows_additional_checkpoint_hash_fields():
    record = {
        **_summary(),
        "best_checkpoint_sha256": "best-hash",
        "latest_checkpoint_sha256": "latest-hash",
    }
    queue._validate_completion_summary(record, _summary(), "deeplabv3plus_resnet50", 2)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("best_epoch", 39),
        ("best_checkpoint_selection_miou", 0.25),
        ("final_epoch", 52),
        ("elapsed_seconds", 1500.0),
    ],
)
def test_completion_summary_rejects_scientific_or_run_record_mismatch(field, bad_value):
    record = _summary()
    record[field] = bad_value
    with pytest.raises(UnsafeRunState, match="recorded completion summary mismatch"):
        queue._validate_completion_summary(
            record, _summary(), "deeplabv3plus_resnet50", 2,
        )


def test_checkpoint_hash_verification_accepts_matching_bytes_and_rejects_mismatch(tmp_path):
    best_path = tmp_path / "best.pt"
    latest_path = tmp_path / "latest.pt"
    best_path.write_bytes(b"best checkpoint fixture")
    latest_path.write_bytes(b"latest checkpoint fixture")
    state = {
        "best_checkpoint_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "latest_checkpoint_sha256": hashlib.sha256(latest_path.read_bytes()).hexdigest(),
    }

    assert queue._verify_checkpoint_hashes(
        state, best_path, latest_path, "deeplabv3plus_resnet50", 2,
    ) == state
    state["latest_checkpoint_sha256"] = "incorrect-hash"
    with pytest.raises(UnsafeRunState, match="checkpoint checksum mismatch"):
        queue._verify_checkpoint_hashes(
            state, best_path, latest_path, "deeplabv3plus_resnet50", 2,
        )


def test_skipped_completed_run_keeps_its_original_training_snapshot():
    original = {"manifest": "original.json", "archive_sha256": "original-hash"}
    maintenance = {"manifest": "maintenance.json", "archive_sha256": "maintenance-hash"}
    state = {"status": "completed", "source_snapshot": original.copy()}

    queue._preserve_training_snapshot(
        state, "skipped", maintenance, "deeplabv3plus_resnet50", 2,
    )

    assert state["source_snapshot"] == original
