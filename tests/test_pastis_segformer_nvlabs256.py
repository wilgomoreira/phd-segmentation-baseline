"""Isolation and safety checks for the reference-aligned SegFormer-B1 pilot."""

import json
import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from scripts.run_pastis_factorial_pilot import build_variant_config, _load_catalog
from scripts.run_pastis_segformer_nvlabs256_sequence import (
    CATALOG,
    SEQUENCE,
    run_sequence,
    validate_condition_artifacts,
)


def test_versioned_catalog_preserves_the_original_and_isolates_all_four_conditions():
    original, _ = _load_catalog()
    catalog, dataset = _load_catalog(CATALOG)
    old_model = yaml.safe_load((Path(original["project_root"]) / original["model_configs"]["segformer_b1"]).read_text())
    new_model = yaml.safe_load((Path(catalog["project_root"]) / catalog["model_configs"]["segformer_b1"]).read_text())
    assert old_model["decoder_channels"] == 128
    assert old_model.get("decoder_profile") is None
    assert new_model["decoder_profile"] == "nvlabs_reference"
    assert new_model["decoder_channels"] == new_model["decoder_hidden_size"] == 256
    assert list(catalog["variants"]) == ["segformer_b1"]
    assert dataset["train_folds"] == [3, 4, 5]
    assert dataset["validation_fold"] == 2 and dataset["test_fold"] == 1
    assert catalog["seed"] == 2401
    assert catalog["optimizer"] == {"name": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4}
    assert catalog["early_stopping"] == {"enabled": True, "patience": 15, "min_delta": 0.001}
    assert catalog["output_root"].endswith("pastis_run01_segformer_b1_nvlabs_decoder256")
    prepared = build_variant_config("segformer_b1", SEQUENCE[0], catalog=catalog, catalog_path=CATALOG)
    assert len(prepared["pilot_metadata"]["implementation_snapshot"]["sha256"]) == 64
    output_roots = {
        Path(build_variant_config("segformer_b1", variant, catalog=catalog, catalog_path=CATALOG)["checkpoint_dir"]).parents[1]
        for variant in SEQUENCE
    }
    assert len(output_roots) == 1
    assert output_roots.pop() == Path(catalog["output_root"]) / "segformer_b1"


def test_new_sequence_dry_run_is_exactly_four_segformer_conditions_and_writes_nothing():
    output_root = Path(_load_catalog(CATALOG)[0]["output_root"])

    def inventory(root):
        if not root.exists():
            return None
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()
        }

    before = inventory(output_root)
    plan = run_sequence(execute=False, catalog_path=CATALOG)
    assert plan["status"] == "dry_run"
    assert [condition["variant"] for condition in plan["plan"]] == list(SEQUENCE)
    assert all("segformer_b1" in condition["output"] for condition in plan["plan"])
    assert inventory(output_root) == before


def test_sequence_checkpoint_gate_checks_completion_provenance_and_decoder_width(tmp_path):
    catalog, _ = _load_catalog(CATALOG)
    variant = "scratch_no_scheduler"
    training_config = build_variant_config("segformer_b1", variant, catalog=catalog, catalog_path=CATALOG)
    variant_root = tmp_path / variant
    run_dir = variant_root / "checkpoints" / "run_test"
    run_dir.mkdir(parents=True)
    (variant_root / "launcher_state.json").write_text(json.dumps({
        "status": "completed", "return_code": 0, "model": "segformer_b1", "variant": variant, "seed": 2401,
    }))
    row = {"epoch": 0, "validation": {"mIoU": 0.25}}
    (run_dir / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    payload = {
        "model": {
            "decoder.fuse.0.weight": torch.zeros(256, 1024, 1, 1),
            "decoder.classifier.weight": torch.zeros(19, 256, 1, 1),
        },
        "epoch": 0,
        "best_validation_miou": 0.25,
        "configuration": {"training": training_config},
    }
    for name in ("best.pt", "latest.pt"):
        torch.save(payload, run_dir / name)
    result = validate_condition_artifacts(variant, variant_root, training_config)
    assert result["status"] == "completed_verified"
    assert result["best_epoch"] == 1
    assert result["best_checkpoint_selection_miou"] == 0.25

    payload["model"]["decoder.fuse.0.weight"] = torch.zeros(128, 1024, 1, 1)
    torch.save(payload, run_dir / "best.pt")
    with pytest.raises(RuntimeError, match="256-channel reference decoder"):
        validate_condition_artifacts(variant, variant_root, training_config)
