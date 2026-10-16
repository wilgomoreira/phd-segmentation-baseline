"""Isolation and approved-settings checks for the run-01 factorial pilot."""

from pathlib import Path

import pytest

from scripts.run_pastis_factorial_pilot import (
    MODELS,
    VARIANTS,
    _load_catalog,
    build_variant_config,
)


def test_factorial_variants_are_isolated_and_keep_approved_settings():
    catalog, _ = _load_catalog()
    output_dirs = set()
    planned = []
    for model in MODELS:
        for variant in VARIANTS:
            item = catalog["variants"][model][variant]
            if item["action"] == "reuse_completed_baseline":
                assert model == "deeplabv3plus_resnet50" and variant == "scratch_no_scheduler"
                assert Path(item["reference"]).is_dir()
                continue
            config = build_variant_config(model, variant, catalog=catalog)
            output = Path(config["checkpoint_dir"]).parent
            assert output not in output_dirs
            output_dirs.add(output)
            planned.append((model, variant, config))
            assert config["seed"] == 2401
            assert config["epochs"] == 100
            assert config["data"]["batch_size"] == 16
            assert config["optimizer"] == {
                "name": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4,
            }
            assert config["loss"] == {"name": "CrossEntropyLoss", "ignore_index": 19}
            assert config["early_stopping"] == {"enabled": True, "patience": 15, "min_delta": 0.001}
            assert config["resume"] is None
            assert config["initialization"]["kind"] in {"scratch", "imagenet_encoder"}
            if variant.endswith("polynomial"):
                assert config["scheduler"] == {
                    "name": "polynomial", "parameters": {"total_iters": 100, "power": 0.9},
                }
            else:
                assert config["scheduler"] == {"name": "none", "parameters": {}}
            if config["initialization"]["kind"] == "imagenet_encoder":
                assert len(config["initialization"]["sha256"]) == 64
                assert config["initialization"]["checkpoint_path"].startswith(
                    "/home/wilgomoreira/workspace/partilha_remota/cache/"
                )
                assert config["resume"] is None
    assert len(planned) == 7
    assert len(output_dirs) == 7


def test_existing_deeplab_scratch_baseline_is_only_a_reference():
    catalog, _ = _load_catalog()
    variant = catalog["variants"]["deeplabv3plus_resnet50"]["scratch_no_scheduler"]
    assert variant["action"] == "reuse_completed_baseline"
    assert Path(variant["reference"]).name.startswith("run_")
    with pytest.raises(ValueError, match="reference, not a new run"):
        build_variant_config("deeplabv3plus_resnet50", "scratch_no_scheduler", catalog=catalog)


def test_pilot_catalog_uses_run01_protocol_without_editing_it():
    catalog, dataset = _load_catalog()
    assert dataset["single_date"] == {"strategy": "nearest_date", "value": "2019-05-15"}
    assert dataset["channels"] == 10
    assert dataset["train_folds"] == [3, 4, 5]
    assert dataset["validation_fold"] == 2
    assert dataset["test_fold"] == 1
    assert catalog["output_root"].endswith("/pastis_factorial_pilot")
