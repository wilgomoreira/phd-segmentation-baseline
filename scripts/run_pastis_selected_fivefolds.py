#!/usr/bin/env python3
"""Run only the remaining folds for the selected PASTIS baseline pair.

Run 01 is reused from its separately verified DeepLab and SegFormer artifacts.
Training begins only without --dry-run. New outputs use a dedicated root and
the SegFormer NVlabs 256-channel profile explicitly.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import scripts.run_pastis_baselines as baseline
from scripts.run_pastis_factorial_pilot import _load_catalog, build_variant_config
from scripts.run_pastis_segformer_nvlabs256_sequence import validate_condition_artifacts


DEFAULT_OUTPUT_ROOT = baseline.DEFAULT_OUTPUT_ROOT
SEED = 2401
SELECTED_RUNS = tuple(
    (model, fold)
    for model in ("deeplabv3plus_resnet50", "segformer_b1")
    for fold in range(2, 6)
)
PRETRAINED_MANIFEST = PROJECT_ROOT / "configs/pretrained/pastis_imagenet_encoders.json"
RUN01_DEEPLAB_ROOT = Path(
    "/home/wilgomoreira/workspace/partilha_remota/experiments/"
    "phd-segmentation-baselines/pastis_full_runs/deeplabv3plus_resnet50/run_01"
)
RUN01_SEGFORMER_ROOT = Path(
    "/home/wilgomoreira/workspace/partilha_remota/experiments/"
    "phd-segmentation-baselines/pastis_run01_segformer_b1_nvlabs_decoder256/"
    "segformer_b1/pretrained_no_scheduler"
)
SEGFORMER_CATALOG = PROJECT_ROOT / "configs/pilots/pastis_run01_segformer_b1_nvlabs256.yaml"
SNAPSHOT_EXCLUDED_DIRS = {".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", ".venv"}
COMPLETION_SUMMARY_FIELDS = (
    "best_epoch",
    "best_checkpoint_selection_miou",
    "final_epoch",
    "elapsed_seconds",
)
_BASE_BUILD_RUN_CONFIG = baseline.build_run_config
_BASE_STRIP_RESUME = baseline._strip_resume


def _selected_build_run_config(model_name: str, run_number: int, seed: int, output_root: Path):
    config = _BASE_BUILD_RUN_CONFIG(model_name, run_number, seed, output_root)
    if model_name == "segformer_b1":
        manifest = json.loads(PRETRAINED_MANIFEST.read_text(encoding="utf-8"))
        initialization = dict(manifest["sources"]["nvidia_mit_b1_imagenet1k"])
        checkpoint = Path(initialization["checkpoint_path"])
        observed_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if observed_sha256 != initialization.get("sha256"):
            raise baseline.UnsafeRunState(f"pretrained MiT-B1 checkpoint hash mismatch: {checkpoint}")
        config["initialization"] = {**initialization, "kind": "imagenet_encoder"}
    return config


def _strip_runtime_verification(config):
    result = _BASE_STRIP_RESUME(config)
    initialization = result.get("initialization")
    if isinstance(initialization, dict):
        initialization.pop("verification", None)
    return result


def _verify_run01_snapshot_after_test_only_edit(catalog: dict, saved_snapshot: dict) -> None:
    """Reconstruct the run-01 fingerprint before the authorized dry-run test fix."""
    files = catalog.get("implementation_snapshot_files")
    if not isinstance(files, list) or sorted(set(files)) != saved_snapshot.get("files"):
        raise baseline.UnsafeRunState("run-01 source snapshot file list differs from the current catalog")
    test_relative = "tests/test_pastis_segformer_nvlabs256.py"
    if test_relative not in files:
        raise baseline.UnsafeRunState("run-01 snapshot does not include the dry-run test file")
    current_test = (PROJECT_ROOT / test_relative).read_text(encoding="utf-8")
    current_test = current_test.replace("import hashlib\n", "", 1)
    start = "def test_new_sequence_dry_run_is_exactly_four_segformer_conditions_and_writes_nothing():\n"
    end = "\n\ndef test_sequence_checkpoint_gate_checks_completion_provenance_and_decoder_width"
    if start not in current_test or end not in current_test or "def inventory(root):" not in current_test:
        raise baseline.UnsafeRunState("cannot reconstruct the recorded run-01 dry-run test source")
    left, rest = current_test.split(start, 1)
    _, right = rest.split(end, 1)
    original_test = left + start + '''    output_root = Path(_load_catalog(CATALOG)[0]["output_root"])
    assert not output_root.exists()
    plan = run_sequence(execute=False, catalog_path=CATALOG)
    assert plan["status"] == "dry_run"
    assert [condition["variant"] for condition in plan["plan"]] == list(SEQUENCE)
    assert all("segformer_b1" in condition["output"] for condition in plan["plan"])
    assert not output_root.exists()
''' + end + right
    trainer_relative = "src/training/trainer.py"
    trainer_path = PROJECT_ROOT / trainer_relative
    current_trainer = trainer_path.read_text(encoding="utf-8")
    current_signature = "def validate(self, *, full_metrics: bool = True) -> dict[str, Any]:"
    original_signature = "def validate(self, *, full_metrics: bool = True) -> dict[str, float | int | None]:"
    if current_signature not in current_trainer:
        raise baseline.UnsafeRunState("cannot reconstruct the recorded trainer type annotation")
    original_trainer = current_trainer.replace(current_signature, original_signature, 1)
    digest = hashlib.sha256()
    for relative in sorted(set(files)):
        if relative == test_relative:
            data = original_test.encode("utf-8")
        elif relative == trainer_relative:
            data = original_trainer.encode("utf-8")
        else:
            data = (PROJECT_ROOT / relative).read_bytes()
        digest.update(relative.encode("utf-8") + b"\0" + data)
    if digest.hexdigest() != saved_snapshot.get("sha256"):
        raise baseline.UnsafeRunState(
            "run-01 source fingerprint does not match after reversing only the dry-run test fix"
        )


def _validate_source_snapshot(manifest_path: Path) -> dict[str, str]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repository") != str(PROJECT_ROOT):
        raise baseline.UnsafeRunState("source snapshot belongs to a different repository")
    current_head = baseline.subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if current_head != manifest.get("git_head"):
        raise baseline.UnsafeRunState("Git HEAD changed after the source snapshot was captured")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise baseline.UnsafeRunState("source snapshot manifest has no file inventory")
    expected = {item["path"]: item["sha256"] for item in entries}
    actual_paths = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and not any(part in SNAPSHOT_EXCLUDED_DIRS for part in path.relative_to(PROJECT_ROOT).parts)
    }
    if actual_paths != set(expected):
        raise baseline.UnsafeRunState("repository file inventory changed after source snapshot capture")
    for relative, expected_hash in expected.items():
        digest = hashlib.sha256()
        with (PROJECT_ROOT / relative).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise baseline.UnsafeRunState(f"source file changed after snapshot capture: {relative}")
    archive = manifest.get("archive") or {}
    archive_path = Path(archive.get("path", ""))
    if not archive_path.is_file() or hashlib.sha256(archive_path.read_bytes()).hexdigest() != archive.get("sha256"):
        raise baseline.UnsafeRunState("source snapshot archive is missing or its checksum changed")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "archive": str(archive_path),
        "archive_sha256": archive["sha256"],
    }


def _verify_checkpoint_hashes(state: dict, best_path: Path, latest_path: Path,
                              model_name: str, run_number: int) -> dict[str, str]:
    """Check checkpoint bytes independently of the scientific summary fields."""
    observed = {
        "best_checkpoint_sha256": baseline.file_sha256(best_path),
        "latest_checkpoint_sha256": baseline.file_sha256(latest_path),
    }
    if any(state.get(key) != digest for key, digest in observed.items()):
        raise baseline.UnsafeRunState(
            f"checkpoint checksum mismatch for {model_name}/run_{run_number:02d}"
        )
    return observed


def _validate_completion_summary(record: dict, expected: dict,
                                 model_name: str, run_number: int) -> None:
    """Compare only the defined scientific summary fields; allow integrity metadata."""
    mismatches = {
        field: (record.get(field), expected.get(field))
        for field in COMPLETION_SUMMARY_FIELDS
        if field not in record or field not in expected or record[field] != expected[field]
    }
    if mismatches:
        raise baseline.UnsafeRunState(
            f"recorded completion summary mismatch for {model_name}/run_{run_number:02d}: "
            f"{mismatches}"
        )


def _preserve_training_snapshot(run_state: dict, disposition: str,
                                snapshot_record: dict, model_name: str,
                                run_number: int) -> None:
    """Attach this source snapshot only to newly trained artifacts."""
    if disposition == "completed":
        run_state["source_snapshot"] = snapshot_record
    elif disposition == "skipped":
        if not isinstance(run_state.get("source_snapshot"), dict):
            raise baseline.UnsafeRunState(
                f"completed {model_name}/run_{run_number:02d} has no original source-snapshot provenance"
            )
    else:
        raise baseline.UnsafeRunState(
            f"unexpected launcher disposition {disposition!r} for {model_name}/run_{run_number:02d}"
        )


def verify_reused_run01() -> dict[str, float]:
    """Verify both run-01 artifacts read-only before the queue can proceed."""
    deeplab_config = baseline.build_run_config(
        "deeplabv3plus_resnet50", 1, SEED, RUN01_DEEPLAB_ROOT,
    )
    deeplab_state = json.loads((RUN01_DEEPLAB_ROOT / "launcher_state.json").read_text(encoding="utf-8"))
    if (deeplab_state.get("status") != "completed"
            or deeplab_state.get("model") != "deeplabv3plus_resnet50"
            or deeplab_state.get("run") != 1):
        raise baseline.UnsafeRunState("DeepLab run-01 is not marked as the expected completed run")
    deeplab_verified = baseline.verify_run_files(
        RUN01_DEEPLAB_ROOT, deeplab_config, repair_metrics=False,
    )
    if not deeplab_verified["complete"]:
        raise baseline.UnsafeRunState("DeepLab run-01 is not at a verified completion point")
    deeplab_latest = baseline._load_checkpoint(deeplab_verified["latest"])

    catalog, _ = _load_catalog(SEGFORMER_CATALOG)
    segformer_config = build_variant_config(
        "segformer_b1", "pretrained_no_scheduler", catalog=catalog,
        catalog_path=SEGFORMER_CATALOG,
    )
    segformer_run_dir = next((RUN01_SEGFORMER_ROOT / "checkpoints").glob("run_*"))
    saved_resolved = json.loads((segformer_run_dir / "resolved_configuration.json").read_text(encoding="utf-8"))
    saved_training = copy.deepcopy(saved_resolved["training"])
    saved_snapshot = copy.deepcopy(saved_training.get("pilot_metadata", {}).get("implementation_snapshot"))
    if not isinstance(saved_snapshot, dict):
        raise baseline.UnsafeRunState("SegFormer run-01 lacks its implementation source fingerprint")
    _verify_run01_snapshot_after_test_only_edit(catalog, saved_snapshot)
    expected_training = copy.deepcopy(segformer_config)
    expected_training["pilot_metadata"]["implementation_snapshot"] = saved_snapshot
    saved_training.get("initialization", {}).pop("verification", None)
    if saved_training != expected_training:
        raise baseline.UnsafeRunState("SegFormer run-01 scientific configuration differs from the selected baseline")
    segformer_config["pilot_metadata"]["implementation_snapshot"] = saved_snapshot
    segformer_verified = validate_condition_artifacts(
        "pretrained_no_scheduler", RUN01_SEGFORMER_ROOT, segformer_config,
    )

    # Prove the selected current model profiles load the run-01 checkpoints exactly.
    from src.training.config import make_model
    for config, root, model_name in (
        (deeplab_config, RUN01_DEEPLAB_ROOT, "deeplabv3plus_resnet50"),
        (segformer_config, RUN01_SEGFORMER_ROOT, "segformer_b1"),
    ):
        dataset_config = baseline._read_yaml(PROJECT_ROOT / config["dataset_config"])
        model_config = baseline._read_yaml(PROJECT_ROOT / config["model_config"])
        model = make_model(model_config, dataset_config, int(config["num_classes"]))
        run_dirs = (sorted((root / "checkpoints").glob("run_*")) if model_name == "segformer_b1"
                    else sorted(root.glob("run_*")))
        for checkpoint_name in ("best.pt", "latest.pt"):
            payload = baseline._load_checkpoint(run_dirs[0] / checkpoint_name)
            try:
                model.load_state_dict(payload["model"], strict=True)
            except RuntimeError as error:
                raise baseline.UnsafeRunState(
                    f"{model_name} run-01 {checkpoint_name} is incompatible with the selected architecture"
                ) from error
        del model
    return {
        "deeplab_run01_selection_miou": float(deeplab_latest["best_validation_miou"]),
        "segformer_run01_selection_miou": float(segformer_verified["best_checkpoint_selection_miou"]),
    }


def verify_completed_queue_run(model_name: str, run_number: int, output_root: Path) -> dict:
    """Verify both saved states and record fields before the queue advances."""
    root = output_root / model_name / f"run_{run_number:02d}"
    config = baseline.build_run_config(model_name, run_number, SEED, root)
    verified = baseline.verify_run_files(root, config, repair_metrics=False)
    if not verified["complete"]:
        raise baseline.UnsafeRunState(f"{model_name}/run_{run_number:02d} is not complete")
    state = json.loads((root / "launcher_state.json").read_text(encoding="utf-8"))
    checkpoint_dir = verified["checkpoint_dir"]
    best_path, latest_path = checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt"
    checkpoint_hashes = _verify_checkpoint_hashes(
        state, best_path, latest_path, model_name, run_number,
    )
    best = baseline._load_checkpoint(best_path)
    latest = baseline._load_checkpoint(latest_path)
    if model_name == "segformer_b1":
        model_config = baseline._read_yaml(PROJECT_ROOT / config["model_config"])
        if model_config.get("decoder_profile") != "nvlabs_reference" or model_config.get("decoder_channels") != 256:
            raise baseline.UnsafeRunState("selected SegFormer run did not use the NVlabs 256-channel model profile")
        for checkpoint_name, checkpoint in (("best.pt", best), ("latest.pt", latest)):
            fusion = checkpoint["model"].get("decoder.fuse.0.weight")
            classifier = checkpoint["model"].get("decoder.classifier.weight")
            if (tuple(getattr(fusion, "shape", ())) != (256, 1024, 1, 1)
                    or tuple(getattr(classifier, "shape", ())) != (19, 256, 1, 1)):
                raise baseline.UnsafeRunState(f"{checkpoint_name} is not an NVlabs 256-channel SegFormer checkpoint")
    from src.training.config import make_model
    dataset_config = baseline._read_yaml(PROJECT_ROOT / config["dataset_config"])
    model_config = baseline._read_yaml(PROJECT_ROOT / config["model_config"])
    model = make_model(model_config, dataset_config, int(config["num_classes"]))
    try:
        model.load_state_dict(best["model"], strict=True)
        model.load_state_dict(latest["model"], strict=True)
    except RuntimeError as error:
        raise baseline.UnsafeRunState(
            f"checkpoint weights do not strictly load into {model_name}/run_{run_number:02d}"
        ) from error
    record = {
        "best_epoch": int(best["epoch"]) + 1,
        "best_checkpoint_selection_miou": float(best["best_validation_miou"]),
        "final_epoch": int(latest["epoch"]) + 1,
        "elapsed_seconds": float(state["training_elapsed_seconds"]),
        **checkpoint_hashes,
    }
    expected = {
        "best_epoch": state.get("best_epoch"),
        "best_checkpoint_selection_miou": state.get("best_checkpoint_selection_miou"),
        "final_epoch": state.get("final_epoch"),
        "elapsed_seconds": state.get("training_elapsed_seconds"),
    }
    _validate_completion_summary(record, expected, model_name, run_number)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-snapshot-manifest", type=Path, help="verified immutable source snapshot manifest (required to execute)")
    parser.add_argument("--dry-run", action="store_true", help="show queue without creating files or training")
    args = parser.parse_args()

    # Do not allow the legacy 128-channel SegFormer config to enter this queue.
    baseline.MODEL_CONFIGS["segformer_b1"] = ("segformer_b1_nvlabs.yaml", "segformer_b1")
    # The generic full-run launcher defaults to scratch. Attach and hash-check
    # the selected NVIDIA MiT-B1 encoder for every new SegFormer fold.
    baseline.build_run_config = _selected_build_run_config
    baseline._strip_resume = _strip_runtime_verification
    output_root = args.output_root.expanduser().resolve()
    reused = verify_reused_run01()
    print(
        "REUSE verified run 01: "
        f"DeepLab selection mIoU={reused['deeplab_run01_selection_miou']:.6f}; "
        f"SegFormer NVlabs-256 selection mIoU={reused['segformer_run01_selection_miou']:.6f}"
    )
    if args.dry_run:
        for model, fold in SELECTED_RUNS:
            config = baseline.build_run_config(model, fold, SEED, output_root / model / f"run_{fold:02d}")
            print(
                f"PLAN {model}/run_{fold:02d}: dataset={config['dataset_config']} "
                f"model_config={config['model_config']} seed={SEED} "
                f"initialization={(config.get('initialization') or {'kind': 'scratch'})['kind']} "
                f"deterministic={config['deterministic']} scheduler=none"
            )
        print(f"Prepared {len(SELECTED_RUNS)} remaining runs; run 01 is reused and no training was launched.")
        return

    if args.source_snapshot_manifest is None:
        raise baseline.UnsafeRunState("--source-snapshot-manifest is required before training can execute")
    snapshot_record = _validate_source_snapshot(args.source_snapshot_manifest)

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".launcher.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise baseline.UnsafeRunState(f"another launcher owns {output_root}") from error
        for model, fold in SELECTED_RUNS:
            snapshot_record = _validate_source_snapshot(args.source_snapshot_manifest)
            disposition = baseline._run_one(model, fold, SEED, output_root, dry_run=False)
            record = verify_completed_queue_run(model, fold, output_root)
            state_path = output_root / model / f"run_{fold:02d}" / "launcher_state.json"
            run_state = json.loads(state_path.read_text(encoding="utf-8"))
            _preserve_training_snapshot(
                run_state, disposition, snapshot_record, model, fold,
            )
            baseline._atomic_json(state_path, run_state)
            print(
                f"VERIFIED {model}/run_{fold:02d}: best_epoch={record['best_epoch']} "
                f"best_checkpoint_selection_mIoU={record['best_checkpoint_selection_miou']:.6f} "
                f"final_epoch={record['final_epoch']} elapsed_seconds={record['elapsed_seconds']:.1f}"
            )


if __name__ == "__main__":
    main()
