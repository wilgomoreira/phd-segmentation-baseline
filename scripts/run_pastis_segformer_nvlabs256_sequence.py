#!/usr/bin/env python3
"""Run the four reference-aligned SegFormer-B1 run-01 conditions sequentially."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pastis_factorial_pilot import (
    _load_catalog,
    _validate_reference,
    build_variant_config,
)

CATALOG = PROJECT_ROOT / "configs/pilots/pastis_run01_segformer_b1_nvlabs256.yaml"
SEQUENCE = (
    "scratch_no_scheduler",
    "scratch_polynomial",
    "pretrained_no_scheduler",
    "pretrained_polynomial",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _gpu_compute_pids() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def validate_condition_artifacts(variant: str, variant_root: Path, training_config: dict[str, Any]) -> dict[str, Any]:
    """Require successful runner state, matching history and 256-channel checkpoints."""
    if variant not in SEQUENCE:
        raise ValueError(f"unexpected sequence variant: {variant}")
    state_path = variant_root / "launcher_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "completed" or state.get("return_code") != 0:
        raise RuntimeError(f"{variant} runner did not complete successfully")
    if state.get("model") != "segformer_b1" or state.get("variant") != variant or state.get("seed") != 2401:
        raise RuntimeError(f"{variant} launcher state provenance mismatch")

    run_dirs = sorted(path for path in (variant_root / "checkpoints").glob("run_*") if path.is_dir())
    if len(run_dirs) != 1:
        raise RuntimeError(f"{variant} must have exactly one isolated trainer run, found {len(run_dirs)}")
    run_dir = run_dirs[0]
    metrics_path = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not math.isfinite(float(row["validation"]["mIoU"])) for row in rows):
        raise RuntimeError(f"{variant} has empty or non-finite validation history")
    expected_best = max(float(row["validation"]["mIoU"]) for row in rows)
    expected_best_epoch = max(rows, key=lambda row: float(row["validation"]["mIoU"]))["epoch"]
    if rows[-1]["epoch"] >= int(training_config["epochs"]):
        raise RuntimeError(f"{variant} metrics contain an out-of-range epoch")

    expected_init = "scratch" if variant.startswith("scratch_") else "imagenet_encoder"
    expected_scheduler = "polynomial" if variant.endswith("polynomial") else "none"
    checkpoint_info: dict[str, Any] = {}
    for checkpoint_name in ("best.pt", "latest.pt"):
        checkpoint_path = run_dir / checkpoint_name
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size < 1024:
            raise RuntimeError(f"{variant} missing or undersized {checkpoint_name}")
        payload = __import__("torch").load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        model_state = payload.get("model")
        if not isinstance(model_state, dict):
            raise RuntimeError(f"{variant}/{checkpoint_name} has no model state")
        fusion = model_state.get("decoder.fuse.0.weight")
        classifier = model_state.get("decoder.classifier.weight")
        if tuple(getattr(fusion, "shape", ())) != (256, 1024, 1, 1):
            raise RuntimeError(f"{variant}/{checkpoint_name} is not the 256-channel reference decoder")
        if tuple(getattr(classifier, "shape", ())) != (19, 256, 1, 1):
            raise RuntimeError(f"{variant}/{checkpoint_name} has an unexpected class head")
        saved_epoch = int(payload.get("epoch", -1))
        saved_metric = float(payload.get("best_validation_miou", float("nan")))
        if saved_epoch < 0 or saved_epoch >= int(training_config["epochs"]) or not math.isfinite(saved_metric):
            raise RuntimeError(f"{variant}/{checkpoint_name} has invalid epoch or best metric")
        saved_config = payload.get("configuration", {})
        saved_training = saved_config.get("training", {})
        if (saved_training.get("seed") != 2401
                or saved_training.get("pilot_metadata", {}).get("factorial_variant") != variant
                or saved_training.get("model_config") != training_config["model_config"]
                or saved_training.get("initialization", {}).get("kind") != expected_init
                or (saved_training.get("scheduler") or {}).get("name", "none") != expected_scheduler
                or saved_training.get("pilot_metadata", {}).get("implementation_snapshot")
                != training_config.get("pilot_metadata", {}).get("implementation_snapshot")):
            raise RuntimeError(f"{variant}/{checkpoint_name} training provenance mismatch")
        if checkpoint_name == "latest.pt" and saved_epoch != int(rows[-1]["epoch"]):
            raise RuntimeError(f"{variant}/latest.pt epoch does not match the last metric row")
        if checkpoint_name == "best.pt":
            if saved_epoch != int(expected_best_epoch) or not math.isclose(saved_metric, expected_best, rel_tol=1e-8, abs_tol=1e-9):
                raise RuntimeError(f"{variant}/best.pt does not match the best recorded checkpoint-selection score")
        checkpoint_info[checkpoint_name] = {"epoch": saved_epoch, "best_validation_miou": saved_metric,
                                            "bytes": checkpoint_path.stat().st_size}
        del payload, model_state

    return {
        "status": "completed_verified",
        "trainer_run_directory": str(run_dir),
        "epochs_completed": len(rows),
        "best_epoch": int(expected_best_epoch) + 1,
        "best_checkpoint_selection_miou": expected_best,
        "best_checkpoint": checkpoint_info["best.pt"],
        "latest_checkpoint": checkpoint_info["latest.pt"],
    }


def _assert_approved_config(catalog: dict[str, Any], dataset: dict[str, Any], catalog_path: Path) -> None:
    expected_date = {"strategy": "nearest_date", "value": "2019-05-15"}
    if (dataset.get("dataset") != "pastis" or dataset.get("channels") != 10
            or dataset.get("num_classes") != 19 or dataset.get("ignore_index") != 19
            or dataset.get("single_date") != expected_date
            or dataset.get("train_folds") != [3, 4, 5]
            or dataset.get("validation_fold") != 2 or dataset.get("test_fold") != 1):
        raise ValueError("run-01 data protocol differs from the approved PASTIS protocol")
    if (catalog.get("seed") != 2401 or catalog.get("epochs") != 100 or catalog.get("num_classes") != 19
            or catalog.get("data", {}).get("batch_size") != 16
            or catalog.get("optimizer") != {"name": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4}
            or catalog.get("loss") != {"name": "CrossEntropyLoss", "ignore_index": 19}
            or catalog.get("early_stopping") != {"enabled": True, "patience": 15, "min_delta": 0.001}
            or catalog.get("schedulers", {}).get("polynomial") != {
                "name": "polynomial", "parameters": {"total_iters": 100, "power": 0.9},
            }):
        raise ValueError("seed, optimizer, loss, scheduler, or Early Stopping differs from approved settings")
    if catalog.get("output_root") != "/home/wilgomoreira/workspace/partilha_remota/experiments/phd-segmentation-baselines/pastis_run01_segformer_b1_nvlabs_decoder256":
        raise ValueError("versioned output root differs from the approved isolated location")
    config = build_variant_config("segformer_b1", SEQUENCE[0], catalog=catalog, catalog_path=catalog_path)
    model_path = Path(config["model_config"])
    import yaml
    model_yaml = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    if (model_yaml.get("decoder_profile") != "nvlabs_reference"
            or model_yaml.get("decoder_hidden_size") != 256
            or model_yaml.get("decoder_channels") != 256):
        raise ValueError("SegFormer model config is not the reference-aligned 256-channel decoder")
    snapshot = config["pilot_metadata"].get("implementation_snapshot")
    if not isinstance(snapshot, dict) or len(snapshot.get("sha256", "")) != 64:
        raise ValueError("versioned pilot is missing its source/configuration fingerprint")
    if config["pilot_metadata"].get("calibration_and_test_access") is not False:
        raise ValueError("pilot must explicitly prohibit calibration and test access")

    dataset_path = Path(catalog["project_root"]) / catalog["dataset_config"]
    normalization_cfg = dataset.get("normalization", {})
    stats_path = Path(normalization_cfg.get("statistics_file", ""))
    if not stats_path.is_absolute():
        stats_path = dataset_path.parent / stats_path
    statistics = yaml.safe_load(stats_path.read_text(encoding="utf-8"))
    if (statistics.get("channels") != 10
            or statistics.get("training_folds") != [3, 4, 5]
            or statistics.get("date_selection") != expected_date
            or len(statistics.get("mean", [])) != 10 or len(statistics.get("std", [])) != 10
            or any(not math.isfinite(float(value)) for value in statistics["mean"])
            or any(not math.isfinite(float(value)) or float(value) <= 0 for value in statistics["std"])):
        raise ValueError("run-01 normalization is not finite 10-channel training-only statistics")


def _load_or_create_sequence_state(root: Path) -> dict[str, Any]:
    state_path = root / "sequence_state.json"
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        state = {"status": "running", "model": "segformer_b1", "decoder_profile": "nvlabs_reference",
                 "decoder_channels": 256, "seed": 2401, "sequence": list(SEQUENCE),
                 "conditions": {name: "pending" for name in SEQUENCE}, "condition_results": {}}
        _atomic_json(state_path, state)
        return state
    if not state_path.is_file():
        raise FileExistsError(f"output root exists without a sequence state; refusing reuse: {root}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("sequence") != list(SEQUENCE) or state.get("decoder_channels") != 256:
        raise ValueError("existing sequence state is incompatible")
    return state


def run_sequence(*, execute: bool, catalog_path: Path = CATALOG) -> dict[str, Any]:
    catalog, dataset = _load_catalog(catalog_path)
    _assert_approved_config(catalog, dataset, catalog_path)
    _validate_reference(catalog)
    root = Path(catalog["output_root"])
    plan = []
    for variant in SEQUENCE:
        config = build_variant_config("segformer_b1", variant, catalog=catalog, catalog_path=catalog_path)
        plan.append({"variant": variant, "output": str(Path(config["checkpoint_dir"]).parent)})
    snapshots = {
        json.dumps(build_variant_config("segformer_b1", variant, catalog=catalog, catalog_path=catalog_path)
                   ["pilot_metadata"]["implementation_snapshot"], sort_keys=True)
        for variant in SEQUENCE
    }
    if len(snapshots) != 1:
        raise RuntimeError("conditions do not share one identical implementation fingerprint")
    implementation_snapshot = json.loads(next(iter(snapshots)))
    if not execute:
        return {"status": "dry_run", "catalog": str(catalog_path), "plan": plan}

    state = _load_or_create_sequence_state(root)
    if state.get("implementation_snapshot", implementation_snapshot) != implementation_snapshot:
        raise ValueError("working tree changed since this sequence was prepared; refusing to continue")
    if state.get("status") == "completed":
        for variant in SEQUENCE:
            config = build_variant_config("segformer_b1", variant, catalog=catalog, catalog_path=catalog_path)
            result = validate_condition_artifacts(variant, root / "segformer_b1" / variant, config)
            state.setdefault("condition_results", {})[variant] = result
        _atomic_json(root / "sequence_state.json", state)
        return state
    log_path = root / "sequence.log"
    state["implementation_snapshot"] = implementation_snapshot
    state["status"] = "running"
    _atomic_json(root / "sequence_state.json", state)
    for variant in SEQUENCE:
        variant_root = root / "segformer_b1" / variant
        expected_config = build_variant_config("segformer_b1", variant, catalog=catalog, catalog_path=catalog_path)
        if variant_root.exists():
            if state.get("conditions", {}).get(variant) != "completed_verified":
                raise FileExistsError(f"incomplete condition exists; refusing restart/resume/overwrite: {variant_root}")
            state["condition_results"][variant] = validate_condition_artifacts(variant, variant_root, expected_config)
            continue

        blockers = _gpu_compute_pids()
        if blockers:
            raise RuntimeError(f"GPU is occupied before {variant}; refusing concurrent training; compute PIDs={blockers}")
        state["current_condition"] = variant
        state["conditions"][variant] = "running"
        _atomic_json(root / "sequence_state.json", state)
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts/run_pastis_factorial_pilot.py"),
            "--execute", "--catalog", str(catalog_path),
            "--model", "segformer_b1", "--variant", variant,
        ]
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== START {variant}: {' '.join(command)} ===\n")
            log.flush()
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            log.write(f"=== EXIT {variant}: return_code={result.returncode} ===\n")
        if result.returncode != 0:
            state["conditions"][variant] = "failed"
            state["status"] = "failed"
            state["failure"] = f"single-condition launcher returned {result.returncode}; sequence stopped"
            _atomic_json(root / "sequence_state.json", state)
            raise RuntimeError(state["failure"])
        try:
            validation = validate_condition_artifacts(variant, variant_root, expected_config)
        except Exception as exc:
            state["conditions"][variant] = "failed_validation"
            state["status"] = "failed"
            state["failure"] = f"checkpoint validation failed for {variant}: {exc}"
            _atomic_json(root / "sequence_state.json", state)
            raise
        state.setdefault("condition_results", {})[variant] = validation
        state["conditions"][variant] = "completed_verified"
        _atomic_json(root / "sequence_state.json", state)

    state["current_condition"] = None
    state["status"] = "completed"
    state["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_json(root / "sequence_state.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args()
    result = run_sequence(execute=args.execute, catalog_path=args.catalog)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
