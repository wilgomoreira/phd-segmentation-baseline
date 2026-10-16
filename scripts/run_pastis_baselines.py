#!/usr/bin/env python3
"""Sequential, resumable launcher for the ten approved PASTIS baseline runs.

This script starts scientific training only when invoked without --dry-run.
Existing run directories are never reused unless their launcher state and
checkpoint/log files prove that they belong to this exact configuration.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
BASE_TRAINING_CONFIG = CONFIG_DIR / "training.yaml"
DEFAULT_OUTPUT_ROOT = Path(
    "/home/wilgomoreira/workspace/partilha_remota/experiments/"
    "phd-segmentation-baselines/pastis_full_runs"
)
MODEL_CONFIGS = {
    "deeplabv3plus_resnet50": ("deeplabv3plus.yaml", "deeplabv3plus"),
    "segformer_b1": ("segformer_b1.yaml", "segformer_b1"),
}
RUNS = tuple(
    (model_name, run_number)
    for model_name in MODEL_CONFIGS
    for run_number in range(1, 6)
)
MAX_EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 0.001
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EXPECTED_OPTIMIZER = {
    "name": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
}
EXPECTED_LOSS = {"name": "CrossEntropyLoss", "ignore_index": 19}
EXPECTED_SCHEDULER = {"name": "none", "parameters": {}}
EXPECTED_EARLY_STOPPING = {
    "enabled": True,
    "patience": PATIENCE,
    "min_delta": MIN_DELTA,
}


class UnsafeRunState(RuntimeError):
    """Raised when resuming or skipping a run cannot be proven safe."""


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def build_run_config(model_name: str, run_number: int, seed: int, output_root: Path) -> dict[str, Any]:
    base = copy.deepcopy(_read_yaml(BASE_TRAINING_CONFIG))
    dataset_config = CONFIG_DIR / "datasets" / f"pastis_run_{run_number:02d}.yaml"
    model_filename, dataset_model_id = MODEL_CONFIGS[model_name]
    model_config = CONFIG_DIR / "models" / model_filename
    if not dataset_config.is_file() or not model_config.is_file():
        raise FileNotFoundError("required checked-in dataset/model config is missing")
    dataset = _read_yaml(dataset_config)
    model = _read_yaml(model_config)
    if dataset.get("dataset") != "pastis" or dataset.get("channels") != 10:
        raise ValueError(f"unexpected dataset protocol in {dataset_config}")
    if dataset.get("single_date") != {"strategy": "nearest_date", "value": "2019-05-15"}:
        raise ValueError(f"unexpected PASTIS date selection in {dataset_config}")
    if dataset.get("ignore_index") != 19 or dataset.get("num_classes") != 19:
        raise ValueError(f"unexpected PASTIS class semantics in {dataset_config}")
    if model.get("pretrained") is not False:
        raise ValueError(f"{model_config} must explicitly disable pretrained weights")
    if dataset_model_id not in dataset.get("models", []):
        raise ValueError(f"{dataset_config} does not declare {model_name}")
    statistics_file = dataset.get("normalization", {}).get("statistics_file")
    if not statistics_file:
        raise ValueError(f"{dataset_config} does not identify training-only normalization")
    statistics_path = (dataset_config.parent / statistics_file).resolve()
    statistics = _read_yaml(statistics_path)
    if (statistics.get("training_folds") != sorted(dataset.get("train_folds", []))
            or statistics.get("date_selection") != dataset.get("single_date")):
        raise ValueError(f"normalization provenance does not match {dataset_config}")

    base.update({
        "seed": seed,
        "dataset_config": str(dataset_config.relative_to(PROJECT_ROOT)),
        "model_config": str(model_config.relative_to(PROJECT_ROOT)),
        "num_classes": 19,
        "epochs": MAX_EPOCHS,
        "checkpoint_dir": str(output_root.resolve()),
        "resume": None,
    })
    base["loss"] = dict(base.get("loss") or {}, **EXPECTED_LOSS)
    base["optimizer"] = dict(EXPECTED_OPTIMIZER)
    base["scheduler"] = copy.deepcopy(EXPECTED_SCHEDULER)
    base["early_stopping"] = copy.deepcopy(EXPECTED_EARLY_STOPPING)
    base.setdefault("data", {})["batch_size"] = 16
    if base["epochs"] != MAX_EPOCHS or base["data"]["batch_size"] != 16:
        raise AssertionError("launcher configuration failed to apply approved run limits")
    return base


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _strip_resume(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["resume"] = None
    return result


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise UnsafeRunState(f"required checkpoint is missing: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise UnsafeRunState(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(value, dict):
        raise UnsafeRunState(f"checkpoint is not a mapping: {path}")
    return value


def _validate_metrics(metrics_path: Path, checkpoint_epoch: int, *, repair_tail: bool) -> list[dict[str, Any]]:
    if not metrics_path.is_file():
        raise UnsafeRunState(f"metrics log is missing: {metrics_path}")
    raw = metrics_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            is_partial_final = index == len(lines) - 1 and not line.endswith(b"\n")
            if not (repair_tail and is_partial_final):
                raise UnsafeRunState(f"invalid metrics line {index + 1} in {metrics_path}") from error
            lines = lines[:index]
            break
        if not isinstance(record, dict):
            raise UnsafeRunState(f"invalid metrics record {index + 1} in {metrics_path}")
        records.append(record)

    # The trainer appends metrics immediately before checkpoint replacement.
    # A crash can therefore leave one complete record beyond latest.pt.
    if len(records) == checkpoint_epoch + 2:
        tail = records[-1]
        if tail.get("epoch") != checkpoint_epoch + 1:
            raise UnsafeRunState("metrics/checkpoint epoch mismatch; refusing automatic repair")
        if not repair_tail:
            raise UnsafeRunState("completed run has metrics beyond its latest checkpoint")
        records = records[:-1]
        lines = lines[:len(records)]
    elif len(records) != checkpoint_epoch + 1:
        raise UnsafeRunState(
            f"metrics contain {len(records)} epochs but latest.pt is epoch {checkpoint_epoch}"
        )
    for expected_epoch, record in enumerate(records):
        if record.get("epoch") != expected_epoch:
            raise UnsafeRunState("metrics epochs are not contiguous from epoch zero")
        score = (record.get("validation") or {}).get("mIoU")
        if score is None or not math.isfinite(float(score)):
            raise UnsafeRunState(f"metrics mIoU is invalid at epoch {expected_epoch}")
    if repair_tail and len(lines) != len(raw.splitlines(keepends=True)):
        _atomic_text(metrics_path, b"".join(lines).decode("utf-8"))
    return records


def verify_run_files(run_root: Path, expected_config: dict[str, Any], *, repair_metrics: bool) -> dict[str, Any]:
    child_dirs = sorted(path for path in run_root.glob("run_*") if path.is_dir())
    if len(child_dirs) != 1:
        raise UnsafeRunState(
            f"{run_root} must contain exactly one trainer run directory; found {len(child_dirs)}"
        )
    checkpoint_dir = child_dirs[0]
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    latest = _load_checkpoint(latest_path)
    best = _load_checkpoint(best_path)
    required_keys = {"model", "optimizer", "scheduler", "epoch", "configuration", "rng_states"}
    if not required_keys.issubset(latest) or not required_keys.issubset(best):
        raise UnsafeRunState("checkpoint is missing model/optimizer/scheduler/configuration state")
    latest_epoch, best_epoch = latest.get("epoch"), best.get("epoch")
    if not isinstance(latest_epoch, int) or not 0 <= latest_epoch < MAX_EPOCHS:
        raise UnsafeRunState("latest checkpoint has an invalid epoch")
    if not isinstance(best_epoch, int) or not 0 <= best_epoch <= latest_epoch:
        raise UnsafeRunState("best checkpoint epoch is inconsistent with latest.pt")
    latest_best_metric = float(latest.get("best_validation_miou", float("nan")))
    best_metric = float(best.get("best_validation_miou", float("nan")))
    if (not math.isfinite(latest_best_metric) or not math.isfinite(best_metric)
            or not math.isclose(latest_best_metric, best_metric, rel_tol=0, abs_tol=1e-12)):
        raise UnsafeRunState("best/latest checkpoints disagree about the best selection mIoU")
    if not isinstance(latest.get("early_stopping"), dict):
        raise UnsafeRunState("latest checkpoint has no Early Stopping state")
    early_state = latest["early_stopping"]
    stopped_flag = early_state.get("stopped")
    patience_count = early_state.get("epochs_without_improvement")
    if (not isinstance(stopped_flag, bool) or not isinstance(patience_count, int)
            or isinstance(patience_count, bool) or patience_count < 0):
        raise UnsafeRunState("latest checkpoint has an invalid Early Stopping state")
    if (patience_count >= PATIENCE) != stopped_flag:
        raise UnsafeRunState("latest checkpoint Early Stopping counter/flag disagree")

    resolved_path = checkpoint_dir / "resolved_configuration.json"
    metadata_path = checkpoint_dir / "run_metadata.json"
    if not resolved_path.is_file() or not metadata_path.is_file():
        raise UnsafeRunState("resolved run metadata is missing")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if _strip_resume(resolved.get("training", {})) != _strip_resume(expected_config):
        raise UnsafeRunState("resolved training config differs from the requested run")
    checkpoint_config = latest["configuration"]
    if not isinstance(checkpoint_config, dict) or _strip_resume(
        checkpoint_config.get("training", {})
    ) != _strip_resume(expected_config):
        raise UnsafeRunState("latest checkpoint configuration differs from the requested run")
    records = _validate_metrics(
        checkpoint_dir / "metrics.jsonl", latest_epoch, repair_tail=repair_metrics,
    )
    recorded_best = max(float(row["validation"]["mIoU"]) for row in records)
    if not math.isclose(recorded_best, latest_best_metric, rel_tol=0, abs_tol=1e-12):
        raise UnsafeRunState("latest checkpoint best mIoU disagrees with metrics history")
    stopped = latest["early_stopping"]["stopped"]
    complete = latest_epoch == MAX_EPOCHS - 1 or stopped
    return {
        "checkpoint_dir": checkpoint_dir,
        "latest": latest_path,
        "epoch": latest_epoch,
        "stopped": stopped,
        "complete": complete,
        "metrics_count": len(records),
    }


def _initialize_root(root: Path, config: dict[str, Any], model_name: str, run_number: int) -> None:
    root.mkdir(parents=True, exist_ok=False)
    config_path = root / "training_config.yaml"
    _atomic_text(config_path, yaml.safe_dump(config, sort_keys=False))
    _atomic_json(root / "launcher_state.json", {
        "schema": 1,
        "model": model_name,
        "run": run_number,
        "config_sha256": canonical_hash(_strip_resume(config)),
        "status": "prepared",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })


def _read_state(root: Path, config: dict[str, Any], model_name: str, run_number: int) -> dict[str, Any]:
    state_path = root / "launcher_state.json"
    config_path = root / "training_config.yaml"
    if not state_path.is_file() or not config_path.is_file():
        raise UnsafeRunState(
            f"output exists without launcher ownership records: {root}; refusing to reuse it"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    saved_config = _read_yaml(config_path)
    if _strip_resume(saved_config) != _strip_resume(config):
        raise UnsafeRunState(f"saved training config differs from requested config: {root}")
    if (state.get("schema") != 1 or state.get("model") != model_name
            or state.get("run") != run_number
            or state.get("config_sha256") != canonical_hash(_strip_resume(config))):
        raise UnsafeRunState(f"launcher state does not match requested run: {root}")
    return state


def _write_state(root: Path, state: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    updated = {**state, **extra, "status": status, "updated_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_json(root / "launcher_state.json", updated)
    return updated


def _find_resume(root: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    children = sorted(path for path in root.glob("run_*") if path.is_dir())
    if not children:
        return None
    verified = verify_run_files(root, config, repair_metrics=True)
    if verified["complete"]:
        return {"complete": True, **verified}
    return {"complete": False, **verified}


def _runner_config_path(key: str, config: dict[str, Any]) -> Path:
    path = CONFIG_DIR / f".launcher_{key}.yaml"
    if path.exists():
        old = _read_yaml(path)
        if _strip_resume(old) != _strip_resume(config):
            raise UnsafeRunState(f"stale temporary runner config differs: {path}")
    _atomic_text(path, yaml.safe_dump(config, sort_keys=False))
    return path


def _run_one(
    model_name: str,
    run_number: int,
    seed: int,
    output_root: Path,
    *,
    dry_run: bool,
) -> str:
    key = f"{model_name}_run_{run_number:02d}"
    root = output_root / model_name / f"run_{run_number:02d}"
    config = build_run_config(model_name, run_number, seed, root)
    if dry_run:
        print(f"PLAN {key}: output={root}; seed={seed}; optimizer=AdamW lr=1e-4 wd=1e-4")
        return "planned"

    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        _initialize_root(root, config, model_name, run_number)
    allowed_root_files = {"launcher_state.json", "training_config.yaml", "launcher.log"}
    unexpected = {
        item.name for item in root.iterdir()
        if item.is_file() and item.name not in allowed_root_files
        and not item.name.endswith(".tmp")
    }
    if unexpected:
        raise UnsafeRunState(f"unexpected files in launcher-owned run root {root}: {sorted(unexpected)}")
    state = _read_state(root, config, model_name, run_number)
    if state.get("status") == "completed":
        verified = verify_run_files(root, config, repair_metrics=False)
        if not verified["complete"]:
            raise UnsafeRunState(f"run marked completed but did not reach a valid stopping point: {root}")
        print(f"SKIP verified complete {key} at epoch {verified['epoch']}")
        return "skipped"

    resume = _find_resume(root, config)
    if resume and resume["complete"]:
        _write_state(root, state, "completed", latest=str(resume["latest"]))
        print(f"SKIP verified complete {key} at epoch {resume['epoch']}")
        return "skipped"
    if resume:
        config["resume"] = str(resume["latest"])
        print(f"RESUME {key} from {resume['latest']} (epoch {resume['epoch']})")
    elif state.get("status") not in {"prepared", "failed", "running"}:
        raise UnsafeRunState(f"unexpected launcher state {state.get('status')!r} in {root}")
    elif state.get("status") in {"failed", "running"}:
        raise UnsafeRunState(
            f"{root} has no verifiable latest.pt after status {state.get('status')!r}; "
            "refusing to restart or overwrite it"
        )

    # Resume from latest only. A run directory with partial or inconsistent
    # artifacts is rejected by verify_run_files rather than guessed at.
    _write_state(root, state, "running", latest=str(resume["latest"]) if resume else None)
    log_path = root / "launcher.log"
    runner_config = _runner_config_path(key, config)
    command = [
        sys.executable, "-u", "-m", "src.training.runner",
        "--config", str(runner_config),
    ]
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] command: {' '.join(command)}\n")
            log.flush()
            result = subprocess.run(
                command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
    finally:
        runner_config.unlink(missing_ok=True)
    if result.returncode != 0:
        _write_state(root, state, "failed", exit_code=result.returncode)
        raise RuntimeError(
            f"{key} runner exited {result.returncode}; preserved output at {root}. "
            "A later invocation will resume only if latest.pt is verifiable."
        )
    verified = verify_run_files(root, _strip_resume(config), repair_metrics=False)
    if not verified["complete"]:
        _write_state(root, state, "failed", latest=str(verified["latest"]))
        raise UnsafeRunState(
            f"{key} runner exited successfully but reached neither epoch 100 nor Early Stopping"
        )
    _write_state(root, state, "completed", latest=str(verified["latest"]), epoch=verified["epoch"])
    print(f"COMPLETE {key} at epoch {verified['epoch']} (early_stopped={verified['stopped']})")
    return "completed"


def run_sequence(
    seed: int,
    output_root: Path,
    *,
    dry_run: bool = False,
    model_name: str | None = None,
    run_number: int | None = None,
) -> list[str]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (model_name is None) != (run_number is None):
        raise ValueError("--model and --run must be supplied together for single-run selection")
    if model_name is None:
        selected_runs = RUNS
    else:
        selected = (model_name, run_number)
        if selected not in RUNS:
            raise ValueError(f"unsupported model/run selection: {selected}")
        selected_runs = (selected,)
    output_root = output_root.expanduser().resolve()
    if dry_run:
        return [
            _run_one(model, number, seed, output_root, dry_run=True)
            for model, number in selected_runs
        ]
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".launcher.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise UnsafeRunState(f"another launcher owns {output_root}") from error
        results = []
        for model, number in selected_runs:
            results.append(_run_one(model, number, seed, output_root, dry_run=False))
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="explicit reproducibility seed for all ten runs")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="print run plan without creating files or training")
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS), help="select one model; requires --run")
    parser.add_argument("--run", type=int, choices=range(1, 6), help="select one PASTIS run; requires --model")
    args = parser.parse_args()
    results = run_sequence(
        args.seed, args.output_root, dry_run=args.dry_run,
        model_name=args.model, run_number=args.run,
    )
    if args.dry_run:
        print(f"Prepared {len(results)} run configuration(s); no files were created.")
    else:
        print(f"Sequential launcher finished: {results.count('completed')} completed, "
              f"{results.count('skipped')} skipped.")


if __name__ == "__main__":
    main()
