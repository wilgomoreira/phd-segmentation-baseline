#!/usr/bin/env python3
"""Dry-run or explicitly execute exactly one PASTIS run-01 pilot condition."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG = PROJECT_ROOT / "configs/pilots/pastis_run01_factorial.yaml"
MODELS = ("deeplabv3plus_resnet50", "segformer_b1")
VARIANTS = ("scratch_no_scheduler", "scratch_polynomial", "pretrained_no_scheduler", "pretrained_polynomial")


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_snapshot(project_root: Path, files: list[str]) -> dict[str, Any]:
    """Hash the exact source/config inputs for a deliberately uncommitted pilot."""
    digest = hashlib.sha256()
    resolved_files = []
    for relative in sorted(set(files)):
        path = (project_root / relative).resolve()
        path.relative_to(project_root.resolve())
        if not path.is_file():
            raise FileNotFoundError(f"implementation snapshot input is missing: {path}")
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        resolved_files.append(relative)
    return {"sha256": digest.hexdigest(), "files": resolved_files}


def _load_catalog(path: Path = CATALOG) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = _yaml(path)
    project_root = Path(catalog["project_root"]).resolve()
    if project_root != PROJECT_ROOT:
        raise ValueError(f"pilot project_root does not match this checkout: {project_root}")
    dataset_path = _resolve(project_root, catalog["dataset_config"])
    dataset = _yaml(dataset_path)
    expected_date = {"strategy": "nearest_date", "value": "2019-05-15"}
    if (dataset.get("dataset") != "pastis" or dataset.get("channels") != 10
            or dataset.get("num_classes") != 19 or dataset.get("ignore_index") != 19
            or dataset.get("single_date") != expected_date):
        raise ValueError("run-01 dataset config differs from the approved PASTIS protocol")
    if dataset.get("test_fold") != 1 or dataset.get("validation_fold") != 2:
        raise ValueError("pilot must use the existing PASTIS run-01 fold assignment")
    if catalog.get("seed") != 2401 or catalog.get("epochs") != 100 or catalog.get("num_classes") != 19:
        raise ValueError("pilot seed, maximum epochs, and class count must remain approved")
    if catalog["data"].get("batch_size") != 16:
        raise ValueError("pilot batch size must remain 16")
    if catalog["optimizer"] != {"name": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4}:
        raise ValueError("pilot optimizer settings differ from the approved settings")
    if catalog["loss"] != {"name": "CrossEntropyLoss", "ignore_index": 19}:
        raise ValueError("pilot loss settings differ from the approved settings")
    if catalog["early_stopping"] != {"enabled": True, "patience": 15, "min_delta": 0.001}:
        raise ValueError("pilot Early Stopping settings differ from approved settings")
    polynomial = catalog["schedulers"].get("polynomial")
    if polynomial != {"name": "polynomial", "parameters": {"total_iters": 100, "power": 0.9}}:
        raise ValueError("PolynomialLR must use the approved 100-epoch, power-0.9 schedule")
    return catalog, dataset


def _verify_reused_baseline(path: Path) -> None:
    state_path = path.parents[0] / "launcher_state.json"
    if not path.is_dir() or not state_path.is_file():
        raise FileNotFoundError(f"completed scratch baseline is unavailable: {path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (state.get("status") != "completed" or state.get("epoch", -1) < 0
            or state.get("model") != "deeplabv3plus_resnet50" or state.get("run") != 1):
        raise ValueError("DeepLab scratch/no-scheduler reference is not marked completed")
    for name in ("best.pt", "latest.pt", "metrics.jsonl", "run_metadata.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"completed baseline is missing {name}")
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    training = metadata.get("training", {})
    if training.get("seed") != 2401 or training.get("scheduler") != {"name": "none", "parameters": {}}:
        raise ValueError("completed reference does not match the approved scratch/no-scheduler condition")
    if (training.get("optimizer") != {"name": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4}
            or training.get("initialization", {"kind": "scratch"}) != {"kind": "scratch"}):
        raise ValueError("completed reference does not match the approved optimizer or scratch initialization")
    if (training.get("dataset_config") != "configs/datasets/pastis_run_01.yaml"
            or training.get("epochs") != 100
            or metadata.get("dataset", {}).get("single_date") != {"strategy": "nearest_date", "value": "2019-05-15"}
            or metadata.get("model", {}).get("name") != "deeplabv3plus_resnet50"
            or training.get("loss") != {"name": "CrossEntropyLoss", "ignore_index": 19}
            or training.get("early_stopping") != {"enabled": True, "patience": 15, "min_delta": 0.001}):
        raise ValueError("completed reference does not match the approved run-01 scientific settings")


def build_variant_config(
    model_name: str,
    variant_name: str,
    *,
    catalog: dict[str, Any] | None = None,
    catalog_path: str | Path = CATALOG,
) -> dict[str, Any]:
    catalog = catalog or _load_catalog(catalog_path)[0]
    if model_name not in MODELS or variant_name not in VARIANTS:
        raise ValueError("unknown model or pilot variant")
    variant = catalog["variants"][model_name][variant_name]
    if variant["action"] == "reuse_completed_baseline":
        raise ValueError("the existing DeepLab scratch/no-scheduler run is a reference, not a new run")
    project_root = Path(catalog["project_root"]).resolve()
    model_path = _resolve(project_root, catalog["model_configs"][model_name])
    dataset_path = _resolve(project_root, catalog["dataset_config"])
    model_config = _yaml(model_path)
    if model_config.get("pretrained") is not False:
        raise ValueError("model constructors must stay in explicit no-autoload mode")
    initialization_entry = catalog["initializations"][variant["initialization"]]
    if initialization_entry["kind"] == "scratch":
        initialization: dict[str, Any] = {"kind": "scratch"}
    else:
        manifest_path = _resolve(project_root, catalog["pretrained_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        initialization = copy.deepcopy(manifest["sources"][initialization_entry["source"]])
        initialization["kind"] = initialization_entry["kind"]
        checkpoint = Path(initialization["checkpoint_path"])
        if _sha256(checkpoint) != initialization["sha256"]:
            raise ValueError(f"pretrained checkpoint hash mismatch: {checkpoint}")
    output_root = Path(catalog["output_root"]).resolve()
    variant_root = output_root / model_name / variant_name
    pilot_metadata = {
        "catalog": str(Path(catalog_path).resolve()),
        "factorial_variant": variant_name,
        "architecture": model_name,
        "run": 1,
        "calibration_and_test_access": False,
    }
    deep_reference = (catalog.get("variants", {}).get("deeplabv3plus_resnet50", {})
                      .get("scratch_no_scheduler", {}).get("reference"))
    if deep_reference is not None:
        pilot_metadata["scratch_reference"] = deep_reference
    if catalog.get("implementation_snapshot_files"):
        pilot_metadata["implementation_snapshot"] = _implementation_snapshot(
            project_root, catalog["implementation_snapshot_files"],
        )
    return {
        "project_root": str(project_root),
        "name": f"pastis_run01_{model_name}_{variant_name}",
        "seed": catalog["seed"],
        "deterministic": catalog["deterministic"],
        "device": catalog["device"],
        "dataset_config": str(dataset_path),
        "model_config": str(model_path),
        "num_classes": catalog["num_classes"],
        "epochs": catalog["epochs"],
        "data": copy.deepcopy(catalog["data"]),
        "loss": copy.deepcopy(catalog["loss"]),
        "optimizer": copy.deepcopy(catalog["optimizer"]),
        "scheduler": copy.deepcopy(catalog["schedulers"][variant["scheduler"]]),
        "early_stopping": copy.deepcopy(catalog["early_stopping"]),
        "initialization": initialization,
        "pilot_metadata": pilot_metadata,
        "checkpoint_dir": str(variant_root / "checkpoints"),
        "resume": None,
    }


def _validate_reference(catalog: dict[str, Any]) -> None:
    deep_variants = catalog.get("variants", {}).get("deeplabv3plus_resnet50")
    if not deep_variants:
        return
    reference_variant = deep_variants.get("scratch_no_scheduler", {})
    if reference_variant.get("action") == "reuse_completed_baseline":
        _verify_reused_baseline(Path(reference_variant["reference"]))


def _run_one(
    model_name: str,
    variant_name: str,
    *,
    execute: bool,
    catalog: dict[str, Any],
    catalog_path: str | Path = CATALOG,
) -> int:
    variant = catalog["variants"][model_name][variant_name]
    if variant["action"] == "reuse_completed_baseline":
        _verify_reused_baseline(Path(variant["reference"]))
        print(f"REUSE {model_name}/{variant_name}: {variant['reference']}")
        return 0
    config = build_variant_config(
        model_name, variant_name, catalog=catalog, catalog_path=catalog_path,
    )
    variant_root = Path(config["checkpoint_dir"]).parent
    if variant_root.exists():
        raise FileExistsError(f"pilot output already exists; refusing overwrite or implicit resume: {variant_root}")
    command = [sys.executable, "-m", "src.training.runner", "--config", str(variant_root / "training.yaml")]
    print(f"PLAN {model_name}/{variant_name}: {' '.join(command)}")
    if not execute:
        return 0
    variant_root.mkdir(parents=True, exist_ok=False)
    config_path = variant_root / "training.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (variant_root / "launcher_state.json").write_text(
        json.dumps({"status": "running", "model": model_name, "variant": variant_name, "seed": 2401}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (variant_root / "launcher.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    state = {"status": "completed" if result.returncode == 0 else "failed",
             "model": model_name, "variant": variant_name, "seed": 2401,
             "return_code": result.returncode}
    (variant_root / "launcher_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="validate and print commands without creating files")
    mode.add_argument("--execute", action="store_true", help="execute exactly one selected variant")
    parser.add_argument("--catalog", type=Path, default=CATALOG,
                        help="pilot catalog; defaults to the original eight-condition catalog")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    if args.execute and (not args.model or not args.variant):
        parser.error("--execute requires exactly one --model and one --variant selection")
    if (args.model is None) != (args.variant is None):
        parser.error("--model and --variant must be supplied together")
    catalog, _ = _load_catalog(args.catalog)
    _validate_reference(catalog)
    selections = [(args.model, args.variant)] if args.model else [
        (model, variant) for model in MODELS for variant in VARIANTS
    ]
    statuses = []
    for model, variant in selections:
        statuses.append(_run_one(
            model, variant, execute=args.execute, catalog=catalog, catalog_path=args.catalog,
        ))
        if statuses[-1]:
            return statuses[-1]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
