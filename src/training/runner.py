"""Command-line entry point that assembles existing dataset/model components."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import build_loaders, load_yaml, make_model, make_optimizer, make_scheduler
from .trainer import SegmentationTrainer, resolve_device, set_seed


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_experiment_directory(base: str | Path, *, resume: str | Path | None = None) -> Path:
    """Create a collision-resistant run directory or reuse the checkpoint's run."""
    if resume:
        checkpoint = Path(resume).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        return checkpoint.parent
    base = Path(base)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    while True:
        path = base / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            continue


def _source_revision(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=3,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "torchvision", "numpy", "PyYAML", "tifffile"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run(config_path: str | Path) -> list[dict[str, Any]]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    seed = config.get("seed")
    num_classes = config.get("num_classes")
    epochs = config.get("epochs")
    if seed is None or num_classes is None or epochs is None:
        raise ValueError("seed, num_classes, and epochs must be explicitly configured")
    set_seed(int(seed), deterministic=bool(config.get("deterministic", True)))
    project_root = config_path.resolve().parent.parent
    dataset_path = Path(config["dataset_config"])
    model_path = Path(config["model_config"])
    if not dataset_path.is_absolute():
        dataset_path = (project_root / dataset_path).resolve()
    if not model_path.is_absolute():
        model_path = (project_root / model_path).resolve()
    dataset_config = load_yaml(dataset_path)
    model_config = load_yaml(model_path)
    train_cfg = config.get("data") or {}
    train_loader, validation_loader = build_loaders(
        dataset_config,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg.get("num_workers", 0)),
        seed=int(seed),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        config_dir=dataset_path.parent,
    )
    model = make_model(model_config, dataset_config, int(num_classes))
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    loss_cfg = config.get("loss") or {}
    if loss_cfg.get("name", "CrossEntropyLoss") != "CrossEntropyLoss":
        raise ValueError("the supervised baseline currently supports CrossEntropyLoss only")
    loss_ignore = loss_cfg.get("ignore_index")
    dataset_ignore = dataset_config.get("ignore_index")
    if loss_ignore is not None and dataset_ignore is not None and loss_ignore != dataset_ignore:
        raise ValueError("loss and dataset ignore_index values must agree")
    ignore_index = loss_ignore if loss_ignore is not None else dataset_ignore
    checkpoint_dir = config.get("checkpoint_dir")
    if not checkpoint_dir:
        raise ValueError("checkpoint_dir must be configured")
    checkpoint_base = Path(checkpoint_dir)
    if not checkpoint_base.is_absolute():
        checkpoint_base = (project_root / checkpoint_base).resolve()
    checkpoint_dir = create_experiment_directory(checkpoint_base, resume=config.get("resume"))
    hashed_training_config = dict(config)
    hashed_training_config.pop("resume", None)
    resolved = {
        "training": config,
        "resolved_training": {
            **config,
            "experiment_directory": str(checkpoint_dir.resolve()),
            "selected_device": str(resolve_device(config.get("device", "auto"))),
            "dataset_config_path": str(dataset_path),
            "model_config_path": str(model_path),
        },
        "dataset": dataset_config,
        "model": model_config,
        "config_hashes": {
            "training": _canonical_hash(hashed_training_config),
            "dataset": _canonical_hash(dataset_config),
            "model": _canonical_hash(model_config),
        },
        "python_version": sys.version,
        "dependencies": _dependency_versions(),
        "source_revision": _source_revision(project_root),
    }
    resolved["configuration_hash"] = _canonical_hash({
        "training": hashed_training_config, "dataset": dataset_config, "model": model_config,
    })
    metadata_path = checkpoint_dir / "run_metadata.json"
    if metadata_path.exists() and not config.get("resume"):
        raise FileExistsError(f"run metadata already exists: {metadata_path}")
    if not config.get("resume"):
        metadata_path.write_text(json.dumps(resolved, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        (checkpoint_dir / "resolved_configuration.json").write_text(
        json.dumps({"training": config, "resolved_training": resolved["resolved_training"],
                    "dataset": dataset_config, "model": model_config},
                       indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
    trainer = SegmentationTrainer(
        model, optimizer, train_loader, validation_loader,
        num_classes=int(num_classes), epochs=int(epochs),
        checkpoint_dir=checkpoint_dir, config=resolved,
        device=config.get("device", "auto"), ignore_index=ignore_index,
        scheduler=scheduler, metric_options=config.get("metrics") or {},
        full_metrics_every_epochs=(config.get("validation") or {}).get("full_metrics_every_epochs"),
        full_metrics_at_end=(config.get("validation") or {}).get("full_metrics_at_end", True),
        early_stopping=config.get("early_stopping"),
    )
    if config.get("resume"):
        trainer.resume(config["resume"])
    return trainer.fit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised segmentation baseline")
    parser.add_argument("--config", required=True, help="path to a completed training YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
