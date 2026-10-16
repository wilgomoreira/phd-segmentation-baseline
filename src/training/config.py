"""YAML configuration loading and dataset/model/optimizer construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import json

import yaml
import torch
from torch.utils.data import DataLoader

from src.datasets import (
    DateSelection,
    FiveBillionPixelsDataset,
    Normalization,
    PastisSingleDateDataset,
    collate_segmentation_batch,
)
from src.models import build_model
from .pastis_protocol import plan_pastis_partitions


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root in {path} must be a mapping")
    return value


def _normalization(
    config: dict[str, Any] | None,
    *,
    config_dir: str | Path | None = None,
    training_folds: list[int] | None = None,
    date_selection: dict[str, Any] | None = None,
) -> Normalization:
    config = config or {}
    if config.get("statistics_file"):
        if config.get("mean") is not None or config.get("std") is not None:
            raise ValueError("normalization statistics_file cannot be combined with inline mean/std")
        if config.get("scale") is not None:
            raise ValueError("training statistics are fitted in raw S2 units; scale must be null")
        stats_path = Path(config["statistics_file"])
        if not stats_path.is_absolute() and config_dir is not None:
            stats_path = Path(config_dir) / stats_path
        statistics = load_yaml(stats_path)
        if statistics.get("channels") != 10 or len(statistics.get("mean", [])) != 10 or len(statistics.get("std", [])) != 10:
            raise ValueError("PASTIS normalization statistics must contain ten mean/std values")
        if training_folds is not None and statistics.get("training_folds") != sorted(training_folds):
            raise ValueError("normalization statistics were not fitted on this run's training folds")
        if date_selection is not None and statistics.get("date_selection") != date_selection:
            raise ValueError("normalization statistics use a different single-date selection")
        return Normalization(scale=config.get("scale"), mean=tuple(statistics["mean"]), std=tuple(statistics["std"]))
    return Normalization(scale=config.get("scale"), mean=config.get("mean"), std=config.get("std"))


def build_dataset(config: dict[str, Any], role: str, *, config_dir: str | Path | None = None):
    """Build one train or validation dataset; this factory has no test role."""
    if role not in ("train", "validation"):
        raise ValueError("only train and validation datasets can be built by this pipeline")
    name = config.get("dataset")
    norm = _normalization(
        config.get("normalization"), config_dir=config_dir,
        training_folds=config.get("train_folds") if name == "pastis" else None,
        date_selection=config.get("single_date") if name == "pastis" else None,
    )
    tile = config.get("tiling") or {}
    tile_size, tile_stride = tile.get("size"), tile.get("stride")
    ignore_index = config.get("ignore_index")
    if name == "pastis":
        folds = config.get(f"{role}_folds")
        if not isinstance(folds, list) or not folds:
            raise ValueError(f"PASTIS requires explicit nonempty {role}_folds")
        selection_tiles = config.get("checkpoint_selection_tiles") if role == "validation" else None
        if role == "validation" and selection_tiles:
            metadata = json.loads((Path(config["root"]) / "metadata.geojson").read_text(encoding="utf-8"))
            patch_ids = [
                int(feature["properties"]["ID_PATCH"])
                for feature in metadata["features"]
                if int(feature["properties"]["Fold"]) in set(folds)
                and feature["properties"].get("TILE") in set(selection_tiles)
            ]
        else:
            patch_ids = None
        date_cfg = config.get("single_date") or {}
        strategy, value = date_cfg.get("strategy"), date_cfg.get("value")
        if strategy is None or value is None:
            raise ValueError("PASTIS date selection must be explicitly configured")
        return PastisSingleDateDataset(
            config["root"], date_selection=DateSelection(strategy, value), folds=folds, patch_ids=patch_ids,
            normalization=norm, ignore_index=ignore_index,
            tile_size=tile_size, tile_stride=tile_stride,
        )
    if name == "five_billion_pixels":
        split_names = config.get("splits") or {}
        split = split_names.get(role)
        if not isinstance(split, str) or not split:
            raise ValueError(f"Five-Billion-Pixels requires an explicit {role} split name")
        if split.strip().lower() in {"test", "testing"}:
            raise ValueError("test split is unavailable to model training and selection")
        return FiveBillionPixelsDataset(
            config["manifest"], root=config.get("root"), split=split,
            normalization=norm, ignore_index=ignore_index,
            tile_size=tile_size, tile_stride=tile_stride,
        )
    raise ValueError(f"unsupported dataset configuration {name!r}")


def build_calibration_dataset(dataset_config: dict[str, Any], *, config_dir: str | Path | None = None):
    """Build only the reserved validation calibration subset for post-fit use.

    The training runner never calls this function. There is deliberately no
    test-dataset factory in the training or calibration pipeline.
    """
    if dataset_config.get("dataset") != "pastis" or "test_fold" not in dataset_config:
        raise ValueError("calibration datasets require an explicit PASTIS five-fold protocol")
    partitions = plan_pastis_partitions(
        Path(dataset_config["root"]) / "metadata.geojson", dataset_config,
    )
    return PastisSingleDateDataset(
        dataset_config["root"],
        date_selection=DateSelection(
            dataset_config["single_date"]["strategy"], dataset_config["single_date"]["value"],
        ),
        folds=[dataset_config["validation_fold"]],
        patch_ids=partitions["calibration"],
        normalization=_normalization(
            dataset_config.get("normalization"), config_dir=config_dir,
            training_folds=dataset_config.get("train_folds"),
            date_selection=dataset_config.get("single_date"),
        ),
        ignore_index=dataset_config.get("ignore_index"),
    )


def build_loaders(
    dataset_config: dict[str, Any], *, batch_size: int, num_workers: int,
    seed: int, pin_memory: bool, config_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if dataset_config.get("dataset") == "pastis":
        train_folds = dataset_config.get("train_folds")
        validation_folds = dataset_config.get("validation_folds")
        if not isinstance(train_folds, list) or not isinstance(validation_folds, list):
            raise ValueError("PASTIS train_folds and validation_folds must both be explicitly configured")
        if set(train_folds) & set(validation_folds):
            raise ValueError("PASTIS train and validation folds must be disjoint")
        if "test_fold" in dataset_config:
            plan_pastis_partitions(Path(dataset_config["root"]) / "metadata.geojson", dataset_config)
    elif dataset_config.get("dataset") == "five_billion_pixels":
        splits = dataset_config.get("splits") or {}
        if not splits.get("train") or not splits.get("validation"):
            raise ValueError("Five-Billion-Pixels train and validation split names are required")
        if splits["train"] == splits["validation"]:
            raise ValueError("Five-Billion-Pixels train and validation splits must differ")
    train_set = build_dataset(dataset_config, "train", config_dir=config_dir)
    validation_set = build_dataset(dataset_config, "validation", config_dir=config_dir)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, generator=generator,
        worker_init_fn=seed_worker, persistent_workers=False,
        collate_fn=collate_segmentation_batch,
    )
    validation_generator = torch.Generator().manual_seed(seed + 1)
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=seed_worker, persistent_workers=False,
        generator=validation_generator,
        collate_fn=collate_segmentation_batch,
    )
    return train_loader, validation_loader


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python in each DataLoader worker from its torch seed."""
    import random
    import numpy as np

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_model(model_config: dict[str, Any], dataset_config: dict[str, Any], num_classes: int):
    dataset = dataset_config.get("dataset")
    if dataset == "pastis":
        in_channels = dataset_config.get("channels")
    elif dataset == "five_billion_pixels":
        in_channels = dataset_config.get("channel_count")
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    if not isinstance(in_channels, int) or in_channels <= 0:
        raise ValueError("dataset input channel count must be explicitly configured")
    declared = (model_config.get("in_channels_by_dataset") or {}).get(dataset)
    if declared is not None and declared != in_channels:
        raise ValueError("model and dataset channel configurations disagree")
    return build_model(model_config, num_classes=num_classes, in_channels=in_channels)


def make_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_cfg = config.get("optimizer") or {}
    if optimizer_cfg.get("name") != "AdamW":
        raise ValueError("only the configured AdamW optimizer is supported")
    lr, weight_decay = optimizer_cfg.get("learning_rate"), optimizer_cfg.get("weight_decay")
    if (lr is None or weight_decay is None or not math.isfinite(float(lr))
            or not math.isfinite(float(weight_decay)) or lr <= 0 or weight_decay < 0):
        raise ValueError("AdamW learning_rate and nonnegative weight_decay must be configured")
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))


def make_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    """Create a supported scheduler from explicit settings; return None when disabled."""
    scheduler_cfg = config.get("scheduler") or {"name": "none"}
    name = scheduler_cfg.get("name")
    args = scheduler_cfg.get("parameters") or {}
    if name in (None, "none"):
        return None
    if name == "cosine":
        if "T_max" not in args:
            raise ValueError("cosine scheduler requires T_max")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **args)
    if name == "step":
        if not {"step_size", "gamma"}.issubset(args):
            raise ValueError("step scheduler requires step_size and gamma")
        return torch.optim.lr_scheduler.StepLR(optimizer, **args)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", **args)
    if name == "polynomial":
        total_iters, power = args.get("total_iters"), args.get("power")
        if (not isinstance(total_iters, int) or isinstance(total_iters, bool) or total_iters < 1
                or not isinstance(power, (int, float)) or isinstance(power, bool)
                or not math.isfinite(float(power)) or power <= 0):
            raise ValueError("polynomial scheduler requires positive total_iters and finite positive power")
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_iters, power=float(power),
        )
    raise ValueError(f"unsupported scheduler {name!r}")
