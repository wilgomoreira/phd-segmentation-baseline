#!/usr/bin/env python3
"""Validate one fold protocol and fit its train-only PASTIS normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import DateSelection
from src.datasets.pastis_statistics import fit_pastis_training_statistics
from src.training.pastis_protocol import plan_pastis_partitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--statistics-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("protocol YAML must contain a mapping")
    partitions = plan_pastis_partitions(args.dataset_root / "metadata.geojson", config)
    date_config = config.get("single_date") or {}
    selection = DateSelection(date_config.get("strategy"), date_config.get("value"))
    destination = args.statistics_output
    if destination is None:
        destination = config.get("normalization", {}).get("statistics_file")
        if destination is None:
            raise ValueError("provide --statistics-output or normalization.statistics_file")
        destination = Path(destination)
        if not destination.is_absolute():
            destination = args.protocol.parent / destination
    result = fit_pastis_training_statistics(
        args.dataset_root,
        partitions["train"],
        date_selection=selection,
        training_folds=config["train_folds"],
        output_path=destination,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "partitions": {name: len(ids) for name, ids in partitions.items()},
        "statistics_file": str(destination.resolve()),
        "training_patch_count": result["training_patch_count"],
        "pixel_count_per_channel": result["pixel_count_per_channel"],
        "mean": result["mean"],
        "std": result["std"],
    }, indent=2))


if __name__ == "__main__":
    main()
