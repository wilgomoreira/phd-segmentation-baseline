"""Manifest-driven four-channel loader and spatial split utilities for 5B-Pixels."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from .base import ArrayManifestDataset, read_jsonl
from .preprocessing import Normalization, channel_first, load_array, resolve_relative


class FiveBillionPixelsDataset(ArrayManifestDataset):
    """Load four-band Gaofen-2 samples from an explicit JSONL manifest.

    Native package filenames/layouts are not assumed. Every row supplies image,
    mask, and image_layout fields; optional keys identify arrays in NPZ files.
    The four channels are preserved in their source order.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        root: str | Path | None = None,
        split: str | None = None,
        normalization: Normalization | None = None,
        ignore_index: int | None = None,
        tile_size: int | tuple[int, int] | None = None,
        tile_stride: int | tuple[int, int] | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        records = read_jsonl(manifest_path)
        self.dataset_root = Path(root) if root is not None else manifest_path.parent
        super().__init__(
            records,
            base_dir=self.dataset_root,
            normalization=normalization,
            ignore_index=ignore_index,
            split=split,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        for record, _ in self.records:
            image = load_array(resolve_relative(self.base_dir, record["image"]), key=record.get("image_key"), mmap=True)
            chw = channel_first(image, record["image_layout"])
            if chw.shape[0] != 4:
                raise ValueError(f"Five-Billion-Pixels input must have four channels, got {chw.shape[0]}")


def assign_spatial_grid_splits(
    records: list[dict[str, Any]],
    *,
    block_size: float,
    fractions: dict[str, float],
    seed: int,
    x_key: str = "x",
    y_key: str = "y",
    guard_cells: int = 1,
) -> list[dict[str, Any]]:
    """Assign complete coordinate grid cells to splits and purge split borders.

    Coordinates must be in a suitable projected CRS and describe each
    pre-extracted patch/window center. Cells are assigned as units. A cell is
    marked ``excluded`` when it lies within ``guard_cells`` (Chebyshev distance)
    of a cell assigned to another split. Use a block size at least as large as
    the maximum patch footprint; this utility does not transform coordinates.
    """
    if not math.isfinite(block_size) or block_size <= 0:
        raise ValueError("block_size must be finite and positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(guard_cells, int) or guard_cells < 0:
        raise ValueError("guard_cells must be a nonnegative integer")
    if not fractions or any(not math.isfinite(v) or v <= 0 for v in fractions.values()):
        raise ValueError("split fractions must be finite and positive")
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("split fractions must sum to 1")
    if not records:
        raise ValueError("at least one spatial record is required")
    if any("split" in row for row in records):
        raise ValueError("refusing to overwrite existing split metadata; pass only records needing a new split")
    cells: dict[tuple[int, int], list[int]] = {}
    for i, row in enumerate(records):
        if x_key not in row or y_key not in row:
            raise ValueError(f"record needs projected coordinates {x_key!r} and {y_key!r}")
        x, y = float(row[x_key]), float(row[y_key])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("spatial coordinates must be finite")
        cell = (math.floor(x / block_size), math.floor(y / block_size))
        cells.setdefault(cell, []).append(i)
    ordered_cells = sorted(cells)
    # Contiguous spatial bands, rather than random cells, avoid scattered split
    # assignments that a geographic guard would otherwise purge almost wholly.
    names = list(fractions)
    random.Random(seed).shuffle(names)
    exact = [fractions[name] * len(ordered_cells) for name in names]
    counts = [math.floor(v) for v in exact]
    remainder_order = sorted(range(len(names)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in remainder_order[: len(ordered_cells) - sum(counts)]:
        counts[i] += 1
    cell_split: dict[tuple[int, int], str] = {}
    cursor = 0
    for name, count in zip(names, counts):
        for cell in ordered_cells[cursor : cursor + count]:
            cell_split[cell] = name
        cursor += count
    excluded: set[tuple[int, int]] = set()
    for cell, assigned in cell_split.items():
        for dx in range(-guard_cells, guard_cells + 1):
            for dy in range(-guard_cells, guard_cells + 1):
                other = (cell[0] + dx, cell[1] + dy)
                if other in cell_split and cell_split[other] != assigned:
                    excluded.add(cell)
                    excluded.add(other)
    result = [dict(row) for row in records]
    for cell, indices in cells.items():
        split_name = "excluded" if cell in excluded else cell_split[cell]
        for index in indices:
            result[index]["split"] = split_name
            result[index]["spatial_block_id"] = f"{cell[0]}:{cell[1]}"
    return result
