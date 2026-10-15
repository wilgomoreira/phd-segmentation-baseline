"""Explicit normalization and array-loading helpers for remote-sensing data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Normalization:
    """Optional linear scale and per-channel standardization.

    No dataset statistics are supplied by default. ``mean`` and ``std`` must
    be provided together, with one value per channel.
    """

    scale: float | None = None
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        result = np.asarray(image, dtype=np.float32)
        if self.scale is not None:
            result = result * float(self.scale)
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must be supplied together")
        if self.mean is not None and self.std is not None:
            if len(self.mean) != result.shape[0] or len(self.std) != result.shape[0]:
                raise ValueError("normalization mean/std must match image channel count")
            std = np.asarray(self.std, dtype=np.float32)
            if not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("normalization std values must be finite and positive")
            mean = np.asarray(self.mean, dtype=np.float32)
            if not np.isfinite(mean).all():
                raise ValueError("normalization mean values must be finite")
            result = (result - mean[:, None, None]) / std[:, None, None]
        if not np.isfinite(result).all():
            raise ValueError("preprocessed image contains non-finite values")
        return np.ascontiguousarray(result)


def load_array(path: str | Path, *, key: str | None = None, mmap: bool = True) -> np.ndarray:
    """Load an explicit array path; supported formats are NPY, NPZ, and TIFF."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if key is None:
            if len(archive.files) != 1:
                archive.close()
                raise ValueError(f"NPZ file {path} has multiple arrays; specify its key")
            key = archive.files[0]
        if key not in archive.files:
            archive.close()
            raise KeyError(f"array key {key!r} not found in {path}")
        data = np.array(archive[key])
        archive.close()
        return data
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError("reading TIFF requires the optional tifffile package") from exc
        if mmap:
            try:
                return tifffile.memmap(path)
            except (ValueError, OSError):
                pass
        return tifffile.imread(path)
    raise ValueError(f"unsupported array format {suffix!r}; use .npy, .npz, or .tif/.tiff")


def channel_first(array: np.ndarray, layout: str) -> np.ndarray:
    """Convert an explicitly declared CHW/HWC array to CHW."""
    if layout == "CHW":
        result = array
    elif layout == "HWC":
        result = np.moveaxis(array, -1, 0)
    else:
        raise ValueError("image layout must be explicitly 'CHW' or 'HWC'")
    if result.ndim != 3:
        raise ValueError("single-date image must resolve to [C,H,W]")
    return result


def tile_windows(height: int, width: int, tile_size: int | tuple[int, int], stride: int | tuple[int, int] | None = None) -> list[tuple[int, int, int, int]]:
    """Return deterministic, full-size windows; incomplete edge tiles are dropped."""
    if isinstance(tile_size, int):
        th, tw = tile_size, tile_size
    else:
        th, tw = tile_size
    if stride is None:
        sh, sw = th, tw
    elif isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride
    if min(th, tw, sh, sw) <= 0 or sh > th or sw > tw:
        raise ValueError("tile and stride dimensions must be positive and stride cannot exceed tile size")
    if height < th or width < tw:
        return []
    ys = list(range(0, height - th + 1, sh))
    xs = list(range(0, width - tw + 1, sw))
    # Include the final edge-aligned full window when the stride leaves a tail.
    if ys[-1] != height - th:
        ys.append(height - th)
    if xs[-1] != width - tw:
        xs.append(width - tw)
    return [(y, x, y + th, x + tw) for y in ys for x in xs]


def _spatial_shape(array: np.ndarray, layout: str, temporal: bool = False) -> tuple[int, int]:
    if temporal:
        if layout != "TCHW" or array.ndim != 4:
            raise ValueError("temporal PASTIS arrays must be explicitly TCHW")
        return int(array.shape[-2]), int(array.shape[-1])
    image = channel_first(array, layout)
    return int(image.shape[-2]), int(image.shape[-1])


def resolve_relative(base_dir: Path, value: str) -> Path:
    """Resolve a manifest path relative to its manifest/root directory."""
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


