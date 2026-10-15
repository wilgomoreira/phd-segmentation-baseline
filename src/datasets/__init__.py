"""Dataset interfaces and preprocessing for supervised segmentation."""

from .base import ArrayManifestDataset, read_jsonl
from .five_billion_pixels import FiveBillionPixelsDataset, assign_spatial_grid_splits
from .pastis import DateSelection, PastisSingleDateDataset
from .preprocessing import Normalization, load_array, tile_windows

__all__ = [
    "ArrayManifestDataset",
    "DateSelection",
    "FiveBillionPixelsDataset",
    "Normalization",
    "PastisSingleDateDataset",
    "assign_spatial_grid_splits",
    "load_array",
    "read_jsonl",
    "tile_windows",
]
