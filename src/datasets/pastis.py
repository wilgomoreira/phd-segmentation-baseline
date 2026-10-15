"""PASTIS native-layout loader with explicit single-date selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from .base import ArrayManifestDataset
from .preprocessing import Normalization, channel_first, load_array, resolve_relative

PASTIS_VOID_LABEL = 19


@dataclass(frozen=True)
class DateSelection:
    """A prespecified, label-independent selection for each PASTIS sequence.

    ``index`` is zero-based in chronological acquisition order. ``nearest_date``
    chooses the acquisition closest to an ISO or YYYYMMDD date; ties choose the
    earlier date. There is intentionally no default selection.
    """

    strategy: Literal["index", "nearest_date"]
    value: int | str


def _parse_date(value: object) -> date:
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date()
    return date.fromisoformat(text[:10])


def _metadata_dates(value: object) -> list[date]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, (list, tuple)):
        raise ValueError("PASTIS dates-S2 metadata must be a list or mapping of acquisition dates")
    return sorted(_parse_date(item) for item in value)


class PastisSingleDateDataset(ArrayManifestDataset):
    """Load official PASTIS arrays as one multispectral Sentinel-2 date.

    Official source layout used here: ``metadata.geojson``,
    ``DATA_S2/S2_{ID_PATCH}.npy`` (T,C,H,W), and
    ``ANNOTATIONS/TARGET_{ID_PATCH}.npy``. Only the image time dimension is
    indexed; labels do not influence date selection.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        date_selection: DateSelection,
        folds: tuple[int, ...] | list[int] | None = None,
        normalization: Normalization | None = None,
        ignore_index: int | None = None,
        tile_size: int | tuple[int, int] | None = None,
        tile_stride: int | tuple[int, int] | None = None,
    ) -> None:
        if date_selection.strategy not in ("index", "nearest_date"):
            raise ValueError("date strategy must be 'index' or 'nearest_date'")
        if date_selection.strategy == "index" and (
            not isinstance(date_selection.value, int) or isinstance(date_selection.value, bool)
        ):
            raise TypeError("index date selection requires an integer value")
        if date_selection.strategy == "nearest_date" and not isinstance(date_selection.value, str):
            raise TypeError("nearest_date selection requires a date string")
        self.root = Path(root)
        self.date_selection = date_selection
        metadata_path = self.root / "metadata.geojson"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"PASTIS metadata file not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as stream:
            geojson = json.load(stream)
        if not isinstance(geojson.get("features"), list):
            raise ValueError("PASTIS metadata.geojson must contain a FeatureCollection")
        fold_set = set(folds) if folds is not None else None
        rows = []
        for feature in geojson["features"]:
            properties = feature.get("properties", {})
            if "ID_PATCH" not in properties or "Fold" not in properties or "dates-S2" not in properties:
                raise ValueError("PASTIS metadata feature needs ID_PATCH, Fold, and dates-S2 properties")
            patch_id = int(properties["ID_PATCH"])
            fold = int(properties["Fold"])
            if fold_set is not None and fold not in fold_set:
                continue
            dates = _metadata_dates(properties["dates-S2"])
            rows.append({
                "image": f"DATA_S2/S2_{patch_id}.npy",
                "mask": f"ANNOTATIONS/TARGET_{patch_id}.npy",
                "image_layout": "TCHW",
                "metadata": {**properties, "geometry": feature.get("geometry")},
                "patch_id": patch_id,
                "fold": fold,
                "dates": dates,
                "sample_id": f"PASTIS:{patch_id}",
            })
        if not rows:
            raise ValueError("no PASTIS patches selected by folds")
        self.selected_indices: dict[int, tuple[int, date]] = {}
        for row in rows:
            image = load_array(resolve_relative(self.root, row["image"]), mmap=True)
            if image.ndim != 4 or image.shape[1] != 10:
                raise ValueError("official PASTIS S2 arrays must have shape [T,10,H,W]")
            dates = row["dates"]
            if image.shape[0] != len(dates):
                raise ValueError(f"image/date count mismatch for PASTIS patch {row['patch_id']}")
            index = self._select_index(dates)
            self.selected_indices[row["patch_id"]] = (index, dates[index])
            row["metadata"]["selected_acquisition_index"] = index
            row["metadata"]["selected_acquisition_date"] = dates[index].isoformat()
            row["metadata"]["date_selection_strategy"] = date_selection.strategy
            row["metadata"]["date_selection_value"] = date_selection.value
        super().__init__(
            rows,
            base_dir=self.root,
            normalization=normalization,
            ignore_index=ignore_index,
            tile_size=tile_size,
            tile_stride=tile_stride,
            temporal=True,
        )

    def _select_index(self, dates: list[date]) -> int:
        if self.date_selection.strategy == "index":
            index = int(self.date_selection.value)
            if not 0 <= index < len(dates):
                raise IndexError(f"date index {index} is out of range for a {len(dates)}-date sample")
            return index
        requested = _parse_date(self.date_selection.value)
        return min(range(len(dates)), key=lambda i: (abs((dates[i] - requested).days), dates[i]))

    def _load_image(self, record: dict) -> np.ndarray:
        array = load_array(resolve_relative(self.base_dir, record["image"]), mmap=True)
        index, _ = self.selected_indices[record["patch_id"]]
        return channel_first(array[index], "CHW")

    def _load_mask(self, record: dict) -> tuple[np.ndarray, np.ndarray]:
        """Read the semantic plane from PASTIS's three-plane target array.

        The official PASTIS benchmark dataloader defines its semantic target as
        ``TARGET[0]``. The remaining planes are not needed for this semantic
        segmentation interface and are deliberately left untouched.
        """
        path = resolve_relative(self.base_dir, record["mask"])
        annotation = load_array(path, key=record.get("mask_key"), mmap=True)
        if annotation.ndim != 3 or annotation.shape[0] != 3:
            raise ValueError("PASTIS annotation must have shape [3,H,W]")
        mask = annotation[0]
        if not np.issubdtype(mask.dtype, np.integer):
            raise TypeError("PASTIS semantic mask must contain integer class IDs")
        if np.any((mask < 0) | (mask > 19)):
            raise ValueError("PASTIS semantic class IDs must be in the range 0..19")

        # Official PASTIS documentation defines ID 19 as void and excludes
        # void pixels from semantic losses/metrics. ID 0 is valid background.
        valid = mask != PASTIS_VOID_LABEL
        if self.ignore_index is not None:
            valid &= mask != self.ignore_index
        valid_path = record.get("valid_mask")
        if valid_path is not None:
            explicit = load_array(
                resolve_relative(self.base_dir, valid_path),
                key=record.get("valid_mask_key"),
                mmap=True,
            )
            explicit = np.squeeze(explicit)
            if explicit.shape != mask.shape:
                raise ValueError("explicit valid mask must match semantic mask dimensions")
            valid &= explicit.astype(bool)
        return mask, valid
