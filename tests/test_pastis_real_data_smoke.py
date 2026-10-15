"""Read-only integration smoke test for a locally released PASTIS copy.

Set ``PASTIS_ROOT`` to the existing dataset directory to enable this test.
The test only memory-maps and reads source files; it never writes to the data.
"""

import os
from pathlib import Path

import pytest

from src.datasets import DateSelection, PastisSingleDateDataset


def test_pastis_real_data_read_only_smoke():
    configured_root = os.environ.get("PASTIS_ROOT")
    if not configured_root:
        pytest.skip("set PASTIS_ROOT to run the read-only real-data smoke test")
    root = Path(configured_root).expanduser()
    if not (root / "metadata.geojson").is_file():
        pytest.skip(f"no PASTIS metadata.geojson found under PASTIS_ROOT={root}")

    dataset = PastisSingleDateDataset(
        root,
        date_selection=DateSelection("index", 0),
        folds=[1],
    )
    sample = dataset[0]

    assert tuple(sample["image"].shape) == (10, 128, 128)
    assert tuple(sample["mask"].shape) == (128, 128)
    assert tuple(sample["valid_mask"].shape) == (128, 128)
    assert int(sample["mask"].min()) >= 0
    assert int(sample["mask"].max()) <= 19
