# Dataset protocol and integration status

This document separates facts verified from authoritative dataset sources from
project choices and fields checked against the locally obtained PASTIS
release. The loaders do not infer
unverified metadata, label encodings, band order, normalization values, or
official train/validation/test splits.

## Common sample interface

Both loaders return a dictionary containing:

- `image`: float32 tensor in `[C,H,W]` order.
- `mask`: integer tensor in `[H,W]` order, preserving source class IDs.
- `valid_mask`: boolean tensor `[H,W]`.
- `metadata`: original IDs, fold/split fields, source metadata, and tile/date
  selection fields where available.

Validity is all true unless the dataset defines a void label, an
`ignore_index` is explicitly configured, or a manifest row supplies an
explicit valid-mask file. Class remapping and label changes are not performed.

`Normalization` supports an explicit multiplicative scale and optional
per-channel mean/std. No statistics or scaling are defaulted. Mean and std
must be supplied together; standard deviations must be positive. For
reproducible experiments, statistics must be estimated from the training
partition only, then frozen before validation/test use. No spatial resampling
or band selection is performed.

Supported array sources are NPY, NPZ with an explicit key if it contains
multiple arrays, and TIFF through optional `tifffile`. Axis layout is explicit
(`CHW` or `HWC`; PASTIS time series uses `TCHW`). NPY files are memory-mapped
where possible. Tiling drops undersized images and incomplete edges, adds the
last full edge-aligned window when needed, and can therefore overlap the
previous window at the edge. Tiling must happen after split assignment, or all
tiles from a source image must inherit that image's split.

## PASTIS: verified facts and loader behavior

The original paper and benchmark repository describe PASTIS as Sentinel-2
multispectral time series with 2,433 patches, 128×128 spatial dimensions, 10
spectral bands, 10 m pixels, and variable sequences (38–61 acquisitions in the
benchmark overview). The benchmark source loader reads `metadata.geojson`,
indexes by `ID_PATCH`, uses the `Fold` metadata, reads image arrays from
`DATA_S2/S2_{ID_PATCH}.npy` in `[T,C,H,W]` order, and semantic masks from
`ANNOTATIONS/TARGET_{ID_PATCH}.npy`. The official benchmark defines five folds;
these fold IDs are preserved as metadata and are not relabeled train,
validation, or test by this project. The official target arrays have shape
`[3,H,W]`; the official benchmark dataloader selects `TARGET[0]` as the
per-pixel semantic class map. Its source does not assign meanings needed by
this semantic loader to planes 1 and 2, so they are left unused. Semantic IDs
are 0 for background, 1–18 for crop classes, and 19 for void. Background is a
valid class; void pixels are excluded from semantic loss/metrics via
`valid_mask`. No class remapping is performed.

`PastisSingleDateDataset` requires a `DateSelection`; there is no implicit
first-date or test-derived choice. Supported strategies are:

- `DateSelection("index", i)`: select the zero-based acquisition index in
  chronological date order.
- `DateSelection("nearest_date", "YYYY-MM-DD")`: select the closest date
  declared in the patch metadata; an exact distance tie selects the earlier
  date.

Selection depends only on the prespecified policy and acquisition dates, never
on target labels, model results, or test performance. Choose the same policy
before evaluating held-out data. The chosen index/date and original patch ID,
fold, and metadata properties are returned. Ten channels are retained; no
RGB-only reduction occurs. Optional tiling happens after date selection.

The official code supports selecting an index or nearest calendar date and
documents a reference date for its day-offset interface. This loader uses
calendar dates directly and does not need a reference-date transformation.
The leakage-controlled run configs fit mean/std from the selected single-date
S2 planes in the three training folds only; calibration and test samples are
excluded. The five run-specific files are written under
`configs/normalization/` by the protocol preparation script. Statistics use
pixel-weighted population standard deviation (`ddof=0`) and include date/fold
provenance plus a hash of the training IDs.

The locally inspected metadata has 2,433 unique patch IDs and five official
folds. The folds occur across the same four Sentinel tiles; cross-fold patch
footprints do not overlap, but are not geographically blocked. The nearest
cross-fold footprint gap measured in projected metadata is about 1.28 km.
Treat official-fold results as patch-level performance, not geographically
independent generalization.

## Five-Billion-Pixels: verified facts and loader behavior

The dataset authors' project page and paper describe 150 Gaofen-2 images at
4 m resolution, over 50,000 km², with more than five billion annotated pixels
and a 24-category system. The authors' page states that unclear areas are
unlabeled, provides coordinate information, and says test data for land-cover
mapping were released. The user-approved project scope requires four-channel
input, and `FiveBillionPixelsDataset` enforces exactly four channels while
preserving the order present in each source image.

The accessible authoritative page does not establish the local package's
filenames, four-band ordering, array axis order, exact unlabeled value, or
whether its released test data are a supervised segmentation split appropriate
to this experiment. Therefore the loader is manifest-driven rather than
guessing a native file layout. JSON Lines records explicitly identify `image`,
`mask`, `image_layout` (`CHW` or `HWC`), optional NPZ keys, optional
`valid_mask`, and metadata such as `image_id`, `geographic_id`, `split`, or
coordinates. Paths are relative to the manifest directory unless absolute.
Source band order is preserved, and `band_order` is not hard-coded.

The JSON Lines manifest is a project interface, not a claim about the
publisher's archive layout. A schematic row is:

```json
{"image":"<verified image path>","mask":"<verified mask path>","image_layout":"HWC","image_id":"<source id>","geographic_id":"<region id>","split":"train","x":0.0,"y":0.0,"metadata":{}}
```

`x` and `y` are only needed by the spatial split planner, must use the
documented projected CRS, and are preserved in returned metadata. Replace the
placeholder paths and IDs only after inspecting the local distribution.

`assign_spatial_grid_splits()` is an optional project split planner, not a
replacement for verified official held-out splits. It:

1. Maps projected coordinate centers to a globally anchored grid using a
   researcher-specified block size.
2. Assigns contiguous sorted spatial cells to splits with reproducible
   fractions and seed.
3. Marks cells within a configured Chebyshev guard distance of another split
   as `excluded`.

It refuses to overwrite an existing `split` field. Apply it only to training
or development source records whose split is still undecided; preserve any
official test records. Coordinates must be in a projected CRS with known
units. The block size must be at least the largest source/patch footprint, and
tile windows from a source image must inherit one split. The guard reduces
near-boundary spatial leakage; exact geographic non-overlap still requires
checking source footprints/geometries because center points alone cannot prove
that arbitrary scenes do not overlap.

## Generic template fields still intentionally unset

`configs/datasets/pastis.yaml` remains an unset generic template. The approved
five-run protocol is recorded in `configs/datasets/pastis_run_01.yaml` through
`pastis_run_05.yaml`. `configs/datasets/five_billion_pixels.yaml`
leaves manifest/root, band order, axis layout, normalization, ignore-index,
coordinate CRS/keys, block size, split fractions/seed/guard, and tiling
unresolved. These values require the exact local release and experiment plan.

## Sources

- PASTIS official benchmark repository: <https://github.com/VSainteuf/pastis-benchmark>
- PASTIS paper: Garnot and Landrieu, ICCV 2021,
  <https://openaccess.thecvf.com/content/ICCV2021/papers/Garnot_Panoptic_Segmentation_of_Satellite_Image_Time_Series_With_Convolutional_Temporal_ICCV_2021_paper.pdf>
- PASTIS supplementary dataset description:
  <https://openaccess.thecvf.com/content/ICCV2021/supplemental/Garnot_Panoptic_Segmentation_of_ICCV_2021_supplemental.pdf>
- Five-Billion-Pixels authors' project page:
  <https://x-ytong.github.io/project/Five-Billion-Pixels.html>
- Tong, Xia, and Zhu, “Enabling Country-Scale Land Cover Mapping with
  Meter-Resolution Satellite Imagery,” ISPRS JPRS 2023,
  <https://doi.org/10.1016/j.isprsjprs.2022.12.011>.

## Decisions required before real-data integration

1. Choose the PASTIS index or fixed calendar date before evaluating held-out
   data; state whether one common calendar date or one common sequence index
   is the scientific target.
2. Inspect the downloaded PASTIS metadata and verify acquisition-array order,
   class IDs against the official release, and intended fold use.
3. Obtain and inspect the exact Five-Billion-Pixels release; document its
   file/package format, band names/order, image and label axis order, label
   values, coordinate reference system, source-image IDs, and official split
   meaning.
4. Decide whether the released Five-Billion-Pixels test data are eligible for
   this supervised benchmark. Never use them to choose the PASTIS date or tune
   preprocessing.
5. Define train-only normalization statistics, optional scale, tile size and
   stride, class mapping (if any), ignore index, and spatial split geometry.
6. Verify geographic footprints and buffer widths so patches from different
   splits cannot overlap. Preserve official splits if they are suitable and
   documented.
