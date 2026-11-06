# Project state

## Current release

**Frozen Baseline v1.0** establishes supervised semantic-segmentation results
for PASTIS and LandCover.ai v1 using DeepLabV3+ ResNet-50 and SegFormer-B1.
Predictions and reported reliability metrics are uncalibrated. No post-hoc
calibration or knowledge distillation is part of this release. Headline held-out
results and aggregation details are in `docs/baseline_results_v1.md`.

PASTIS uses its five official patch folds and a single-date ten-band Sentinel-2
adaptation. LandCover.ai uses a geographically grouped test set of six scenes.
These are different evaluation protocols and are not directly comparable.

## Repository implementation

- Model wrappers support DeepLabV3+ with ResNet-50 and SegFormer-B1 and return
  full-resolution raw logits. PASTIS and four-channel Five-Billion-Pixels data
  infrastructure exists in `src/`; LandCover.ai experiment results are
  recorded, but a LandCover.ai dataset loader/configuration is not yet present
  in this repository.
- Training supports AdamW, configurable schedulers, deterministic seeding,
  isolated run directories, resumable checkpoints, provenance capture, and
  validation-mIoU checkpoint selection.
- Evaluation implements segmentation, probability-quality, and predictive
  reliability metrics. In this release these are applied to raw model outputs;
  calibration fitting and calibrated results are outside scope.
- `results/baseline_metrics.csv` and `docs/baseline_results_v1.md` contain the
  consolidated Frozen Baseline v1.0 summary. Dataset files, model checkpoints,
  and large generated evaluation artifacts are not bundled in the repository.

## Dataset decisions

- **Selected benchmark datasets:** PASTIS and LandCover.ai v1.
- **Selected architectures:** DeepLabV3+ ResNet-50 and SegFormer-B1.
- **PASTIS protocol:** five official folds; single-date adaptation; spatial
  correlation between patches is acknowledged.
- **LandCover.ai protocol:** geographic grouping with six held-out test scenes;
  pooled pixel metrics are spatially dependent.
- **Five-Billion-Pixels:** exploratory loader/configuration remains in the code
  but it is not one of the selected datasets in Frozen Baseline v1.0.

## Known scope limits

- Full result reproduction requires the locally acquired datasets, checkpoints,
  and experiment artifacts, which are not checked into Git.
- The repository does not yet contain the LandCover.ai loading and experiment
  configuration used to produce its results. Add and document those before
  claiming the full benchmark can be rerun from this repository alone.
- Results describe the stated held-out partitions. PASTIS fold scores are not
  independent-region estimates, and six LandCover.ai test scenes are a limited
  geographic sample.
- The next research phase may evaluate calibration and distillation separately
  after approval of their protocols; no such methods or results are included
  here.
