# Supervised Semantic Segmentation Baselines

Reproducible supervised baselines for remote-sensing semantic segmentation. The
current benchmark uses **PASTIS** (Sentinel-2 crop segmentation) and
**LandCover.ai v1** (aerial RGB land-cover segmentation), with DeepLabV3+
(ResNet-50) and SegFormer-B1.

The released reference is **Frozen Baseline v1.0**. It reports raw, uncalibrated
predictions: no post-hoc calibration or knowledge distillation has been applied.
The PASTIS evaluation uses its five official patch folds; LandCover.ai uses six
geographically held-out scenes. These protocols have different sampling units
and their scores should not be compared across datasets.

## Headline held-out results

| Dataset / protocol | Model and selected setup | mIoU | Pixel accuracy | Macro F1 |
|---|---|---:|---:|---:|
| PASTIS, five-fold mean | DeepLabV3+ ResNet-50, scratch, fixed LR | 27.46% | 68.85% | 38.00% |
| PASTIS, five-fold mean | SegFormer-B1, fine-tuned, fixed LR | 28.94% | 70.46% | 39.93% |
| LandCover.ai, six-scene pixel pooled | DeepLabV3+ ResNet-50, fine-tuned, fixed LR | 73.57% | 93.09% | 83.89% |
| LandCover.ai, six-scene pixel pooled | SegFormer-B1, fine-tuned, fixed LR | 73.86% | 92.00% | 84.25% |

PASTIS values are the arithmetic mean across official fold results (the full
report also gives sample standard deviations). LandCover.ai values are pooled
over valid pixels from six geographic test scenes. Full uncalibrated predictive
quality and reliability metrics are in
[`docs/baseline_results_v1.md`](docs/baseline_results_v1.md), with a machine-
readable table at [`results/baseline_metrics.csv`](results/baseline_metrics.csv).

## Project contents

- `src/`: dataset, model, training, and metric implementations.
- `configs/`: model and PASTIS experiment configurations.
- `scripts/`: experiment launchers and protocol preparation.
- `docs/`: dataset, model, training, evaluation, and frozen-results notes.
- `tests/`: unit and synthetic-data integration checks.

Dataset data, trained checkpoints, and generated run outputs are not stored in
this repository. See the protocol documents before reproducing the results.

## Dataset references

- **PASTIS:** Vivien Sainte Fare Garnot and Loïc Landrieu, “Panoptic
  Segmentation of Satellite Image Time Series with Convolutional Temporal
  Attention Networks,” *ICCV*, 2021. [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Garnot_Panoptic_Segmentation_of_Satellite_Image_Time_Series_With_Convolutional_Temporal_ICCV_2021_paper.pdf) ·
  [Official benchmark repository and dataset instructions](https://github.com/VSainteuf/pastis-benchmark).
- **LandCover.ai:** Adrian Boguszewski et al., “LandCover.ai: Dataset for
  Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial
  Imagery,” *CVPR Workshops*, 2021, pp. 1102–1110. [Paper](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Boguszewski_LandCover.ai_Dataset_for_Automatic_Mapping_of_Buildings_Woodlands_Water_and_CVPRW_2021_paper.pdf) ·
  [Dataset website](https://landcover.ai/).

The results here use a single-date semantic-segmentation adaptation of PASTIS
and a geographically grouped evaluation protocol for LandCover.ai; consult the
project protocol and results documents for those adaptations. Cite the original
dataset publications when using the datasets.
