# 05. phd_segmentation_baselines

## Scientific objective

Establish conventional supervised semantic segmentation baselines for remote-sensing imagery. The project will compare segmentation quality and predictive uncertainty across selected datasets and models, with a reproducible evaluation setup to be specified in later work.

## Selected datasets

- **PASTIS** — Sentinel-2, adapted to a single-date setting.
- **Five-Billion-Pixels** — Gaofen-2.

## Selected models

- **DeepLabV3+** with ResNet-50 backbone.
- **SegFormer-B1**.

## Evaluation metrics

- Segmentation and probabilistic quality: mIoU, ECE, AdaECE, NLL, Brier.
- Uncertainty and risk: AUROC, AURC, E-AURC, AUSE, FPR95.

Dataset loaders and evaluation metrics are implemented; segmentation model
wrappers are described in [model_architectures.md](docs/model_architectures.md).
The supervised training pipeline is described in
[training_protocol.md](docs/training_protocol.md). Model class counts and
real-data preprocessing remain pending verification of dataset label
conventions.
