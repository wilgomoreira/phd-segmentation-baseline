# Project state

## Status

Evaluation protocol v1 remains intact. Synthetic-only dataset infrastructure
for PASTIS single-date inputs and four-channel Five-Billion-Pixels inputs is
implemented. DeepLabV3+ ResNet-50 and SegFormer-B1 model wrappers are implemented
and CPU tested. The training pipeline now has isolated run directories,
configuration provenance, CPU-first checkpoint resume, and scheduled exact
validation metrics while selecting checkpoints by mIoU each epoch. Real-data
integration remains pending local release verification and researcher
configuration.

## Completed decisions

- Objective: supervised semantic segmentation baselines for remote-sensing imagery.
- Datasets: PASTIS (Sentinel-2, single-date adaptation) and Five-Billion-Pixels (Gaofen-2).
- Models: DeepLabV3+ (ResNet-50) and SegFormer-B1.
- Both models return full-resolution raw logits, accept the configured 10- or
  4-channel inputs, and initialize without pretrained-weight downloads.
- DeepLabV3+ uses a dilated ResNet-50 output-stride-16 encoder, ASPP, and a
  low-level feature decoder; SegFormer uses a self-contained PyTorch MiT-B1
  encoder and all-MLP segmentation decoder. Architecture choices are detailed
  in `docs/model_architectures.md`.
- Evaluation: approved v1 metrics and tie conventions remain in `docs/evaluation_protocol.md`; evaluation source files were not modified in this dataset task.
- SegFormer MiT-B1 now includes the four official stage-output LayerNorms,
  epsilon 1e-6, NVlabs encoder initialization, and the B1 decoder dimensions
  (256 projection embedding, 128 fusion channels). Its stage shapes/heads,
  depths, reductions, and backward path are structurally tested. Known
  differences from MMSEG are documented in `docs/model_architectures.md`.
- Model YAML values are checked against the constructed architecture. Supported
  decoder/DeepLab settings are applied; unknown keys and unsupported MiT-B1
  structural edits fail explicitly.
- Training uses multiclass pixelwise cross entropy over the effective valid
  mask, AdamW, configurable supported schedulers, deterministic seed handling,
  epoch-level resume checkpoints, and streaming validation metrics. Global
  loss and mIoU run every validation epoch; exact full metrics can run
  periodically and/or at the end.
- Model selection uses validation mIoU only. The training API builds only
  train/validation loaders and has no test-loader path. Checkpoints include
  model/optimizer/scheduler state, epoch/config, and Python/NumPy/PyTorch/CUDA/
  DataLoader RNG state.
- Common sample output includes image, integer mask, explicit valid-pixel mask, and metadata.
- PASTIS native metadata/array paths and TCHW 10-band time-series convention were checked against the official benchmark source; date selection is explicit and label-independent.
- Five-Billion-Pixels loader uses an explicit manifest, requires four channels, and preserves source channel order.
- Optional configurable normalization, NPY/NPZ/TIFF array reading, tiling, and a deterministic guarded spatial-grid split planner are implemented.
- No real datasets were read or downloaded; no normalization statistics, class remapping, ignore index, or Five-Billion-Pixels band order were fabricated.
- New runs receive unique output directories. Provenance records resolved
  training/dataset/model configurations, SHA-256 hashes, Python/dependency
  versions, selected device, experiment path, and source revision when
  available. Resume loads checkpoint tensors through CPU before restoring
  Python, NumPy, PyTorch CPU/CUDA, and DataLoader generator states.
- CPU test suite: `python -m pytest -q` — 39 passed, including model structure
  and configuration checks, backward pass, RNG/resume consistency, run
  isolation, scheduled validation, masked training, checkpoint selection, and
  synthetic loader/model/evaluator integration.
- Verification in this task used CPU only. CUDA resume and execution have not
  been validated.
- Model runtime dependencies are PyTorch and torchvision; both were already
  available locally. No dependencies were installed. The installed
  Transformers import path was unusable because its SciPy dependency is binary
  incompatible with local NumPy 2.2.6, so SegFormer is implemented without that
  runtime dependency. TIFF reading uses optional `tifffile` when present.

## Decisions required before real-data integration

- Prespecify the PASTIS single-date strategy and value before held-out evaluation; decide whether the target is a shared calendar date or shared sequence index.
- Verify local PASTIS date-array alignment, class IDs, void encoding, and fold use against the downloaded metadata.
- Inspect the Five-Billion-Pixels release to determine paths/container format, four-band semantics/order, axes, label encoding, CRS, source identifiers, and official split meaning.
- Decide whether its published test data are suitable for this supervised baseline; protect all official test records from split planning.
- Set train-only normalization, tiling dimensions/stride, ignore index, class mapping if any, and spatial block/buffer parameters.
- Verify actual scene footprints and geographic separation; the grid planner relies on projected coordinates and a block size covering the largest source footprint.

## Other pending project decisions

- Dataset-specific class mappings, preprocessing details, and benchmark reporting breakdowns after local metadata review.
- Training configuration, optimization, schedules, and dependency version pins.
- The decoder's BatchNorm2d versus MMSEG SyncBN difference may affect exact
  training comparability; choose whether that is acceptable before reporting
  direct published-model comparators. The project has not validated numerical
  equivalence against NVlabs checkpoint weights.
- Real-run training values: class count and mapping, split/fold assignments,
  batch size, learning rate, weight decay, schedule, epoch count, seed,
  normalization/ignore semantics, and output/checkpoint locations.
- Decide whether to use pretrained initialization later; if so, document source
  checkpoint and multispectral first-layer adaptation. Current models reject
  automatic pretrained initialization and use random weights.
- Decide whether the DeepLabV3+ ASPP/decoder should use depthwise-separable
  convolutions; current implementation uses regular convolutions.
