# Supervised training protocol

## Scope and data separation

`python -m src.training.runner --config configs/training.yaml` runs supervised
multiclass segmentation with the existing PASTIS or Five-Billion-Pixels loader,
one of the existing model implementations, and the streaming evaluator. It
constructs only **train** and **validation** datasets. There is deliberately no
test-loader argument or test evaluation in the training pipeline. Validation
mIoU selects `best.pt`; the test partition must remain untouched until model
selection and configuration are complete.

For PASTIS, configure disjoint `train_folds` and `validation_folds` explicitly.
The official Fold labels are not implicitly treated as train/validation/test.
Any held-out fold must be omitted from both lists. For Five-Billion-Pixels,
configure different manifest labels under `splits.train` and
`splits.validation`; labels `test` and `testing` are rejected for either role.
The manifest may contain test records, but the loaders filter them out before
training or validation. Do not configure the test label as either split.

## Input, loss, and metric semantics

Each batch follows the dataset interface: float image `[B,C,H,W]`, integer
mask `[B,H,W]`, boolean `valid_mask` `[B,H,W]`, and metadata. The configured
dataset channel count is checked against the model config. Class count is a
required training setting and must follow a verified label mapping; it is not
inferred from a particular batch.

The loss is multiclass pixelwise cross entropy. The effective valid set is the
intersection of the dataset's `valid_mask` and `mask != ignore_index` when an
ignore index is configured. Invalid target positions are replaced with a safe
placeholder before evaluating unreduced cross entropy and contribute zero to
the loss sum or gradient. Batch losses are normalized by valid-pixel count;
epoch training and validation loss are global valid-pixel means. Batches with
zero valid pixels are skipped; an epoch with no valid pixels fails explicitly.

Validation runs with gradients disabled. Every epoch computes global validation
loss and mIoU. mIoU uses one global confusion matrix and excludes classes with
zero union. Exact protocol-v1 calibration and ranking metrics can run every N
epochs and/or on the final epoch, controlled by the `validation` training
settings. Scheduled full evaluation feeds raw logits, masks, and effective
valid masks to the unchanged `SegmentationMetricAccumulator`. AUROC/FPR95 may
be `null` when only one error class occurs; they are not coerced to arbitrary
values. No histogram approximation is used. See
`docs/evaluation_protocol.md` for mathematical definitions.

## Reproducibility and resume

`seed` initializes Python, NumPy, PyTorch CPU and CUDA generators, plus the
training and validation DataLoader generators. DataLoader workers are seeded
from their PyTorch worker seed. `deterministic: true` disables cuDNN benchmarking
and asks PyTorch to use deterministic algorithms; operations without a
deterministic implementation may raise an error. `device: auto` selects CUDA
only if available; `cpu` and `cuda` can be requested explicitly. No mixed
precision or implicit weight initialization/download is performed.

Each new invocation creates a unique UTC timestamp plus random-suffix directory
under the configured `checkpoint_dir`; a fresh run never reuses existing run
contents. Resuming uses the directory containing the requested checkpoint. A
new run records `run_metadata.json` and `resolved_configuration.json`, including
resolved training, dataset, and model configurations, canonical SHA-256
configuration hashes, Python and available dependency versions, and the Git
source revision when available. If no Git revision is available, that field is
null. Dependencies are reported, not installed or automatically pinned.

At each completed epoch, `latest.pt` is atomically replaced and `best.pt` is
updated only when validation mIoU strictly improves (ties keep the earlier
checkpoint). Each checkpoint stores model, optimizer, optional scheduler,
completed epoch index, full resolved configuration, best validation mIoU, and
Python/NumPy/PyTorch/CUDA/DataLoader RNG state. Resume loads through CPU first so
RNG and DataLoader generator state tensors remain CPU-resident; model and
optimizer state is then copied to the active device. CUDA RNG is restored only
when CUDA is available. Set `resume` to a saved checkpoint path to continue at
the next epoch. Resume requires the same config except that the `resume` field
itself may differ. The checkpoint boundary is the end of an epoch;
interruption during an epoch repeats that epoch from its beginning rather than
restoring a partially consumed iterator.

`metrics.jsonl` contains one record per completed epoch with train and
validation loss, valid-pixel counts, validation mIoU, learning rate, and other
validation metrics only when scheduled full evaluation ran. It contains no
test results.

## Configuration reference

The example `configs/training.yaml` intentionally leaves experiment-specific
values unset. Fill its `null` fields only after the scientific choices below
are settled.

| Parameter | Meaning |
|---|---|
| `seed` | Required integer seed for global and loader randomness. |
| `deterministic` | Request deterministic PyTorch algorithms; may reject unsupported operations. |
| `device` | `auto`, `cpu`, or `cuda`; CUDA availability is checked. |
| `dataset_config` | Path to the selected dataset YAML. |
| `model_config` | Path to the selected model YAML. |
| `num_classes` | Required output class count from the verified label mapping. |
| `epochs` | Required number of complete training epochs. |
| `data.batch_size` | Required positive batch size. |
| `data.num_workers` | DataLoader worker processes; zero uses the main process. |
| `data.pin_memory` | Pin host batches (typically useful with CUDA). |
| `loss.name` | Must be `CrossEntropyLoss`. |
| `loss.ignore_index` | Optional ignored label; must agree with dataset `ignore_index` if both are set. |
| `optimizer.name` | Currently `AdamW`. |
| `optimizer.learning_rate` | Required positive initial learning rate. |
| `optimizer.weight_decay` | Required nonnegative AdamW weight decay. |
| `scheduler.name` | `null`/`none`, `cosine`, `step`, or `plateau`. |
| `scheduler.parameters` | Constructor arguments: cosine requires `T_max`; step requires `step_size` and `gamma`; plateau accepts `ReduceLROnPlateau` options except mode (fixed to maximize mIoU). |
| `metrics.ece_bins` | Equal-width ECE bin count; protocol default is 15. |
| `metrics.adaptive_ece_bins` | Adaptive ECE bin count; protocol default is 15. |
| `metrics.storage` | Exact rank storage, `memory` or `disk`; memory is convenient for small validation sets, disk reduces resident record storage for large sets. |
| `metrics.temp_dir` | Optional directory for disk-backed metric records. |
| `validation.full_metrics_every_epochs` | Optional positive epoch interval for exact full validation metrics; null disables periodic full evaluation. |
| `validation.full_metrics_at_end` | Run exact full validation metrics on the configured final epoch (default true). |
| `checkpoint_dir` | Required output directory for checkpoints and epoch logs. |
| `resume` | Optional path to a compatible latest or best checkpoint. |

Dataset YAML parameters are documented in `docs/dataset_protocol.md`; model
architecture/input parameters are documented in `docs/model_architectures.md`.
Normalization values, ignore index, number of classes, PASTIS fold assignments,
Five-Billion-Pixels split labels, optimizer hyperparameters, scheduler, batch
size, epoch count, checkpoint location, and seed remain unset in the template
because they are experiment-specific or require verified local dataset
metadata. `num_workers: 0`, `pin_memory: false`, `deterministic: true`, and
`device: auto` are operational starting settings, not dataset-derived values.

## Scheduler behavior

- `cosine`: `CosineAnnealingLR`; requires `T_max`; stepped once after each
  validation epoch.
- `step`: `StepLR`; requires `step_size` and `gamma`; stepped once after
  validation.
- `plateau`: `ReduceLROnPlateau(mode="max")`; receives validation mIoU.
- `null`/`none`: no learning-rate scheduler.

Scheduler state is included in checkpoints. If a scheduler is used, its
configuration must remain identical on resume.

## Scientific decisions still required

1. Confirm class count and class-ID mapping independently for each dataset.
2. Choose PASTIS train/validation folds and reserve all held-out test data;
   choose the already documented single-date policy before evaluation.
3. Verify Five-Billion-Pixels official split semantics and choose suitable
   train/validation manifest labels without using its test split for selection.
4. Set train-only normalization, valid/ignore label semantics, batch size,
   optimizer values, scheduler, epoch budget, and checkpoint/log locations.
5. Select an explicit seed and decide whether deterministic-op enforcement is
   required for the intended hardware/software stack.

No real data or pretrained weights are fetched, and this pipeline does not
perform training unless invoked with a completed configuration.
