# PASTIS run-01 pretrained-initialization and scheduler pilot

This is a controlled 2-by-2 pilot for DeepLabV3+ ResNet-50 and SegFormer-B1:

| Initialization | Learning-rate schedule | Pilot treatment |
| --- | --- | --- |
| Scratch | None | Control |
| Scratch | PolynomialLR | Scheduler effect |
| ImageNet encoder | None | Initialization effect |
| ImageNet encoder | PolynomialLR | Combined treatment |

The completed DeepLabV3+ scratch/no-scheduler run 01 is the existing control and
is referenced in the catalog. The pilot launcher never opens it for writing or
uses its checkpoint to initialize another treatment. The remaining seven
conditions each have a separate directory under
`/home/wilgomoreira/workspace/partilha_remota/experiments/phd-segmentation-baselines/pastis_factorial_pilot`.
The launcher accepts exactly one model/variant pair in execution mode and
refuses to reuse an existing output directory. Its default workflow is a dry
run. The existing ten-run scratch launcher and its settings are unchanged.

All conditions use PASTIS run 01, the existing nearest-date selection for
2019-05-15, all ten bands in the configured B2, B3, B4, B5, B6, B7, B8, B8A,
B11, B12 order, and the saved training-fold-only normalization. The run keeps
the approved official folds, 19 classes with ignore index 19, seed 2401,
batch size 16, AdamW at learning rate 0.0001 and weight decay 0.0001,
CrossEntropyLoss, maximum 100 epochs, and Early Stopping patience 15 with
`min_delta: 0.001`. Training and checkpoint selection remain the only loaded
partitions. There is no calibration or test loader in the training runner.

## Encoder checkpoints and adaptation

The checked-in provenance and expected hashes are in
`configs/pretrained/pastis_imagenet_encoders.json`. The downloaded files are
stored on NFS under
`/home/wilgomoreira/workspace/partilha_remota/cache/pretrained/phd-segmentation-baselines`.
At initialization, the runner verifies SHA-256 and loads with PyTorch
`weights_only=True`; it makes no network request and runs no remote model code.

For both models, the RGB kernels are mapped by wavelength: pretrained Red
(channel 0) to Sentinel-2 B4 (input channel 2), Green (channel 1) to B3
(channel 1), and Blue (channel 2) to B2 (channel 0). The other seven input
kernels are zero. All encoder
parameters remain trainable. Encoder keys and tensor shapes must match
completely; missing, unexpected, or mismatched encoder tensors fail before
training. For NVIDIA MiT-B1, the source checkpoint's separate key and value
projection tensors are concatenated in key-then-value order to match this
implementation. The ImageNet classification heads are discarded. The
DeepLab ASPP/decoder/classifier and SegFormer segmentation decoder/classifier
remain randomly initialized. The existing model constructors still use
`pretrained: false`; initialization is explicit in each pilot training config.

The torchvision ResNet-50 V2 checkpoint URL and SHA-256 are pinned in the
manifest. Torchvision's source project is BSD-3-Clause; its official model
weights page does not state a separate license for the binary weight file, so
the manifest records that distinction. The NVIDIA MiT-B1 checkpoint is pinned
to Hugging Face revision
`13ddceec4e8bdf401e7cd7acf5aebc526222518c`; the NVIDIA SegFormer license limits
use to non-commercial research or evaluation. Check institutional terms before
redistribution or other use.

## Polynomial schedule

The polynomial conditions use `PolynomialLR(total_iters=100, power=0.9)` and
step it once after each completed training and checkpoint-selection epoch,
matching the current trainer order. This gives the per-epoch sequence
`lr = 1e-4 * (1 - epoch / 100)^0.9` after each scheduler step, reaching zero
after epoch 100. The scheduler state is saved in both checkpoints and restored
on resume; tests compare resumed and uninterrupted learning-rate sequences.
Early Stopping can end a run before the schedule reaches its tail, as intended
by the approved maximum-epoch and patience settings.

## Dry run and proposed execution commands

Run the full configuration dry run without creating files:

```bash
/home/wilgomoreira/.venvs/phd-segmentation-baselines-py311/bin/python \
  scripts/run_pastis_factorial_pilot.py --dry-run
```

After explicit approval to train, execute one condition at a time. For
example, the first new DeepLabV3+ treatment would be:

```bash
/home/wilgomoreira/.venvs/phd-segmentation-baselines-py311/bin/python \
  scripts/run_pastis_factorial_pilot.py --execute \
  --model deeplabv3plus_resnet50 --variant scratch_polynomial
```

The matching SegFormer-B1 pretrained-plus-polynomial treatment would be:

```bash
/home/wilgomoreira/.venvs/phd-segmentation-baselines-py311/bin/python \
  scripts/run_pastis_factorial_pilot.py --execute \
  --model segformer_b1 --variant pretrained_polynomial
```

The launcher requires both selectors with `--execute`, so one command cannot
start the other conditions. If a selected output directory already exists,
the launcher stops instead of overwriting or implicitly resuming it. This
change set prepares the pilot only; it does not start training.

Sources: [torchvision ResNet-50 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
[NVIDIA MiT-B1 checkpoint and model card](https://huggingface.co/nvidia/mit-b1/tree/13ddceec4e8bdf401e7cd7acf5aebc526222518c),
[SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE),
[PyTorch PolynomialLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html).
