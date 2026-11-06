# PASTIS SegFormer-B1 reference-aligned decoder pilot

This versioned four-condition comparison uses the NVlabs SegFormerHead fusion
width of 256 while retaining the original 128-channel model and its run
artifacts unchanged.

The model configuration is
`configs/models/segformer_b1_nvlabs.yaml`. It sets decoder profile
`nvlabs_reference`, per-stage MLP width 256, and fusion width 256. The model
projects the four MiT-B1 scales independently, resizes them to C1 resolution,
concatenates C4 through C1, and applies a 1024-to-256 1x1 convolution,
SyncBatchNorm, ReLU, 0.1 dropout, and a 19-class 1x1 classifier. Output logits
are resized to the input resolution. The original
`configs/models/segformer_b1.yaml` continues to select the historical
128-channel, BatchNorm2d, C1-to-C4 implementation.

The catalog `configs/pilots/pastis_run01_segformer_b1_nvlabs256.yaml` keeps the
approved run-01 protocol: 10 Sentinel-2 bands, observation nearest
2019-05-15, training-only run-01 normalization, official training folds 3–5,
validation fold 2 for checkpoint selection/calibration separation, test fold 1
reserved, seed 2401, batch 16, AdamW at 1e-4 with weight decay 1e-4, 100 epochs,
Early Stopping patience 15 and min_delta 0.001, and the existing scratch or
MiT-B1 pretrained initialization crossed with no scheduler or PolynomialLR
(power 0.9, 100 epochs).

The sequential launcher runs only SegFormer-B1 variants, in this order:

1. Scratch + no scheduler
2. Scratch + PolynomialLR
3. Pretrained + no scheduler
4. Pretrained + PolynomialLR

It waits for each condition, verifies the launcher state, metric history,
checkpoint provenance, epoch records, and 256-channel fusion tensor before
continuing. A failed condition stops the sequence. Existing or incomplete
variant directories are never overwritten or silently restarted. It checks
for another CUDA compute process before each new condition. The launcher also
hashes the exact source and configuration files into each run's pilot metadata,
so the uncommitted implementation is fingerprinted alongside the repository
HEAD recorded by the trainer.

The dedicated NFS output root is
`/home/wilgomoreira/workspace/partilha_remota/experiments/phd-segmentation-baselines/pastis_run01_segformer_b1_nvlabs_decoder256`.
The prepared tmux command is:

```bash
tmux new-session -d -s pastis-segformer-b1-nvlabs256 \
  'cd /home/wilgomoreira/workspace/partilha_remota/projects/phd-segmentation-baselines && /home/wilgomoreira/.venvs/phd-segmentation-baselines-py311/bin/python -u scripts/run_pastis_segformer_nvlabs256_sequence.py --execute'
```

The old model profile and eight factorial results under
`pastis_factorial_pilot` are retained as the 128-channel reference and are not
inputs to this sequence.
