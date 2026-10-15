# Segmentation model architecture decisions

## Shared interface

Both models are constructed with explicit positive `num_classes` and
`in_channels`, accept floating tensors `[B, C, H, W]`, and return raw logits
`[B, num_classes, H, W]`. The PASTIS adaptation uses 10 input channels and
Five-Billion-Pixels uses 4. The first convolution/patch embedding is configured
for that channel count; inputs are not silently mapped to RGB. No probability
activation is applied. `pretrained=True` raises an error rather than attempting
a download. Both implementations currently start from random initialization.

## DeepLabV3+ with ResNet-50

`DeepLabV3PlusResNet50` uses torchvision's ResNet-50 architecture with
`weights=None`. Its final stage uses dilation to produce output stride 16. The
decoder consists of an ASPP context module (rates 6, 12, 18 and an image-level
pooling branch), a 48-channel projection of the stride-4 low-level feature,
two 3×3 decoder convolutions, and a class projection. The result is bilinearly
resized to the input spatial dimensions (`align_corners=False`). GroupNorm is
used in newly added head layers to avoid batch-size-one failures in the
global-pooling branch. The ResNet backbone retains its standard BatchNorm.

This follows the DeepLabV3+ encoder/ASPP/low-level-decoder design while using
the project's selected ResNet-50 backbone. The original paper reports ResNet-101
and Xception experiments; the ResNet-50 choice is a project baseline decision.
The ASPP/decoder here use regular convolutions rather than the optional
depthwise-separable variant described in the paper.

## SegFormer-B1 / MiT-B1

`SegFormerB1` is a self-contained PyTorch implementation of the MiT-B1 encoder
and SegFormer all-MLP decoder, with no Transformers runtime dependency. Its
stage depths are `[2,2,2,2]`, dimensions `[64,128,320,512]`, attention heads
`[1,2,5,8]`, spatial-reduction ratios `[8,4,2,1]`, patch sizes `[7,3,3,3]`,
and strides `[4,2,2,2]`. It uses MLP expansion 4, LayerNorm epsilon `1e-6`,
drop-path ramp 0 to 0.1, and the NVlabs MiT truncated-normal/LayerNorm/Conv
initialization. Each stage has the official post-block LayerNorm before its
feature map is returned.

The B1 decoder projects each stage to 256 channels, resizes to stage 1,
concatenates the four maps, then fuses them to 128 channels before the
classifier. The projection and fusion dimensions are explicit in
`configs/models/segformer_b1.yaml`. Lower-resolution logits are bilinearly
resized to input dimensions (`align_corners=False`). Supported model YAML
fields are applied by the factory; unknown keys and changes to fixed MiT-B1
structure are rejected.

### Remaining differences from NVlabs/MMSEG

- The classifier is configured for the project's class count and the first
  patch projection is initialized for 4 or 10 channels; no ADE pretrained
  checkpoint is loaded or RGB kernel adaptation is attempted.
- The decoder uses ordinary `BatchNorm2d` rather than MMSEG's configured
  `SyncBN`, since this implementation is local and can run on CPU. This can
  affect training statistics and distributed comparability.
- Attention uses separate query and key/value Linear modules rather than the
  official combined QKV module. The computation is equivalent when weights are
  mapped, but state-dict names/layout differ.
- MMCV/MMSEG component wrappers, optimizer parameter-wise rules, auxiliary
  heads, and training/inference wrappers are not included. The segmentation
  classifier uses the official head's 0.01 normal weight initialization.
- No numerical equivalence against official pretrained weights has been
  measured. That requires weight conversion and an approved external checkpoint.

## Checkpoints, class counts, and open choices

The model YAMLs specify architecture fields and dataset channel counts, but
leave `num_classes` unset because local class encodings and any class mapping
must first be verified. No pretrained checkpoint is fetched or adapted. If
pretrained initialization is later approved, the source checkpoint, first-layer
adaptation for 4/10 channels, classifier replacement, and initialization rule
for added input channels must be documented and tested as a separate decision.
The exact class mapping, pretrained-vs-random initialization, and whether to use
separable DeepLab head convolutions remain research decisions.

## References

- Chen et al., “Encoder-Decoder with Atrous Separable Convolution for Semantic
  Image Segmentation,” ECCV 2018:
  <https://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf>
- Xie et al., “SegFormer: Simple and Efficient Design for Semantic
  Segmentation with Transformers,” NeurIPS 2021:
  <https://arxiv.org/abs/2105.15203>
- Hugging Face Transformers SegFormer model documentation (architecture
  reference): <https://huggingface.co/docs/transformers/main/model_doc/segformer>
- Torchvision ResNet-50 API:
  <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html>
- NVlabs SegFormer MiT implementation:
  <https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py>
- NVlabs SegFormer B1 configuration:
  <https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py>
- NVlabs SegFormer decoder head:
  <https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py>
