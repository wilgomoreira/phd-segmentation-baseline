"""DeepLabV3+ with a randomly initialized torchvision ResNet-50 encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


def _conv_norm_relu(in_channels: int, out_channels: int, kernel_size: int = 3,
                    *, dilation: int = 1, groups: int = 1) -> nn.Sequential:
    padding = dilation * (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,
                  dilation=dilation, groups=groups, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.ReLU(inplace=True),
    )


class _ASPP(nn.Module):
    def __init__(self, in_channels: int, channels: int = 256,
                 rates: tuple[int, ...] = (6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            _conv_norm_relu(in_channels, channels, kernel_size=1),
            *[_conv_norm_relu(in_channels, channels, dilation=r, groups=1) for r in rates],
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels), nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * (len(rates) + 2), channels, 1, bias=False),
            nn.GroupNorm(8, channels), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([branch(x) for branch in self.branches] + [pooled], dim=1))


class DeepLabV3PlusResNet50(nn.Module):
    """DeepLabV3+ encoder-decoder using ResNet-50, output stride 16.

    Parameters are randomly initialized. The first convolution is constructed
    for ``in_channels`` directly, so multispectral inputs are not collapsed to RGB.
    """

    def __init__(self, num_classes: int, in_channels: int, *, pretrained: bool = False,
                 output_stride: int = 16, aspp_rates: tuple[int, ...] = (6, 12, 18),
                 decoder_channels: int = 256) -> None:
        super().__init__()
        if num_classes < 1 or in_channels < 1:
            raise ValueError("num_classes and in_channels must be positive")
        if pretrained:
            raise ValueError("automatic pretrained weights are unsupported; initialize explicitly")
        if not isinstance(output_stride, int) or isinstance(output_stride, bool) or output_stride != 16:
            raise ValueError("only DeepLabV3+ output_stride=16 is implemented")
        if not aspp_rates or any(not isinstance(rate, int) or rate <= 0 for rate in aspp_rates):
            raise ValueError("aspp_rates must be positive integers")
        if (not isinstance(decoder_channels, int) or isinstance(decoder_channels, bool) or
                decoder_channels <= 0 or decoder_channels % 8):
            raise ValueError("decoder_channels must be positive and divisible by 8 for GroupNorm")
        encoder = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4
        )
        self.output_stride, self.aspp_rates, self.decoder_channels = output_stride, tuple(aspp_rates), decoder_channels
        self.aspp = _ASPP(2048, channels=decoder_channels, rates=tuple(aspp_rates))
        self.low_level_projection = _conv_norm_relu(256, 48, kernel_size=1)
        self.decoder = nn.Sequential(
            _conv_norm_relu(decoder_channels + 48, decoder_channels),
            _conv_norm_relu(decoder_channels, decoder_channels),
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError(f"expected input [B,{self.in_channels},H,W]")
        output_size = images.shape[-2:]
        x = self.stem(images)
        low = self.layer1(x)
        x = self.layer4(self.layer3(self.layer2(low)))
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, self.low_level_projection(low)], dim=1))
        logits = self.classifier(x)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
