"""Self-contained SegFormer-B1 (MiT-B1 encoder and all-MLP decoder)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class _OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, kernel: int, stride: int, norm_eps: float) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel, stride=stride,
                              padding=kernel // 2)
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        height, width = x.shape[-2:]
        tokens = self.norm(x.flatten(2).transpose(1, 2))
        return tokens, height, width


class _EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, reduction: int, norm_eps: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("embedding dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.reduction = reduction
        if reduction > 1:
            self.sr = nn.Conv2d(dim, dim, reduction, stride=reduction)
            self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q = self.q(x).reshape(batch, tokens, self.heads, self.head_dim).permute(0, 2, 1, 3)
        if self.reduction > 1:
            reduced = x.transpose(1, 2).reshape(batch, dim, height, width)
            reduced = self.sr(reduced).flatten(2).transpose(1, 2)
            reduced = self.norm(reduced)
        else:
            reduced = x
        kv = self.kv(reduced).reshape(batch, -1, 2, self.heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(output)


class _MixFeedForward(nn.Module):
    def __init__(self, dim: int, ratio: int) -> None:
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, _ = x.shape
        x = self.fc1(x).transpose(1, 2).reshape(batch, -1, height, width)
        x = self.dwconv(x).flatten(2).transpose(1, 2)
        return self.fc2(F.gelu(x))


class _MiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, reduction: int, ratio: int,
                 drop_path: float, norm_eps: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = _EfficientSelfAttention(dim, heads, reduction, norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = _MixFeedForward(dim, ratio)
        self.drop_path = _DropPath(drop_path)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class _MiTEncoder(nn.Module):
    """Mix Transformer encoder stages for MiT-B1."""

    def __init__(self, in_channels: int, *, norm_eps: float = 1e-6,
                 drop_path_rate: float = 0.1) -> None:
        super().__init__()
        dims = (64, 128, 320, 512)
        depths = (2, 2, 2, 2)
        heads = (1, 2, 5, 8)
        reductions = (8, 4, 2, 1)
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        previous_channels = in_channels
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            kernel = 7 if i == 0 else 3
            stride = 4 if i == 0 else 2
            self.embeds.append(_OverlapPatchEmbed(previous_channels, dim, kernel, stride, norm_eps))
            self.stages.append(nn.ModuleList([
                _MiTBlock(dim, heads[i], reductions[i], 4, rates[sum(depths[:i]) + j], norm_eps)
                for j in range(depth)
            ]))
            previous_channels = dim
        self.stage_norms = nn.ModuleList([nn.LayerNorm(dim, eps=norm_eps) for dim in dims])
        self.dims = dims
        self.depths, self.heads, self.reductions = depths, heads, reductions
        self.mlp_ratio, self.patch_kernels, self.patch_strides = 4, (7, 3, 3, 3), (4, 2, 2, 2)
        self.norm_eps, self.drop_path_rate = norm_eps, drop_path_rate
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
            nn.init.normal_(module.weight, mean=0.0, std=(2.0 / fan_out) ** 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for embed, blocks, norm in zip(self.embeds, self.stages, self.stage_norms):
            x, height, width = embed(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x)
            feature = x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], height, width)
            features.append(feature)
            x = feature
        return features


class _SegFormerDecoder(nn.Module):
    def __init__(self, dimensions: tuple[int, ...], num_classes: int,
                 embed_dim: int = 256, channels: int = 128,
                 profile: str = "legacy_128") -> None:
        super().__init__()
        if profile not in {"legacy_128", "nvlabs_reference"}:
            raise ValueError("decoder profile must be 'legacy_128' or 'nvlabs_reference'")
        if profile == "nvlabs_reference" and (embed_dim != 256 or channels != 256):
            raise ValueError("the NVlabs SegFormer head requires embed_dim=256 and fusion channels=256")
        self.projections = nn.ModuleList([nn.Linear(dim, embed_dim) for dim in dimensions])
        self.embed_dim, self.channels, self.profile = embed_dim, channels, profile
        self.concatenation_order = ("C4", "C3", "C2", "C1") if profile == "nvlabs_reference" else (
            "C1", "C2", "C3", "C4"
        )
        norm = nn.SyncBatchNorm if profile == "nvlabs_reference" else nn.BatchNorm2d
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(dimensions), channels, 1, bias=False),
            norm(channels), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
        )
        self.classifier = nn.Conv2d(channels, num_classes, 1)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            feature_size = feature.shape[-2:]
            feature = projection(feature.flatten(2).transpose(1, 2))
            feature = feature.transpose(1, 2).reshape(
                feature.shape[0], -1, *feature_size
            )
            feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feature)
        if self.profile == "nvlabs_reference":
            # NVlabs SegFormerHead concatenates the coarsest feature first.
            projected.reverse()
        return self.classifier(self.fuse(torch.cat(projected, dim=1)))


class SegFormerB1(nn.Module):
    """MiT-B1 plus SegFormer's hierarchical all-MLP segmentation decoder.

    The implementation follows the published MiT-B1 dimensions and starts
    from random initialization. ``in_channels`` is applied to the first
    overlapping patch embedding without RGB conversion.
    """

    def __init__(self, num_classes: int, in_channels: int, *, pretrained: bool = False,
                 decoder_hidden_size: int = 256, decoder_channels: int = 128,
                 decoder_profile: str = "legacy_128",
                 norm_eps: float = 1e-6, drop_path_rate: float = 0.1) -> None:
        super().__init__()
        if num_classes < 1 or in_channels < 1:
            raise ValueError("num_classes and in_channels must be positive")
        if pretrained:
            raise ValueError("automatic pretrained weights are unsupported; initialize explicitly")
        self.in_channels = in_channels
        self.num_classes = num_classes
        if (not isinstance(decoder_hidden_size, int) or isinstance(decoder_hidden_size, bool) or
                not isinstance(decoder_channels, int) or isinstance(decoder_channels, bool) or
                decoder_hidden_size < 1 or decoder_channels < 1 or norm_eps <= 0 or
                not 0 <= drop_path_rate < 1):
            raise ValueError("invalid SegFormer decoder or normalization parameters")
        self.encoder = _MiTEncoder(in_channels, norm_eps=norm_eps, drop_path_rate=drop_path_rate)
        self.decoder_profile = decoder_profile
        self.decoder = _SegFormerDecoder(
            self.encoder.dims, num_classes, decoder_hidden_size, decoder_channels, decoder_profile,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError(f"expected input [B,{self.in_channels},H,W]")
        logits = self.decoder(self.encoder(images))
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
