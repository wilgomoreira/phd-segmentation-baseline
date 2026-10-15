"""Strict constructor for the project's approved segmentation baselines."""

from __future__ import annotations

from typing import Any

from .deeplabv3plus import DeepLabV3PlusResNet50
from .segformer import SegFormerB1


def _require_equal(config: dict[str, Any], key: str, expected: Any) -> None:
    if key in config and config[key] != expected:
        raise ValueError(f"unsupported or inconsistent {key}: expected {expected!r}, got {config[key]!r}")


def build_model(config: dict[str, Any], *, num_classes: int, in_channels: int):
    """Build only the declared architecture; reject unknown or ignored YAML keys."""
    name = config.get("name") or config.get("architecture")
    key = str(name).lower().replace("-", "").replace("_", "")
    common = {"name", "architecture", "backbone", "in_channels_by_dataset", "num_classes", "pretrained"}
    configured_classes = config.get("num_classes")
    if "pretrained" in config and not isinstance(config["pretrained"], bool):
        raise TypeError("pretrained must be a YAML boolean")
    channel_map = config.get("in_channels_by_dataset", {})
    if not isinstance(channel_map, dict) or any(
        dataset not in {"pastis", "five_billion_pixels"} or
        not isinstance(channels, int) or isinstance(channels, bool) or channels < 1
        for dataset, channels in channel_map.items()
    ):
        raise ValueError("in_channels_by_dataset must map supported datasets to positive integers")
    if configured_classes is not None and (not isinstance(configured_classes, int) or isinstance(configured_classes, bool)):
        raise TypeError("num_classes must be an integer or null")
    if configured_classes is not None and configured_classes != num_classes:
        raise ValueError("training num_classes conflicts with model configuration")
    if key in {"segformerb1", "segformer"}:
        allowed = common | {"depths", "hidden_sizes", "num_heads", "sr_ratios", "mlp_ratio",
                            "decoder_hidden_size", "decoder_channels", "norm_eps", "drop_path_rate"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unsupported SegFormer configuration keys: {sorted(unknown)}")
        _require_equal(config, "architecture", "SegFormer")
        _require_equal(config, "backbone", "MiT-B1")
        for field, expected in (("depths", [2, 2, 2, 2]), ("hidden_sizes", [64, 128, 320, 512]),
                                ("num_heads", [1, 2, 5, 8]), ("sr_ratios", [8, 4, 2, 1]),
                                ("mlp_ratio", 4), ("norm_eps", 1e-6), ("drop_path_rate", 0.1)):
            _require_equal(config, field, expected)
        return SegFormerB1(
            num_classes=num_classes, in_channels=in_channels,
            pretrained=bool(config.get("pretrained", False)),
            decoder_hidden_size=config.get("decoder_hidden_size", 256),
            decoder_channels=config.get("decoder_channels", 128),
            norm_eps=float(config.get("norm_eps", 1e-6)),
            drop_path_rate=float(config.get("drop_path_rate", 0.1)),
        )
    if key in {"deeplabv3plusresnet50", "deeplabv3plus"}:
        allowed = common | {"output_stride", "aspp_rates", "decoder_channels"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unsupported DeepLabV3+ configuration keys: {sorted(unknown)}")
        _require_equal(config, "architecture", "DeepLabV3+")
        _require_equal(config, "backbone", "ResNet-50")
        return DeepLabV3PlusResNet50(
            num_classes=num_classes, in_channels=in_channels,
            pretrained=bool(config.get("pretrained", False)),
            output_stride=config.get("output_stride", 16),
            aspp_rates=tuple(config.get("aspp_rates", (6, 12, 18))),
            decoder_channels=config.get("decoder_channels", 256),
        )
    raise ValueError(f"unsupported model configuration name: {name!r}")
