"""Explicit, checksum-verified ImageNet encoder initialization."""

from __future__ import annotations

import hashlib
from pathlib import Path
import torch
from torch import nn


def adapt_rgb_kernel_to_pastis(rgb_weight: torch.Tensor) -> torch.Tensor:
    """Map RGB kernels onto PASTIS B2,B3,B4 and zero the other seven bands.

    PASTIS channel order is B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12;
    Visible-band spectral matching maps blue->B2, green->B3, and red->B4.
    """
    if rgb_weight.ndim != 4 or rgb_weight.shape[1] != 3:
        raise ValueError("expected an RGB convolution kernel with shape [out,3,height,width]")
    adapted = rgb_weight.new_zeros((rgb_weight.shape[0], 10, *rgb_weight.shape[2:]))
    adapted[:, 0] = rgb_weight[:, 2]  # Blue -> B2
    adapted[:, 1] = rgb_weight[:, 1]  # Green -> B3
    adapted[:, 2] = rgb_weight[:, 0]  # Red -> B4
    return adapted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoder_keys(model: nn.Module) -> set[str]:
    if hasattr(model, "stem") and hasattr(model, "layer4"):
        return {key for key in model.state_dict()
                if key.startswith(("stem.0.", "stem.1.", "layer1.", "layer2.", "layer3.", "layer4."))}
    if hasattr(model, "encoder") and hasattr(model.encoder, "embeds"):
        return {f"encoder.{key}" for key in model.encoder.state_dict()}
    raise TypeError(f"unsupported encoder model type: {type(model).__name__}")


def _map_resnet50(model: nn.Module, source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target_keys = _encoder_keys(model)
    mapped: dict[str, torch.Tensor] = {}
    for key, tensor in source.items():
        if key == "conv1.weight":
            mapped["stem.0.weight"] = adapt_rgb_kernel_to_pastis(tensor)
        elif key.startswith("bn1."):
            mapped["stem.1." + key[len("bn1."):]] = tensor
        elif key.startswith(("layer1.", "layer2.", "layer3.", "layer4.")):
            mapped[key] = tensor
        elif not key.startswith("fc."):
            raise ValueError(f"unexpected ResNet-50 checkpoint parameter: {key}")
    return _validate_mapping(model, target_keys, mapped)


def _segformer_target_key(source_key: str) -> str | None:
    prefix = "segformer.encoder."
    if not source_key.startswith(prefix):
        return None
    key = source_key[len(prefix):]
    if key.startswith("patch_embeddings."):
        key = key.replace("patch_embeddings.", "embeds.", 1)
        key = key.replace(".layer_norm.", ".norm.")
        return "encoder." + key
    if key.startswith("layer_norm."):
        return "encoder.stage_norms." + key[len("layer_norm."):]
    if key.startswith("block."):
        # HF: block.S.B.*; local model: stages.S.B.*
        parts = key.split(".")
        if len(parts) < 5:
            raise ValueError(f"unrecognized SegFormer encoder key: {source_key}")
        stage, block, component = parts[1], parts[2], ".".join(parts[3:])
        substitutions = (
            ("layer_norm_1.", "norm1."),
            ("layer_norm_2.", "norm2."),
            ("attention.self.query.", "attn.q."),
            ("attention.self.sr.", "attn.sr."),
            ("attention.self.layer_norm.", "attn.norm."),
            ("attention.output.dense.", "attn.proj."),
            ("mlp.dense1.", "mlp.fc1."),
            ("mlp.dwconv.dwconv.", "mlp.dwconv."),
            ("mlp.dense2.", "mlp.fc2."),
        )
        for old, new in substitutions:
            if component.startswith(old):
                component = new + component[len(old):]
                break
        else:
            if component.startswith("attention.self.key.") or component.startswith("attention.self.value."):
                kind, suffix = component.split(".", 3)[2], component.rsplit(".", 1)[1]
                return f"encoder.stages.{stage}.{block}.attn.kv_{kind}.{suffix}"
            raise ValueError(f"unrecognized SegFormer encoder key: {source_key}")
        return f"encoder.stages.{stage}.{block}.{component}"
    raise ValueError(f"unrecognized SegFormer encoder key: {source_key}")


def _map_segformer(model: nn.Module, source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target_keys = _encoder_keys(model)
    mapped: dict[str, torch.Tensor] = {}
    kv_parts: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, tensor in source.items():
        target_key = _segformer_target_key(key)
        if target_key is None:
            if not key.startswith("classifier."):
                raise ValueError(f"unexpected SegFormer checkpoint parameter: {key}")
            continue
        if ".attn.kv_key." in target_key or ".attn.kv_value." in target_key:
            base, kind, suffix = target_key.rsplit(".", 2)
            kv_parts.setdefault((base, suffix), {})[kind] = tensor
        else:
            mapped[target_key] = tensor
    for (base, suffix), parts in kv_parts.items():
        if set(parts) != {"kv_key", "kv_value"}:
            raise ValueError(f"incomplete SegFormer key/value pair for {base}.{suffix}")
        mapped[f"{base}.kv.{suffix}"] = torch.cat((parts["kv_key"], parts["kv_value"]), dim=0)
    # Adapt the first stage's RGB input kernel, preserving its target layout.
    first = "encoder.embeds.0.proj.weight"
    if first not in mapped:
        raise ValueError("SegFormer checkpoint is missing its first input projection")
    mapped[first] = adapt_rgb_kernel_to_pastis(mapped[first])
    return _validate_mapping(model, target_keys, mapped)


def _validate_mapping(model: nn.Module, target_keys: set[str], mapped: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    missing = target_keys - set(mapped)
    unexpected = set(mapped) - target_keys
    if missing or unexpected:
        raise ValueError(f"encoder key coverage mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    target = model.state_dict()
    mismatched = {
        key: (tuple(mapped[key].shape), tuple(target[key].shape))
        for key in target_keys if mapped[key].shape != target[key].shape
    }
    if mismatched:
        raise ValueError(f"encoder parameter shape mismatch: {mismatched}")
    result = model.load_state_dict(mapped, strict=False)
    expected_missing = {
        key for key in set(target) - target_keys
        if not key.endswith("num_batches_tracked")
    }
    if set(result.unexpected_keys) or set(result.missing_keys) != expected_missing:
        raise ValueError("model load did not leave exactly the custom decoder/head randomly initialized")
    if any(not parameter.requires_grad for name, parameter in model.named_parameters()
           if name in target_keys):
        raise ValueError("all pretrained encoder parameters must remain trainable")
    return mapped


def initialize_pretrained_encoder(model: nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    """Load one approved local checkpoint after checking its provenance hash."""
    if not isinstance(config, dict) or config.get("kind") != "imagenet_encoder":
        raise ValueError("pretrained initialization requires kind: imagenet_encoder")
    path = Path(config["checkpoint_path"])
    if not path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint does not exist: {path}")
    actual_hash = _sha256(path)
    if actual_hash != config.get("sha256"):
        raise ValueError(f"pretrained checkpoint SHA-256 mismatch: {actual_hash}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise ValueError("pretrained checkpoint must be a plain tensor state dictionary")
    model_key = config.get("model")
    if model_key == "torchvision_resnet50_imagenet1k_v2":
        mapped = _map_resnet50(model, checkpoint)
    elif model_key == "nvidia_mit_b1_imagenet1k":
        mapped = _map_segformer(model, checkpoint)
    else:
        raise ValueError(f"unsupported pretrained source model {model_key!r}")
    return {
        "model": model_key,
        "checkpoint_path": str(path.resolve()),
        "sha256": actual_hash,
        "source_url": config["source_url"],
        "source_revision": config.get("source_revision"),
        "license": config["license"],
        "adaptation": config["adaptation"],
        "encoder_tensors_loaded": len(mapped),
    }
