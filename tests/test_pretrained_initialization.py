"""Tests for explicit RGB-to-multispectral encoder initialization."""

import torch
import pytest
from torchvision.models import resnet50

from src.models.deeplabv3plus import DeepLabV3PlusResNet50
from src.models.pretrained import adapt_rgb_kernel_to_pastis, _map_resnet50, _map_segformer
from src.models.segformer import SegFormerB1


def test_rgb_kernels_map_to_b2_b3_b4_and_other_bands_are_zero():
    rgb = torch.stack([torch.full((2, 3, 3), float(index)) for index in range(3)], dim=1)
    mapped = adapt_rgb_kernel_to_pastis(rgb)
    assert mapped.shape == (2, 10, 3, 3)
    assert torch.equal(mapped[:, 0], rgb[:, 2])  # Blue kernel -> B2
    assert torch.equal(mapped[:, 1], rgb[:, 1])  # Green kernel -> B3
    assert torch.equal(mapped[:, 2], rgb[:, 0])  # Red kernel -> B4
    assert torch.count_nonzero(mapped[:, 3:]) == 0


def test_resnet_encoder_loads_completely_but_keeps_task_head_random():
    model = DeepLabV3PlusResNet50(num_classes=19, in_channels=10)
    custom_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                     if not name.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4."))}
    source = resnet50(weights=None).state_dict()
    mapped = _map_resnet50(model, source)
    assert len(mapped) == 318
    assert torch.equal(model.stem[0].weight[:, 0], source["conv1.weight"][:, 2])  # Blue -> B2
    assert torch.equal(model.stem[0].weight[:, 1], source["conv1.weight"][:, 1])  # Green -> B3
    assert torch.equal(model.stem[0].weight[:, 2], source["conv1.weight"][:, 0])  # Red -> B4
    assert torch.count_nonzero(model.stem[0].weight[:, 3:]) == 0
    assert all(torch.equal(custom_before[name], parameter) for name, parameter in model.named_parameters()
               if name in custom_before)
    assert all(parameter.requires_grad for name, parameter in model.named_parameters()
               if name.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4.")))


def test_resnet_loader_fails_on_missing_unexpected_or_mismatched_encoder_tensors():
    model = DeepLabV3PlusResNet50(num_classes=19, in_channels=10)
    source = resnet50(weights=None).state_dict()
    missing = dict(source)
    missing.pop("layer2.0.conv1.weight")
    with pytest.raises(ValueError, match="coverage mismatch"):
        _map_resnet50(model, missing)
    unexpected = dict(source)
    unexpected["layer5.0.conv1.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected ResNet"):
        _map_resnet50(model, unexpected)
    mismatched = dict(source)
    mismatched["layer2.0.conv1.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        _map_resnet50(model, mismatched)


def _segformer_hf_state(model):
    local = model.encoder.state_dict()
    source = {}
    for key, value in local.items():
        if key.startswith("embeds."):
            _, stage, component, *tail = key.split(".")
            component = {"proj": "proj", "norm": "layer_norm"}[component]
            out = f"segformer.encoder.patch_embeddings.{stage}.{component}.{'.'.join(tail)}"
            tensor = value
            if stage == "0" and component == "proj" and tail == ["weight"]:
                tensor = torch.randn(value.shape[0], 3, *value.shape[2:])
            source[out] = tensor
        elif key.startswith("stage_norms."):
            _, stage, *tail = key.split(".")
            source[f"segformer.encoder.layer_norm.{stage}.{'.'.join(tail)}"] = value
        else:
            _, stage, block, *parts = key.split(".")
            component = ".".join(parts)
            mapping = {
                "norm1.": "layer_norm_1.", "norm2.": "layer_norm_2.",
                "attn.q.": "attention.self.query.", "attn.sr.": "attention.self.sr.",
                "attn.norm.": "attention.self.layer_norm.", "attn.proj.": "attention.output.dense.",
                "mlp.fc1.": "mlp.dense1.", "mlp.dwconv.": "mlp.dwconv.dwconv.",
                "mlp.fc2.": "mlp.dense2.",
            }
            if component.startswith("attn.kv."):
                suffix = component[len("attn.kv."):]
                key_tensor, value_tensor = value.chunk(2, dim=0)
                source[f"segformer.encoder.block.{stage}.{block}.attention.self.key.{suffix}"] = key_tensor
                source[f"segformer.encoder.block.{stage}.{block}.attention.self.value.{suffix}"] = value_tensor
                continue
            for old, new in mapping.items():
                if component.startswith(old):
                    component = new + component[len(old):]
                    break
            else:
                raise AssertionError(f"test fixture cannot map {key}")
            source[f"segformer.encoder.block.{stage}.{block}.{component}"] = value
    source["classifier.weight"] = torch.zeros(1000, 512)
    source["classifier.bias"] = torch.zeros(1000)
    return source


@pytest.mark.parametrize("decoder_profile,decoder_channels", [
    ("legacy_128", 128),
    ("nvlabs_reference", 256),
])
def test_segformer_encoder_key_mapping_is_complete_and_decoder_remains_random(
    decoder_profile, decoder_channels,
):
    model = SegFormerB1(
        num_classes=19, in_channels=10, decoder_channels=decoder_channels,
        decoder_profile=decoder_profile,
    )
    decoder_before = {name: parameter.detach().clone() for name, parameter in model.decoder.named_parameters()}
    source = _segformer_hf_state(model)
    mapped = _map_segformer(model, source)
    assert len(mapped) == len(model.encoder.state_dict())
    first_kernel = model.encoder.embeds[0].proj.weight
    source_kernel = source["segformer.encoder.patch_embeddings.0.proj.weight"]
    assert torch.equal(first_kernel[:, 0], source_kernel[:, 2])  # Blue -> B2
    assert torch.equal(first_kernel[:, 1], source_kernel[:, 1])  # Green -> B3
    assert torch.equal(first_kernel[:, 2], source_kernel[:, 0])  # Red -> B4
    assert torch.count_nonzero(first_kernel[:, 3:]) == 0
    assert all(torch.equal(decoder_before[name], parameter) for name, parameter in model.decoder.named_parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
