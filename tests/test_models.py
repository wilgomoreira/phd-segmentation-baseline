"""Small deterministic CPU checks for the segmentation model wrappers."""

import pytest
import torch
import yaml

from src.models import DeepLabV3PlusResNet50, SegFormerB1
from src.models import build_model


@pytest.mark.parametrize("model_type", [DeepLabV3PlusResNet50, SegFormerB1])
@pytest.mark.parametrize("in_channels", [4, 10])
def test_model_shape_finite_and_backward(model_type, in_channels):
    torch.manual_seed(1234)
    model = model_type(num_classes=5, in_channels=in_channels)
    model.train()
    images = torch.randn(2, in_channels, 32, 32, requires_grad=True)
    logits = model(images)
    assert logits.shape == (2, 5, 32, 32)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("model_type", [DeepLabV3PlusResNet50, SegFormerB1])
def test_model_configuration_matches_input_and_class_counts(model_type):
    model = model_type(num_classes=7, in_channels=10)
    assert model.in_channels == 10
    assert model.num_classes == 7
    if model_type is DeepLabV3PlusResNet50:
        assert model.stem[0].in_channels == 10
        assert model.classifier.out_channels == 7
    else:
        assert model.encoder.embeds[0].proj.in_channels == 10
        assert model.decoder.classifier.out_channels == 7
        assert model.encoder.dims == (64, 128, 320, 512)


@pytest.mark.parametrize("model_type", [DeepLabV3PlusResNet50, SegFormerB1])
def test_model_rejects_pretrained_flag_and_wrong_channel_count(model_type):
    with pytest.raises(ValueError, match="pretrained"):
        model_type(num_classes=2, in_channels=4, pretrained=True)
    model = model_type(num_classes=2, in_channels=4)
    with pytest.raises(ValueError, match="expected input"):
        model(torch.zeros(1, 3, 32, 32))


def test_segformer_b1_official_stage_and_decoder_structure():
    model = SegFormerB1(num_classes=3, in_channels=10)
    encoder = model.encoder
    assert encoder.depths == (2, 2, 2, 2)
    assert encoder.dims == (64, 128, 320, 512)
    assert encoder.heads == (1, 2, 5, 8)
    assert encoder.reductions == (8, 4, 2, 1)
    assert encoder.patch_kernels == (7, 3, 3, 3)
    assert encoder.patch_strides == (4, 2, 2, 2)
    assert encoder.mlp_ratio == 4
    assert len(encoder.stage_norms) == 4
    assert all(norm.eps == 1e-6 for norm in encoder.stage_norms)
    assert all(block.norm1.eps == 1e-6 for stage in encoder.stages for block in stage)
    assert torch.equal(encoder.stage_norms[0].weight, torch.ones(64))
    assert torch.equal(encoder.stage_norms[0].bias, torch.zeros(64))
    with torch.no_grad():
        stage_features = encoder(torch.randn(1, 10, 64, 64))
    assert [tuple(feature.shape) for feature in stage_features] == [
        (1, 64, 16, 16), (1, 128, 8, 8), (1, 320, 4, 4), (1, 512, 2, 2),
    ]
    assert [projection.out_features for projection in model.decoder.projections] == [256] * 4
    assert model.decoder.fuse[0].in_channels == 1024
    assert model.decoder.fuse[0].out_channels == 128
    assert model.decoder.classifier.in_channels == 128


def test_model_yaml_is_applied_and_inconsistent_values_rejected():
    with open("configs/models/segformer_b1.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model = build_model(config, num_classes=4, in_channels=10)
    assert model.decoder.embed_dim == config["decoder_hidden_size"]
    assert model.decoder.channels == config["decoder_channels"]
    bad = dict(config, depths=[3, 2, 2, 2])
    with pytest.raises(ValueError, match="depths"):
        build_model(bad, num_classes=4, in_channels=10)
    bad = dict(config, unsupported_knob=10)
    with pytest.raises(ValueError, match="unsupported"):
        build_model(bad, num_classes=4, in_channels=10)


def test_deeplab_yaml_applies_decoder_and_aspp_settings():
    with open("configs/models/deeplabv3plus.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model = build_model(config, num_classes=4, in_channels=4)
    assert model.output_stride == config["output_stride"] == 16
    assert model.aspp_rates == tuple(config["aspp_rates"])
    assert model.decoder_channels == config["decoder_channels"]
    assert model.classifier.in_channels == config["decoder_channels"]
    with pytest.raises(ValueError, match="output_stride"):
        build_model(dict(config, output_stride=8), num_classes=4, in_channels=4)
