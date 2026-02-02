from __future__ import annotations

import torch

from src.models import create_model


def test_simple_cnn_forward_output_shape() -> None:
	"""Test SimpleCNN model creation and forward pass."""
	model = create_model("simple_cnn", in_channels=3, pretrained=False)
	input_tensor = torch.randn(2, 3, 224, 224)
	output = model(input_tensor)
	assert output.shape == (2, 1)


def test_resnet_first_conv_matches_channels() -> None:
	model = create_model("resnet50", in_channels=4, pretrained=False)
	assert model.conv1.in_channels == 4


@torch.no_grad()
def test_efficientnet_forward_output_shape() -> None:
	model = create_model("efficientnet_b2", in_channels=3, pretrained=False)
	input_tensor = torch.randn(2, 3, 224, 224)
	output = model(input_tensor)
	assert output.shape == (2, 1)


@torch.no_grad()
def test_deit_small_forward_output_shape() -> None:
	"""Test DeiT-Small model creation and forward pass."""
	model = create_model("deit_small", in_channels=3, pretrained=False)
	input_tensor = torch.randn(2, 3, 224, 224)
	output = model(input_tensor)
	assert output.shape == (2, 1)


@torch.no_grad()
def test_deit_base_forward_output_shape() -> None:
	"""Test DeiT-Base model creation and forward pass."""
	model = create_model("deit_base", in_channels=3, pretrained=False)
	input_tensor = torch.randn(2, 3, 224, 224)
	output = model(input_tensor)
	assert output.shape == (2, 1)