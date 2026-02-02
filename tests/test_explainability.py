from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.explainability import (
	TumorGradCAM,
	compute_iou_with_mask,
	get_target_layer,
	sanity_check_data_randomization,
	sanity_check_model_randomization,
)
from src.models import create_model


def test_get_target_layer_resnet():
	"""Test target layer selection for ResNet50."""
	model = create_model("resnet50", in_channels=3, pretrained=False)
	layer = get_target_layer(model, "resnet50")
	assert hasattr(model, "layer4")
	assert layer is model.layer4[-1]


def test_get_target_layer_efficientnet():
	"""Test target layer selection for EfficientNet-B2."""
	model = create_model("efficientnet_b2", in_channels=3, pretrained=False)
	layer = get_target_layer(model, "efficientnet_b2")
	# Should not raise error
	assert layer is not None


def test_get_target_layer_unknown_model():
	"""Test error handling for unknown model."""
	model = nn.Module()
	with pytest.raises(ValueError, match="Unsupported model architecture"):
		get_target_layer(model, "unknown_model")


def test_gradcam_generates_heatmap():
	"""Test Grad-CAM heatmap generation."""
	model = create_model("resnet50", in_channels=3, pretrained=False)
	model.eval()
	device = torch.device("cpu")

	cam = TumorGradCAM(model, "resnet50", device)

	# Create dummy input
	image = torch.randn(3, 224, 224)

	heatmap = cam.generate_heatmap(image)

	# Check output
	assert heatmap.shape == (224, 224)
	assert heatmap.min() >= 0.0
	assert heatmap.max() <= 1.0


def test_gradcam_with_prediction():
	"""Test Grad-CAM with prediction metadata."""
	model = create_model("resnet50", in_channels=3, pretrained=False)
	model.eval()
	device = torch.device("cpu")

	cam = TumorGradCAM(model, "resnet50", device)

	image = torch.randn(3, 224, 224)

	result = cam.generate_with_prediction(
		image=image,
		patient_id="00000",
		slice_idx=42,
		true_label=1,
	)

	assert result.heatmap.shape == (224, 224)
	assert 0.0 <= result.prediction <= 1.0
	assert result.patient_id == "00000"
	assert result.slice_idx == 42
	assert result.true_label == 1


def test_compute_iou_with_mask():
	"""Test IoU computation between heatmap and segmentation."""
	# Perfect overlap
	heatmap = np.ones((100, 100), dtype=np.float32)
	segmentation = np.ones((100, 100), dtype=np.float32)
	iou = compute_iou_with_mask(heatmap, segmentation, threshold=0.5)
	assert iou == pytest.approx(1.0)

	# No overlap
	heatmap = np.zeros((100, 100), dtype=np.float32)
	heatmap[:50, :] = 1.0
	segmentation = np.zeros((100, 100), dtype=np.float32)
	segmentation[50:, :] = 1.0
	iou = compute_iou_with_mask(heatmap, segmentation, threshold=0.5)
	assert iou == pytest.approx(0.0)

	# Partial overlap
	heatmap = np.zeros((100, 100), dtype=np.float32)
	heatmap[:75, :] = 1.0
	segmentation = np.zeros((100, 100), dtype=np.float32)
	segmentation[25:, :] = 1.0
	iou = compute_iou_with_mask(heatmap, segmentation, threshold=0.5)
	# Intersection: 50%, Union: 100% -> IoU = 0.5
	assert 0.4 < iou < 0.6


def test_sanity_check_model_randomization():
	"""Test sanity check: model weight randomization."""
	model = create_model("resnet50", in_channels=3, pretrained=False)
	model.eval()
	device = torch.device("cpu")

	image = torch.randn(3, 224, 224)

	result = sanity_check_model_randomization(model, image, "resnet50", device)

	assert "original" in result
	assert "randomized" in result
	assert "correlation" in result
	assert "pass" in result

	assert result["original"].shape == (224, 224)
	assert result["randomized"].shape == (224, 224)
	assert isinstance(result["correlation"], float)
	assert isinstance(result["pass"], bool)


def test_sanity_check_data_randomization():
	"""Test sanity check: data randomization."""
	model = create_model("resnet50", in_channels=3, pretrained=False)
	model.eval()
	device = torch.device("cpu")

	image = torch.randn(3, 224, 224)

	result = sanity_check_data_randomization(model, image, "resnet50", device)

	assert "original" in result
	assert "randomized" in result
	assert "correlation" in result
	assert "pass" in result

	assert result["original"].shape == (224, 224)
	assert result["randomized"].shape == (224, 224)