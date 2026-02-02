from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
	def __init__(self, in_channels: int = 3, dropout: float = 0.5):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 32, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(64, 128, 3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(128, 256, 3, padding=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(1),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(256, 512),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		return self.classifier(x)


def _update_first_conv(module: nn.Module, in_channels: int) -> nn.Module:
	"""Replace first conv layer with new one having correct in_channels."""
	conv = module
	if conv.in_channels == in_channels:
		return conv

	# Create new conv with same properties but different in_channels
	new_conv = nn.Conv2d(
		in_channels=in_channels,
		out_channels=conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		dilation=conv.dilation,
		groups=1,  # Reset groups to 1 for channel adaptation
		bias=conv.bias is not None,
	)

	# Copy/adapt weights
	old_weight = conv.weight.data
	if in_channels > conv.in_channels:
		# Repeat weights across input channels
		repeat_times = (in_channels + conv.in_channels - 1) // conv.in_channels
		new_weight = old_weight.repeat(1, repeat_times, 1, 1)[:, :in_channels, :, :]
	else:
		# Slice weights to match smaller input
		new_weight = old_weight[:, :in_channels, :, :]

	new_conv.weight.data = new_weight
	if conv.bias is not None:
		new_conv.bias.data = conv.bias.data.clone()

	return new_conv


def create_model(name: str, in_channels: int = 3, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
	name = name.lower()

	if name == "simple_cnn":
		model = SimpleCNN(in_channels=in_channels, dropout=dropout)
	elif name == "resnet50":
		model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
		model.conv1 = _update_first_conv(model.conv1, in_channels)
		model.fc = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(model.fc.in_features, 1),
		)
	elif name == "efficientnet_b2":
		model = timm.create_model("efficientnet_b2", pretrained=pretrained, in_chans=in_channels, num_classes=1)
		if dropout > 0:
			if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
				layers = list(model.classifier.children())
				layers.insert(0, nn.Dropout(p=dropout))
				model.classifier = nn.Sequential(*layers)
	elif name == "deit_small":
		model = timm.create_model("deit_small_patch16_224", pretrained=pretrained, in_chans=in_channels, num_classes=1)
		if dropout > 0 and hasattr(model, "head"):
			model.head = nn.Sequential(nn.Dropout(p=dropout), model.head)
	elif name == "deit_base":
		model = timm.create_model("deit_base_patch16_224", pretrained=pretrained, in_chans=in_channels, num_classes=1)
		if dropout > 0 and hasattr(model, "head"):
			model.head = nn.Sequential(nn.Dropout(p=dropout), model.head)
	else:
		raise ValueError(f"Unsupported model architecture: {name}")

	return model
