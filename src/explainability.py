from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.transform import resize


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
	"""
	Identify the appropriate target layer for Grad-CAM based on model architecture.

	For CNNs, we target the last convolutional layer before global pooling.
	"""
	model_name_lower = model_name.lower()

	if "resnet" in model_name_lower:
		# ResNet: target last block of layer4
		if hasattr(model, "layer4"):
			return model.layer4[-1]
		raise ValueError(f"ResNet model missing 'layer4' attribute")

	elif "efficientnet" in model_name_lower:
		# EfficientNet from timm: target final conv layer
		if hasattr(model, "conv_head"):
			return model.conv_head
		elif hasattr(model, "blocks"):
			# Alternative structure
			return model.blocks[-1][-1] if isinstance(model.blocks[-1], nn.Sequential) else model.blocks[-1]
		raise ValueError(f"EfficientNet model missing expected conv layers")

	elif "vit" in model_name_lower:
		# Vision Transformer: target last block's normalization layer
		if hasattr(model, "blocks") and len(model.blocks) > 0:
			return model.blocks[-1].norm1
		raise ValueError(f"ViT model missing 'blocks' attribute")

	else:
		raise ValueError(f"Unsupported model architecture: {model_name}")


@dataclass
class GradCAMResult:
	"""Container for Grad-CAM heatmap and metadata."""
	heatmap: np.ndarray  # Shape: (H, W), values in [0, 1]
	patient_id: str
	slice_idx: int
	prediction: float  # Probability
	true_label: int
	model_name: str


class TumorGradCAM:
	"""Grad-CAM implementation for brain tumor detection models."""

	def __init__(self, model: nn.Module, model_name: str, device: torch.device):
		"""
		Initialize Grad-CAM for a trained model.

		Args:
			model: Trained PyTorch model (in eval mode)
			model_name: Architecture identifier (for layer selection)
			device: torch.device for computation
		"""
		self.model = model.to(device)
		self.model.eval()
		self.model_name = model_name
		self.device = device

		target_layer = get_target_layer(model, model_name)

		# Add reshape transform for ViT models
		reshape_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
		if "vit" in model_name.lower():
			reshape_transform = self._build_vit_reshape()

		self.cam = GradCAM(
			model=model,
			target_layers=[target_layer],
			reshape_transform=reshape_transform,
		)

		self._binary_output: Optional[bool] = None

	def _infer_vit_grid(self) -> Tuple[int, int]:
		patch_embed = getattr(self.model, "patch_embed", None)
		if patch_embed is not None:
			grid_size = getattr(patch_embed, "grid_size", None)
			if grid_size is not None:
				return int(grid_size[0]), int(grid_size[1])
			num_patches = getattr(patch_embed, "num_patches", None)
			if num_patches is not None:
				side = int(math.sqrt(num_patches))
				if side * side == int(num_patches):
					return side, side

		pos_embed = getattr(self.model, "pos_embed", None)
		if pos_embed is not None:
			num_tokens = pos_embed.shape[1] - 1  # drop CLS token
			side = int(math.sqrt(num_tokens))
			if side * side == num_tokens:
				return side, side

		raise ValueError("Unable to infer ViT grid size for Grad-CAM reshape. Verify model compatibility.")

	def _build_vit_reshape(self) -> Callable[[torch.Tensor], torch.Tensor]:
		height, width = self._infer_vit_grid()

		def reshape_vit(tensor: torch.Tensor) -> torch.Tensor:
			# ViT outputs (batch, num_tokens, embedding_dim)
			tokens = tensor[:, 1:, :]  # drop CLS token
			if tokens.size(1) != height * width:
				raise ValueError(
					f"Unexpected ViT token count {tokens.size(1)}; expected {height * width}"
				)
			result = tokens.reshape(tensor.size(0), height, width, tensor.size(2))
			result = result.permute(0, 3, 1, 2)  # -> (B, C, H, W)
			return result.contiguous()

		return reshape_vit

	def generate_heatmap(
		self,
		image: torch.Tensor,
		target_class: Optional[int] = None,
	) -> np.ndarray:
		"""
		Generate Grad-CAM heatmap for a single image.

		Args:
			image: Input tensor, shape (C, H, W)
			target_class: Target class index (None = predicted class)

		Returns:
			Heatmap array, shape (H, W), values in [0, 1]
		"""
		if image.ndim == 3:
			image = image.unsqueeze(0)  # Add batch dimension

		image = image.to(self.device)

		# Generate CAM with proper target specification
		if target_class is None:
			targets = None
		else:
			if self._binary_output is None:
				with torch.no_grad():
					logits_preview = self.model(image)
					logits_preview = logits_preview.view(logits_preview.size(0), -1)
					self._binary_output = logits_preview.shape[1] == 1

			target_idx = target_class
			if self._binary_output:
				target_idx = 0  # Single-logit heads expose only one target
			targets = [ClassifierOutputTarget(target_idx)]
		grayscale_cam = self.cam(input_tensor=image, targets=targets)

		return grayscale_cam[0]  # Shape: (H, W)

	def generate_with_prediction(
		self,
		image: torch.Tensor,
		patient_id: str,
		slice_idx: int,
		true_label: int,
	) -> GradCAMResult:
		"""
		Generate Grad-CAM heatmap along with model prediction.

		Returns:
			GradCAMResult with heatmap and metadata
		"""
		image_input = image.unsqueeze(0).to(self.device)

		# Get prediction
		with torch.no_grad():
			logits = self.model(image_input).squeeze()
			prob = torch.sigmoid(logits).item()

		# Generate heatmap
		heatmap = self.generate_heatmap(image)

		return GradCAMResult(
			heatmap=heatmap,
			patient_id=patient_id,
			slice_idx=slice_idx,
			prediction=prob,
			true_label=true_label,
			model_name=self.model_name,
		)


def compute_iou_with_mask(
	heatmap: np.ndarray,
	segmentation: np.ndarray,
	threshold: float = 0.5,
) -> float:
	"""
	Compute IoU between Grad-CAM attention and tumor segmentation.

	Requirement: Target IoU > 0.5 indicates model focuses on tumor regions.

	Args:
		heatmap: Grad-CAM heatmap, shape (H, W), values in [0, 1]
		segmentation: Tumor segmentation mask, shape (H, W) or (H, W, D)
		threshold: Heatmap threshold for binarization (default 0.5)

	Returns:
		IoU score in [0, 1]
	"""
	# Handle 3D segmentation (extract 2D slice if needed)
	if segmentation.ndim == 3:
		raise ValueError("Pass 2D segmentation slice, not 3D volume")

	# Resize segmentation to match heatmap size
	if segmentation.shape != heatmap.shape:
		seg_resized = resize(segmentation, heatmap.shape, order=0, anti_aliasing=False, preserve_range=True)
	else:
		seg_resized = segmentation

	# Binarize
	cam_binary = (heatmap >= threshold).astype(np.uint8)
	seg_binary = (seg_resized > 0).astype(np.uint8)

	# Compute IoU
	intersection = np.logical_and(cam_binary, seg_binary).sum()
	union = np.logical_or(cam_binary, seg_binary).sum()

	return float(intersection / union) if union > 0 else 0.0


def visualize_gradcam_overlay(
	result: GradCAMResult,
	original_slice: np.ndarray,
	output_path: Optional[Path] = None,
	show_segmentation: Optional[np.ndarray] = None,
	*,
	figure_size: tuple[float, float] = (18.0, 6.0),
	title_fontsize: float = 14.0,
	label_fontsize: float = 12.0,
	tick_fontsize: float = 10.0,
) -> plt.Figure:
	"""
	Create high-quality Grad-CAM visualization.

	Args:
		result: GradCAMResult from generate_with_prediction()
		original_slice: Original MRI slice for background (H, W) or (C, H, W)
		output_path: If provided, save figure to this path
		show_segmentation: Optional segmentation mask to overlay

	Returns:
		matplotlib Figure
	"""
	# Extract background image (use first channel if multi-channel)
	if original_slice.ndim == 3:
		background = original_slice[0]  # Use first modality
	else:
		background = original_slice

	# Normalize background to [0, 1]
	bg_min, bg_max = background.min(), background.max()
	if bg_max > bg_min:
		background_norm = (background - bg_min) / (bg_max - bg_min)
	else:
		background_norm = np.zeros_like(background)

	# Create figure
	num_cols = 4 if show_segmentation is not None else 3
	fig, axes = plt.subplots(1, num_cols, figsize=figure_size)

	# Plot 1: Original MRI
	axes[0].imshow(background, cmap='gray')
	axes[0].set_title(
		f"Original MRI\nPatient: {result.patient_id}\nSlice: {result.slice_idx}",
		fontsize=title_fontsize,
	)
	axes[0].axis('off')

	# Plot 2: Grad-CAM heatmap
	axes[1].imshow(result.heatmap, cmap='jet', vmin=0, vmax=1)
	axes[1].set_title("Grad-CAM Heatmap", fontsize=title_fontsize)
	axes[1].axis('off')

	# Plot 3: Overlay
	background_rgb = np.stack([background_norm] * 3, axis=-1)
	overlay = show_cam_on_image(background_rgb, result.heatmap, use_rgb=True)
	axes[2].imshow(overlay)

	pred_class = int(result.prediction > 0.5)
	correctness = "CORRECT" if pred_class == result.true_label else "WRONG"
	axes[2].set_title(
		f"Overlay\n"
		f"Pred: {result.prediction:.3f} ({pred_class})\n"
		f"True: {result.true_label} | {correctness}",
		fontsize=title_fontsize,
	)
	axes[2].axis('off')

	# Optional Plot 4: Segmentation comparison
	if show_segmentation is not None:
		if show_segmentation.shape != result.heatmap.shape:
			seg_resized = resize(show_segmentation, result.heatmap.shape, order=0, anti_aliasing=False)
		else:
			seg_resized = show_segmentation

		# Show heatmap overlaid on segmentation
		seg_binary = (seg_resized > 0).astype(float)
		axes[3].imshow(background, cmap='gray', alpha=0.6)
		axes[3].imshow(seg_binary, cmap='Reds', alpha=0.3)
		axes[3].imshow(result.heatmap, cmap='jet', alpha=0.4)

		iou = compute_iou_with_mask(result.heatmap, seg_resized)
		axes[3].set_title(f"Seg + CAM\nIoU: {iou:.3f}", fontsize=title_fontsize)
		axes[3].axis('off')

	for ax in axes:
		ax.tick_params(labelsize=tick_fontsize)
	plt.tight_layout()

	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=300, bbox_inches='tight')
		plt.close(fig)

	return fig


def sanity_check_model_randomization(
	model: nn.Module,
	image: torch.Tensor,
	model_name: str,
	device: torch.device,
) -> Dict[str, np.ndarray]:
	"""
	Sanity check: Compare Grad-CAM before/after model weight randomization.

	Requirement: Heatmaps should differ significantly (correlation < 0.3).
	If they're similar, Grad-CAM is not actually using learned features.

	Args:
		model: Trained model
		image: Input image tensor (C, H, W)
		model_name: Model architecture name
		device: Computation device

	Returns:
		Dictionary with 'original', 'randomized' heatmaps and 'correlation' score
	"""
	# Generate heatmap with trained model
	cam_original = TumorGradCAM(model, model_name, device)
	heatmap_original = cam_original.generate_heatmap(image)

	# Create randomized copy
	randomized_model = copy.deepcopy(model)
	with torch.no_grad():
		for param in randomized_model.parameters():
			param.data = torch.randn_like(param.data)

	# Generate heatmap with random weights
	cam_random = TumorGradCAM(randomized_model, model_name, device)
	heatmap_random = cam_random.generate_heatmap(image)

	# Compute correlation
	correlation = np.corrcoef(
		heatmap_original.flatten(),
		heatmap_random.flatten()
	)[0, 1]

	return {
		"original": heatmap_original,
		"randomized": heatmap_random,
		"correlation": float(correlation),
		"pass": bool(correlation < 0.3),  # Should be LOW
	}


def sanity_check_data_randomization(
	model: nn.Module,
	original_image: torch.Tensor,
	model_name: str,
	device: torch.device,
) -> Dict[str, np.ndarray]:
	"""
	Sanity check: Compare Grad-CAM on real image vs random noise.

	Requirement: Heatmaps should differ significantly.
	If they're similar, Grad-CAM is highlighting arbitrary features.

	Args:
		model: Trained model
		original_image: Real MRI image (C, H, W)
		model_name: Model architecture name
		device: Computation device

	Returns:
		Dictionary with 'original', 'randomized' heatmaps and 'correlation' score
	"""
	cam = TumorGradCAM(model, model_name, device)

	# Real image heatmap
	heatmap_original = cam.generate_heatmap(original_image)

	# Random noise image
	random_image = torch.randn_like(original_image)
	heatmap_random = cam.generate_heatmap(random_image)

	# Compute correlation
	correlation = np.corrcoef(
		heatmap_original.flatten(),
		heatmap_random.flatten()
	)[0, 1]

	return {
		"original": heatmap_original,
		"randomized": heatmap_random,
		"correlation": float(correlation),
		"pass": bool(correlation < 0.3),  # Should be LOW
	}