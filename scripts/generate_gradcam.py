#!/usr/bin/env python3
"""
Generate GradCAM visualizations for model explainability.
Creates heatmap overlays on actual MRI slices for interpretability.
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DatasetConfig, create_patient_centric_data_loaders
from src.training import TumorDetectionModule
from scripts.train import create_model_4ch, find_patient_dirs


class GradCAM:
	"""Gradient-weighted Class Activation Mapping."""
	
	def __init__(self, model: nn.Module, target_layer: str):
		self.model = model
		self.target_layer = target_layer
		self.gradients = None
		self.activations = None
		self._register_hooks()
	
	def _register_hooks(self):
		"""Register forward and backward hooks."""
		def forward_hook(module, input, output):
			self.activations = output.detach()
		
		def backward_hook(module, grad_input, grad_output):
			self.gradients = grad_output[0].detach()
		
		# Find target layer
		for name, module in self.model.named_modules():
			if self.target_layer in name:
				module.register_forward_hook(forward_hook)
				module.register_full_backward_hook(backward_hook)
				break
	
	def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
		"""Generate CAM for input."""
		self.model.eval()
		
		# Forward pass
		output = self.model(input_tensor)
		if class_idx is None:
			class_idx = output.argmax(dim=1)
		
		# Backward pass
		self.model.zero_grad()
		one_hot = torch.zeros_like(output)
		one_hot[0, class_idx] = 1
		output.backward(gradient=one_hot)
		
		# Calculate CAM
		gradients = self.gradients.cpu().data.numpy()[0]  # (C, H, W)
		activations = self.activations.cpu().data.numpy()[0]  # (C, H, W)
		
		weights = np.mean(gradients, axis=(1, 2))  # (C,)
		cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)  # (H, W)
		
		cam = np.maximum(cam, 0)  # ReLU
		cam = cv2.resize(cam, (224, 224))
		cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
		
		return cam


def create_heatmap_overlay(
	slice_img: np.ndarray,
	heatmap: np.ndarray,
	alpha: float = 0.4
) -> np.ndarray:
	"""Overlay heatmap on slice image."""
	# Normalize slice to [0, 255]
	slice_normalized = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255
	slice_colored = cv2.applyColorMap(slice_normalized.astype(np.uint8), cv2.COLORMAP_GRAY)
	
	# Normalize heatmap to [0, 255]
	heatmap_normalized = (heatmap * 255).astype(np.uint8)
	heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
	
	# Blend
	overlay = cv2.addWeighted(slice_colored, 1 - alpha, heatmap_colored, alpha, 0)
	return overlay


def main():
	print("\n" + "="*70)
	print("EXPLAINABILITY: GradCAM Visualization")
	print("="*70 + "\n")
	
	model_name = "efficientnet_b2"
	results_dir = Path("experiments_logs")
	output_dir = results_dir / "gradcam_visualizations"
	output_dir.mkdir(parents=True, exist_ok=True)
	
	# Load model
	print(f"Loading model: {model_name}")
	model = create_model_4ch(model_name)
	model.eval()
	
	# Prepare data
	task1_dir = Path("data/BraTS2021_Training_Data")
	patient_dirs = find_patient_dirs(task1_dir, num_patients=5)  # Sample 5 patients
	config = DatasetConfig(
		modalities=["t1", "t1ce", "t2", "flair"],
		image_size=(224, 224),
	)
	_, val_loader, _ = create_patient_centric_data_loaders(patient_dirs, config)
	
	# Initialize GradCAM
	gradcam = GradCAM(model, target_layer="features")
	
	# Process a few validation samples
	print("\nGenerating GradCAM visualizations...")
	num_samples = 0
	max_samples = 10
	
	with torch.no_grad():
		for batch_idx, batch in enumerate(val_loader):
			if num_samples >= max_samples:
				break
			
			images = batch["images"].squeeze(0)  # (num_slices, C, H, W)
			labels = batch["labels"].squeeze(0)  # (num_slices,)
			patient_id = batch["patient_id"][0]
			
			# Process a few slices from this patient
			for slice_idx in range(min(3, images.shape[0])):
				if num_samples >= max_samples:
					break
				
				img = images[slice_idx:slice_idx+1].cuda()  # (1, C, H, W)
				label = labels[slice_idx].item()
				
				# Generate CAM
				cam = gradcam.generate_cam(img)
				
				# Get the first modality (T1) for visualization
				base_slice = images[slice_idx, 0].numpy()
				
				# Create overlay
				overlay = create_heatmap_overlay(base_slice, cam)
				
				# Save
				output_path = output_dir / f"patient_{patient_id}_slice_{slice_idx:03d}_tumor_{int(label)}.png"
				cv2.imwrite(str(output_path), overlay)
				
				print(f"  Saved: {output_path.name}")
				num_samples += 1
	
	print(f"\n✓ Generated {num_samples} GradCAM visualizations")
	print(f"Output: {output_dir}\n")


if __name__ == "__main__":
	main()
