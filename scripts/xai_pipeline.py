#!/usr/bin/env python3
"""
XAI Pipeline: Select best samples (TP/FP/TN/FN) and generate GradCAM visualizations.

Strategy:
  1. Run inference on test set, record predictions + ground truth labels
  2. Compute IoU with segmentation masks for sample quality assessment
  3. Select diverse samples: best TP, worst TP, best FP, etc.
  4. Generate GradCAM heatmaps for selected samples
  5. Save visualizations with metadata
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DatasetConfig, create_patient_centric_data_loaders
from src.preprocessing import load_nifti
from src.training import TumorDetectionModule
from scripts.train import create_model, find_patient_dirs


class GradCAM:
	"""Gradient-weighted Class Activation Mapping for explainability."""
	
	def __init__(self, model: nn.Module, target_layer_name: str = "blocks"):
		"""Initialize GradCAM."""
		self.model = model
		self.target_layer_name = target_layer_name
		self.gradients = None
		self.activations = None
		self.last_layer = None
		self._find_target_layer()
	
	def _find_target_layer(self):
		"""Find the target convolutional layer by name, or last conv if not found."""
		# First try to find the specified layer by name
		if self.target_layer_name and self.target_layer_name != "blocks":
			for name, module in self.model.named_modules():
				if self.target_layer_name in name and isinstance(module, (nn.Conv2d, nn.Sequential)):
					# If it's a Sequential block, find the last Conv2d in it
					if isinstance(module, nn.Sequential):
						for subname, submodule in module.named_modules():
							if isinstance(submodule, nn.Conv2d):
								self.last_layer = submodule
								self.last_layer_name = name + "." + subname
					else:
						self.last_layer = module
						self.last_layer_name = name
					break
		
		# Fallback: find the last Conv2d layer
		if self.last_layer is None:
			for name, module in self.model.named_modules():
				if isinstance(module, nn.Conv2d):
					self.last_layer = module
					self.last_layer_name = name
		
		if self.last_layer is None:
			print("  WARNING: No Conv2d layer found for GradCAM")
			return
		
		# Register hooks
		def forward_hook(module, input, output):
			self.activations = output.detach()
		
		def backward_hook(module, grad_input, grad_output):
			self.gradients = grad_output[0].detach()
		
		self.last_layer.register_forward_hook(forward_hook)
		self.last_layer.register_full_backward_hook(backward_hook)
		print(f"  ✓ GradCAM registered on: {self.last_layer_name}")
	
	def generate_cam(self, input_tensor: torch.Tensor, model_output: torch.Tensor) -> np.ndarray:
		"""Generate Class Activation Map."""
		self.model.train()  # Enable gradients
		self.model.zero_grad()
		
		try:
			# Backward pass
			loss = model_output.mean()
			loss.backward()
			
			# Check if we captured gradients
			if self.gradients is None or self.activations is None:
				return np.ones((224, 224)) * 0.5
			
			# Extract and process
			gradients = self.gradients.cpu().numpy()
			activations = self.activations.cpu().numpy()
			
			# Compute weights (average gradient across spatial dimensions)
			weights = np.mean(gradients[0], axis=(1, 2))  # Average over H, W
			
			# Apply weights to activations
			cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations[0], axis=0)
			
			# Normalize
			cam = np.maximum(cam, 0)  # ReLU
			cam_max = cam.max()
			if cam_max > 0:
				cam = cam / cam_max
			else:
				cam = np.ones_like(cam) * 0.5
			
			# Smooth the CAM to avoid blockiness (interpolate activation map)
			from scipy import ndimage
			cam_smooth = ndimage.gaussian_filter(cam, sigma=2.0)
			
			self.model.eval()
			return cam_smooth
		
		except Exception as e:
			print(f"    GradCAM error: {str(e)[:50]}")
			self.model.eval()
			return np.ones((224, 224)) * 0.5


def compute_iou_with_segmentation(
	slice_img: np.ndarray,
	seg_slice: np.ndarray,
	prediction: int,
	ground_truth: int
) -> Dict[str, float]:
	"""Compute IoU between prediction and ground truth using segmentation mask.
	
	Args:
		slice_img: MRI slice (H, W)
		seg_slice: Segmentation mask (H, W), values > 0 = tumor
		prediction: Model prediction (0 or 1)
		ground_truth: Ground truth label (0 or 1)
	
	Returns:
		Dict with IoU and quality metrics
	"""
	seg_binary = (seg_slice > 0).astype(int)
	
	# If ground truth is tumor
	if ground_truth == 1:
		# IoU = overlap with segmentation / union
		intersection = np.sum(seg_binary)
		union = np.sum(seg_binary)  # Perfect case: all tumor pixels found
		iou = intersection / (union + 1e-8) if union > 0 else 0
	else:
		# For non-tumor slices, high IoU means correctly avoiding tumor areas
		seg_inverse = 1 - seg_binary
		intersection = np.sum(seg_inverse)
		iou = intersection / (seg_inverse.shape[0] * seg_inverse.shape[1] + 1e-8)
	
	return {"iou": iou}


def collect_predictions(
	model: nn.Module,
	test_loader: DataLoader,
	device: str = "cuda"
) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray]:
	"""Run inference on test set and collect predictions with metadata.
	
	Returns:
		Tuple of (predictions_list, all_logits, all_labels, all_iou_scores)
	"""
	model = model.to(device)
	model.eval()
	
	predictions = []
	all_logits = []
	all_labels = []
	all_iou = []
	
	with torch.no_grad():
		for batch_idx, batch in enumerate(test_loader):
			images = batch["images"].squeeze(0).to(device)  # (num_slices, C, H, W)
			labels = batch["labels"].squeeze(0).to(device)  # (num_slices,)
			patient_id = batch["patient_id"][0]
			
			try:
				seg_volume = load_nifti(Path("data/BraTS2021_Training_Data") / patient_id / f"{patient_id}_seg.nii.gz")
			except Exception as e:
				print(f"   WARNING: Could not load segmentation for {patient_id}: {e}")
				seg_volume = np.zeros_like(images[0, 0].cpu().numpy(), shape=(240, 240, 155))
			
			logits = model(images)  # (num_slices, 1)
			probs = torch.sigmoid(logits).cpu().numpy().squeeze()
			preds = (probs > 0.5).astype(int)
			labels_np = labels.cpu().numpy()
			logits_np = logits.cpu().numpy().squeeze()
			
			# Handle single slice case
			if probs.ndim == 0:
				probs = np.array([probs])
				preds = np.array([preds])
				labels_np = np.array([labels_np])
				logits_np = np.array([logits_np])
			
			for slice_idx in range(len(preds)):
				gt = int(labels_np[slice_idx])
				pred = int(preds[slice_idx])
				logit = float(logits_np[slice_idx]) if logits_np.ndim > 0 else float(logits_np)
				
				# Compute IoU with segmentation
				try:
					seg_slice = seg_volume[:, :, slice_idx]
					iou = compute_iou_with_segmentation(
						images[slice_idx, 0].cpu().numpy(),
						seg_slice,
						pred,
						gt
					)["iou"]
				except:
					iou = 0.5  # Default if segmentation access fails
				
				predictions.append({
					"patient_id": patient_id,
					"slice_idx": slice_idx,
					"prediction": pred,
					"ground_truth": gt,
					"logit": logit,
					"iou": iou,
				})
				
				all_logits.append(logit)
				all_labels.append(gt)
				all_iou.append(iou)
	
	return predictions, np.array(all_logits), np.array(all_labels), np.array(all_iou)


def classify_predictions(
	predictions: List[Dict]
) -> Dict[str, List[Dict]]:
	"""Classify predictions into TP, FP, TN, FN.
	
	Returns:
		Dict mapping category name to list of predictions
	"""
	tp = [p for p in predictions if p["prediction"] == 1 and p["ground_truth"] == 1]
	fp = [p for p in predictions if p["prediction"] == 1 and p["ground_truth"] == 0]
	tn = [p for p in predictions if p["prediction"] == 0 and p["ground_truth"] == 0]
	fn = [p for p in predictions if p["prediction"] == 0 and p["ground_truth"] == 1]
	
	# Sort by IoU (descending) to get best quality samples first
	for category in [tp, fp, tn, fn]:
		category.sort(key=lambda x: x["iou"], reverse=True)
	
	return {
		"TP": tp,
		"FP": fp,
		"TN": tn,
		"FN": fn,
	}


def select_diverse_patients(
	predictions: List[Dict],
	num_patients: int = 10,
	slices_per_patient: int = 2,
	target_class: str = "TP"
) -> List[Dict]:
	"""Select diverse samples (TP or TN) from multiple patients.
	
	Args:
		target_class: "TP" (Tumor) or "TN" (No Tumor)
	"""
	import random
	
	# Filter based on target class
	if target_class == "TP":
		samples = [p for p in predictions if p["prediction"] == 1 and p["ground_truth"] == 1]
	elif target_class == "TN":
		samples = [p for p in predictions if p["prediction"] == 0 and p["ground_truth"] == 0]
	else:
		return []
	
	# Group by patient
	patient_groups = {}
	for sample in samples:
		pid = sample["patient_id"]
		if pid not in patient_groups:
			patient_groups[pid] = []
		patient_groups[pid].append(sample)
	
	# Sort each patient's samples by IoU (descending)
	# For TN, IoU is usually 0 or undefined, so sort by confidence (logit) instead?
	# Actually, IoU for TN is calculated as "intersection of background".
	# Let's stick to IoU for consistency, assuming compute_iou handles TN well.
	for pid in patient_groups:
		patient_groups[pid].sort(key=lambda x: x["iou"], reverse=True)
	
	# Randomly select num_patients
	selected_patients = random.sample(list(patient_groups.keys()), 
	                                   min(num_patients, len(patient_groups)))
	
	# Get max slices_per_patient from each selected patient
	diverse_samples = []
	for pid in selected_patients:
		diverse_samples.extend(patient_groups[pid][:slices_per_patient])
	
	# Add category label for later use
	for sample in diverse_samples:
		sample["category"] = target_class
		
	return diverse_samples


def create_visualization(
	slice_img: np.ndarray,
	heatmap: np.ndarray,
	segmentation: np.ndarray,
	prediction: int,
	ground_truth: int,
	iou: float
) -> Image.Image:
	"""Create 3-column visualization with VISIBLE green and red overlays:
	Column 1: MRI slice + ground truth segmentation (bright green outline)
	Column 2: MRI slice + GradCAM heatmap (hot colormap)
	Column 3: All three combined (MRI + green seg + red GradCAM)
	"""
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	
	# Normalize images for visualization using PERCENTILE (robust to outliers)
	# Min-max gets crushed by bright artifacts; percentile makes brain visible
	v_max = np.percentile(slice_img, 99.5)
	slice_norm = np.clip(slice_img / (v_max + 1e-8), 0, 1)
	
	# Normalize heatmap to [0, 1]
	if heatmap.max() > heatmap.min():
		heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
	else:
		heatmap_norm = heatmap
	
	# Binary segmentation mask
	seg_binary = (segmentation > 0).astype(float)
	
	# Create proper color overlays using masked arrays
	# Green overlay for segmentation (bright, 0.7 alpha)
	green_overlay = np.zeros((*seg_binary.shape, 3), dtype=float)
	green_overlay[:, :, 1] = 1.0  # Green channel
	green_overlay[:, :, 0] = 0.0  # Red channel
	green_overlay[:, :, 2] = 0.0  # Blue channel
	
	# Hot colormap for GradCAM
	cmap_hot = cm.get_cmap('hot')
	heatmap_colored = cmap_hot(heatmap_norm)[:, :, :3]  # Get RGB, drop alpha
	
	# Column 1: Slice + Ground Truth Segmentation (bright green)
	axes[0].imshow(slice_norm, cmap="gray", alpha=1.0)  # Full opacity for brain scan
	# Overlay green only where segmentation is 1
	seg_overlay = green_overlay.copy()
	seg_overlay[seg_binary == 0] = 0  # Make transparent where no tumor
	axes[0].imshow(seg_overlay, alpha=0.7)  # High alpha to see green clearly
	axes[0].set_title("Ground Truth\n(Green)", fontsize=14, pad=10)
	axes[0].axis("off")
	
	# Column 2: Slice + GradCAM Heatmap (hot colormap)
	axes[1].imshow(slice_norm, cmap="gray", alpha=1.0)  # Full opacity for brain scan
	heatmap_overlay = heatmap_colored.copy()
	heatmap_overlay[heatmap_norm < 0.1] = np.array([0, 0, 0])  # Black where heatmap is weak
	axes[1].imshow(heatmap_overlay, alpha=0.7)  # High alpha to see red clearly
	axes[1].set_title("GradCAM Attention\n(Red)", fontsize=14, pad=10)
	axes[1].axis("off")
	
	# Column 3: All three combined (MRI base + green + red)
	axes[2].imshow(slice_norm, cmap="gray", alpha=1.0)  # Full opacity for brain scan
	# Green overlay
	seg_overlay = green_overlay.copy()
	seg_overlay[seg_binary == 0] = 0
	axes[2].imshow(seg_overlay, alpha=0.5)  # Semi-transparent green
	# Red overlay
	heatmap_overlay = heatmap_colored.copy()
	heatmap_overlay[heatmap_norm < 0.1] = np.array([0, 0, 0])
	axes[2].imshow(heatmap_overlay, alpha=0.5)  # Semi-transparent red
	axes[2].set_title("Combined\n(Green+Red)", fontsize=14, pad=10)
	axes[2].axis("off")
	
	# Add prediction info at top with clear positioning
	pred_text = "Tumor" if prediction == 1 else "No Tumor"
	gt_text = "Tumor" if ground_truth == 1 else "No Tumor"
	correct = "✓ CORRECT" if prediction == ground_truth else "✗ ERROR"
	status_color = "green" if prediction == ground_truth else "red"
	
	fig.suptitle(
		f"Prediction: {pred_text} | GT: {gt_text} | {correct} (IoU: {iou:.3f})",
		fontsize=15,
		color=status_color,
		y=0.98  # Position higher to avoid overlap
	)
	
	plt.subplots_adjust(top=0.85)  # Leave space for title
	
	# Convert to PIL Image using savefig + BytesIO
	import io
	buf = io.BytesIO()
	fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
	buf.seek(0)
	image = Image.open(buf)
	image.load()  # Force load before closing buffer
	plt.close(fig)
	
	return image


def main():
	import argparse
	
	parser = argparse.ArgumentParser(description="Run XAI pipeline on trained models")
	parser.add_argument("--model", type=str, default="efficientnet_b2",
	                    choices=["efficientnet_b2", "resnet50", "deit_small", "deit_base"],
	                    help="Which model to analyze")
	parser.add_argument("--num-patients", type=int, default=10,
	                    help="Number of test patients to analyze")
	args = parser.parse_args()
	
	print("\n" + "="*80)
	print("XAI PIPELINE: Sample Selection & GradCAM Visualization")
	print("="*80 + "\n")
	
	# Configuration
	model_name = args.model
	task1_dir = Path("data/BraTS2021_Training_Data")
	results_dir = Path("experiments_logs")
	output_dir = results_dir / "xai_analysis" / model_name
	output_dir.mkdir(parents=True, exist_ok=True)
	
	print(f"Model: {model_name}")
	print(f"Output: {output_dir}\n")
	
	# Load model
	print("1. Loading model checkpoint...")
	model = create_model(model_name)
	
	# Find the latest timestamped checkpoint directory for this model
	checkpoint_found = False
	checkpoint_path = None
	
	# Look for timestamped directories (e.g., efficientnet_b2_20260114_230340)
	tumor_detection_dir = results_dir / "tumor_detection"
	if tumor_detection_dir.exists():
		# Find all directories matching model_name_*
		model_dirs = sorted([d for d in tumor_detection_dir.iterdir() 
		                    if d.is_dir() and d.name.startswith(f"{model_name}_")],
		                   reverse=True)  # Most recent first
		
		for model_dir in model_dirs:
			checkpoint_dir = model_dir / "checkpoints"
			if checkpoint_dir.exists():
				# Find best checkpoint
				best_ckpts = list(checkpoint_dir.glob(f"{model_name}_best*.ckpt"))
				if best_ckpts:
					checkpoint_path = best_ckpts[0]
					print(f"   ✓ Found checkpoint: {checkpoint_path}")
					checkpoint_found = True
					break
	
	if not checkpoint_found:
		print("   WARNING: No checkpoint found. Testing with random weights.")
	
	# Load checkpoint if found
	if checkpoint_found and checkpoint_path:
		try:
			# Create a temporary training config for module initialization
			from src.training import TrainingConfig
			temp_config = TrainingConfig()
			lightning_module = TumorDetectionModule.load_from_checkpoint(
				checkpoint_path, 
				model=model,
				config=temp_config
			)
			model = lightning_module.model
			print(f"   ✓ Loaded weights from checkpoint")
		except Exception as e:
			print(f"   WARNING: Could not load checkpoint weights: {e}")
			print(f"   (Using model without pretrained weights)")
	
	model.eval()
	print("   ✓ Model loaded\n")
	
	# Prepare data (use num_patients from args)
	print("2. Preparing test data...")
	patient_dirs = find_patient_dirs(task1_dir, num_patients=args.num_patients)
	config = DatasetConfig(
		modalities=["t1ce", "t2", "flair"],  # 3 channels: most informative for tumor detection
		image_size=(224, 224),
	)
	_, val_loader, _ = create_patient_centric_data_loaders(patient_dirs, config)
	print(f"   ✓ Loaded {len(patient_dirs)} patients\n")
	
	# Run inference
	print("3. Running inference...")
	predictions, logits, labels, iou_scores = collect_predictions(model, val_loader)
	print(f"   ✓ Processed {len(predictions)} slices\n")
	
	# Classify and select samples
	print("4. Classifying predictions...")
	classified = classify_predictions(predictions)
	for category, preds in classified.items():
		print(f"   {category}: {len(preds)} samples")

	print("\n5. Selecting diverse samples (TP and TN)...")
	# Select max 2 slices from 20 random patients for TP
	diverse_tp = select_diverse_patients(predictions, num_patients=20, slices_per_patient=2, target_class="TP")
	# Select max 2 slices from 5 random patients for TN (No Tumor)
	diverse_tn = select_diverse_patients(predictions, num_patients=5, slices_per_patient=2, target_class="TN")
	
	all_samples = diverse_tp + diverse_tn
	print(f"   Total TP samples: {len(classified['TP'])} | Selected: {len(diverse_tp)}")
	print(f"   Total TN samples: {len(classified['TN'])} | Selected: {len(diverse_tn)}")
	print(f"   ✓ Selected {len(all_samples)} total samples for visualization\n")
	
	# Initialize GradCAM
	print("6. Initializing GradCAM...")
	# Use single best layer per model
	if model_name == "resnet50":
		target_layer = "layer4"  # Final layer - shows what model ACTUALLY learned (the truth)
	elif model_name == "efficientnet_b2":
		target_layer = "features.8"  # Final feature layer
	else:  # deit_small
		target_layer = "blocks"  # Let it find the last conv
	
	gradcam = GradCAM(model, target_layer_name=target_layer)
	print()
	
	# Generate visualizations
	print("7. Generating GradCAM visualizations...")
	viz_dir = output_dir / "visualizations"
	viz_dir.mkdir(parents=True, exist_ok=True)
	
	metadata_list = []
	viz_count = 0
	
	for sample in all_samples:
		patient_id = sample["patient_id"]
		slice_idx = sample["slice_idx"]
		category = sample.get("category", "Unknown")
		
		# Load patient data
		patient_dir = task1_dir / patient_id
		modality_files = {
			"t1ce": patient_dir / f"{patient_id}_t1ce.nii.gz",
			"t2": patient_dir / f"{patient_id}_t2.nii.gz",
			"flair": patient_dir / f"{patient_id}_flair.nii.gz",
		}
		
		try:
			# Load and extract slice (using t1ce for visualization)
			slice_img = load_nifti(modality_files["t1ce"])[:, :, slice_idx]
			
			# Load segmentation for this slice
			seg_file = patient_dir / f"{patient_id}_seg.nii.gz"
			seg_volume = load_nifti(seg_file)
			seg_slice = seg_volume[:, :, slice_idx]
			
			# Forward pass for GradCAM
			modalities = {mod: load_nifti(path)[:, :, slice_idx] for mod, path in modality_files.items()}
			
			# Stack modalities (normalized) - 3 channels: t1ce, t2, flair
			from src.preprocessing import normalize_volume
			stacked = np.stack([
				normalize_volume(modalities[m]) for m in ["t1ce", "t2", "flair"]
			])
			
			# Resize to 224x224 if needed
			from torchvision import transforms
			resize_transform = transforms.Compose([
				transforms.Resize((224, 224))
			])
			
			stacked_tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
			stacked_resized = resize_transform(stacked_tensor).squeeze(0).numpy()
			
			img_tensor = torch.tensor(
				stacked_resized,
				dtype=torch.float32,
				requires_grad=True
			).unsqueeze(0).cuda()
			
			# Forward pass with gradients enabled
			output = model(img_tensor)
			cam = gradcam.generate_cam(img_tensor, output)
			
			# Resize CAM back to original slice size for visualization
			cam_resized = cv2.resize(cam, (slice_img.shape[1], slice_img.shape[0]), interpolation=cv2.INTER_CUBIC)
			
			# FILTERING: Skip uninformative samples
			
			# 1. Skip slices with insufficient brain tissue ONLY for TP
			# For TN, we want to see model behavior even on small/apical slices
			brain_mask = slice_img > 10
			brain_area = brain_mask.sum()
			
			if category == "TP":
				if brain_area < 4000:  # Approx 8% of image area
					continue
				
				# 2. Skip slices with tiny tumor area ONLY for TP
				seg_slice_binary = (seg_slice > 0).astype(int)
				tumor_area = seg_slice_binary.sum()
				if tumor_area < 50:  # Relaxed threshold for more samples
					continue
			
			# For TN, we just need to ensure it's not a completely empty black image
			elif category == "TN":
				if slice_img.max() < 10:  # Skip completely black images
					continue
			
			# Create visualization
			viz_img = create_visualization(
				slice_img,
				cam_resized,
				seg_slice,
				sample["prediction"],
				sample["ground_truth"],
				sample["iou"]
			)
			
			# Save with Category prefix
			filename = f"{category}_{patient_id}_slice{slice_idx:03d}_iou{sample['iou']:.3f}.png"
			filepath = viz_dir / filename
			viz_img.save(filepath)
			
			metadata_list.append({
				"filename": filename,
				"patient_id": patient_id,
				"slice_idx": slice_idx,
				"prediction": sample["prediction"],
				"ground_truth": sample["ground_truth"],
				"iou": sample["iou"],
				"category": category
			})
			
			print(f"   ✓ {filename}")
			viz_count += 1
		
		except Exception as e:
			print(f"   ✗ Error processing {patient_id} slice {slice_idx}: {str(e)[:60]}")

	# Save metadata
	metadata_path = output_dir / "metadata.json"
	with open(metadata_path, "w") as f:
		json.dump({
"model": model_name,
"total_visualizations": viz_count,
"samples": metadata_list,
}, f, indent=2)

	print(f"\n✓ Generated {viz_count} visualizations")
	print(f"✓ Saved metadata: {metadata_path}")
	print(f"✓ Output directory: {output_dir}\n")


if __name__ == "__main__":
	main()
