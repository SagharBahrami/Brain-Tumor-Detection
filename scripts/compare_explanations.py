from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.explainability import (  # noqa: E402
	TumorGradCAM,
	compute_iou_with_mask,
)
from src.models import create_model  # noqa: E402
from src.preprocessing import get_modality_paths, load_nifti, normalize_volume, stack_modalities  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compare Grad-CAM explanations across multiple models")

	parser.add_argument(
		"--configs",
		type=Path,
		required=True,
		help="Path to JSON file with model configurations",
	)
	parser.add_argument("--patient-dir", type=Path, required=True, help="Path to patient directory")
	parser.add_argument("--slice-idx", type=int, required=True, help="Slice index to visualize")
	parser.add_argument(
		"--modalities",
		nargs="+",
		default=["t1ce", "t2", "flair"],
		help="MRI modalities",
	)
	parser.add_argument("--true-label", type=int, choices=[0, 1], required=True, help="Ground truth tumor label")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("results/explanation_comparisons"),
		help="Output directory",
	)
	parser.add_argument("--threshold", type=float, default=0.5, help="Heatmap threshold for IoU")
	parser.add_argument("--image-size", type=int, default=224, help="Image size (must match training)")

	return parser.parse_args(argv)


def load_patient_slice(
	patient_dir: Path,
	slice_idx: int,
	modalities: List[str],
	image_size: tuple[int, int] = (224, 224),
) -> tuple[torch.Tensor, np.ndarray]:
	"""
	Load and preprocess a specific patient slice.

	Returns:
		image: Stacked modality tensor (C, H, W) resized to image_size
		segmentation: Tumor mask (H, W) in original resolution
	"""
	modality_paths = get_modality_paths(patient_dir, modalities)
	seg_path = get_modality_paths(patient_dir, ["seg"])["seg"]

	volumes = {}
	for modality, path in modality_paths.items():
		vol = load_nifti(path)
		vol_norm = normalize_volume(vol)
		volumes[modality] = vol_norm

	segmentation = load_nifti(seg_path)

	# Stack modalities with resizing (matches training preprocessing)
	stacked = stack_modalities(volumes, slice_idx, size=image_size)
	image = torch.from_numpy(stacked.astype(np.float32))

	seg_slice = segmentation[:, :, slice_idx]

	return image, seg_slice


def load_model_configs(config_path: Path) -> List[Dict]:
	"""Load model configurations from JSON file."""
	with open(config_path, "r") as f:
		configs = json.load(f)

	for config in configs:
		if "checkpoint" not in config or "model" not in config:
			raise ValueError(f"Each config must have 'checkpoint' and 'model' fields: {config}")
		config["checkpoint"] = Path(config["checkpoint"])

	return configs


def create_comparison_plot(
	results: Dict[str, dict],
	original_slice: np.ndarray,
	seg_slice: np.ndarray,
	patient_id: str,
	slice_idx: int,
	output_path: Path,
) -> None:
	"""
	Create side-by-side Grad-CAM comparison visualization.

	Shows original MRI, segmentation, and Grad-CAM heatmap for each model.
	"""
	num_models = len(results)
	fig, axes = plt.subplots(2, num_models + 1, figsize=(5 * (num_models + 1), 10))

	if num_models == 1:
		axes = axes.reshape(2, -1)

	# Extract background image (use first channel)
	if original_slice.ndim == 3:
		background = original_slice[0]
	else:
		background = original_slice

	# Normalize background to [0, 1]
	bg_min, bg_max = background.min(), background.max()
	if bg_max > bg_min:
		background_norm = (background - bg_min) / (bg_max - bg_min)
	else:
		background_norm = np.zeros_like(background)

	# Row 1: Original + Heatmaps
	axes[0, 0].imshow(background, cmap="gray")
	axes[0, 0].set_title(f"Original MRI\nPatient: {patient_id}\nSlice: {slice_idx}", fontsize=10)
	axes[0, 0].axis("off")

	for idx, (model_name, result) in enumerate(results.items(), start=1):
		axes[0, idx].imshow(result["heatmap"], cmap="jet", vmin=0, vmax=1)
		pred_class = int(result["prediction"] > 0.5)
		axes[0, idx].set_title(
			f"{model_name}\nPred: {result['prediction']:.3f} ({pred_class})",
			fontsize=10,
		)
		axes[0, idx].axis("off")

	# Row 2: Segmentation + Overlays
	axes[1, 0].imshow(background, cmap="gray", alpha=0.6)
	seg_binary = (seg_slice > 0).astype(float)
	axes[1, 0].imshow(seg_binary, cmap="Reds", alpha=0.5)
	axes[1, 0].set_title(f"Tumor Segmentation\nPixels: {(seg_slice > 0).sum()}", fontsize=10)
	axes[1, 0].axis("off")

	for idx, (model_name, result) in enumerate(results.items(), start=1):
		axes[1, idx].imshow(background, cmap="gray", alpha=0.6)
		axes[1, idx].imshow(seg_binary, cmap="Reds", alpha=0.3)
		axes[1, idx].imshow(result["heatmap"], cmap="jet", alpha=0.5)
		axes[1, idx].set_title(
			f"Seg + CAM\nIoU: {result['iou']:.3f}",
			fontsize=10,
		)
		axes[1, idx].axis("off")

	plt.suptitle(
		f"Grad-CAM Comparison - True Label: {results[list(results.keys())[0]]['true_label']}",
		fontsize=14,
		fontweight="bold",
	)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
	args = parse_args(argv)

	patient_id = args.patient_dir.name.split("_")[-1]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("=" * 80)
	print("GRAD-CAM COMPARISON TOOL")
	print("=" * 80)
	print(f"\nPatient: {patient_id}")
	print(f"Slice: {args.slice_idx}")
	print(f"Device: {device}")

	# Load patient data
	print(f"\nLoading patient data from {args.patient_dir}...")
	image, seg_slice = load_patient_slice(
		args.patient_dir,
		args.slice_idx,
		args.modalities,
		image_size=(args.image_size, args.image_size),
	)
	print(f"  Image shape: {image.shape}")
	print(f"  Tumor pixels: {(seg_slice > 0).sum()}")

	# Load model configurations
	model_configs = load_model_configs(args.configs)
	print(f"\nLoaded {len(model_configs)} model configurations:")
	for config in model_configs:
		print(f"  - {config.get('name', config['model'])}")

	# Generate Grad-CAM for each model
	results = {}

	print("\n" + "=" * 80)
	print("GENERATING GRAD-CAM HEATMAPS")
	print("=" * 80)

	for config in model_configs:
		model_name = config.get("name", config["model"])
		print(f"\n[{model_name}] Loading model...")

		model = create_model(
			name=config["model"],
			in_channels=len(args.modalities),
			pretrained=False,
			dropout=config.get("dropout", 0.2),
		)
		model.to(device)

		checkpoint = torch.load(config["checkpoint"], map_location=device)
		state_dict = checkpoint.get("state_dict", checkpoint)
		state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}
		model.load_state_dict(state_dict)
		model.eval()

		print(f"[{model_name}] Generating Grad-CAM...")
		cam = TumorGradCAM(model, config["model"], device)

		# Get prediction
		with torch.no_grad():
			logits = model(image.unsqueeze(0).to(device)).squeeze()
			prediction = torch.sigmoid(logits).item()

		# Generate heatmap
		heatmap = cam.generate_heatmap(image)

		# Compute IoU
		iou = compute_iou_with_mask(heatmap, seg_slice, threshold=args.threshold)

		results[model_name] = {
			"heatmap": heatmap,
			"prediction": prediction,
			"iou": iou,
			"true_label": args.true_label,
			"checkpoint": str(config["checkpoint"]),
		}

		print(f"[{model_name}] Prediction: {prediction:.4f}, IoU: {iou:.4f}")

	# Create output directory
	output_dir = ensure_dir(args.output_dir / f"patient_{patient_id}")

	# Generate comparison visualization
	print("\n" + "=" * 80)
	print("GENERATING VISUALIZATION")
	print("=" * 80)

	viz_path = output_dir / f"slice_{args.slice_idx}_comparison.png"
	create_comparison_plot(
		results=results,
		original_slice=image.cpu().numpy(),
		seg_slice=seg_slice,
		patient_id=patient_id,
		slice_idx=args.slice_idx,
		output_path=viz_path,
	)
	print(f"Saved to: {viz_path}")

	# Create comparison table
	comparison_data = []
	for model_name, result in results.items():
		pred_class = int(result["prediction"] > 0.5)
		correct = pred_class == result["true_label"]

		comparison_data.append({
			"Model": model_name,
			"Prediction": result["prediction"],
			"Predicted_Class": pred_class,
			"True_Label": result["true_label"],
			"Correct": correct,
			"IoU": result["iou"],
			"IoU_Pass": result["iou"] > 0.5,  
		})

	comparison_df = pd.DataFrame(comparison_data)
	comparison_df = comparison_df.sort_values("IoU", ascending=False).reset_index(drop=True)

	csv_path = output_dir / f"slice_{args.slice_idx}_comparison.csv"
	comparison_df.to_csv(csv_path, index=False)

	# Save full results
	results_export = {
		model_name: {
			"prediction": float(result["prediction"]),
			"iou": float(result["iou"]),
			"checkpoint": result["checkpoint"],
		}
		for model_name, result in results.items()
	}

	results_json_path = output_dir / f"slice_{args.slice_idx}_results.json"
	with open(results_json_path, "w") as f:
		json.dump(results_export, f, indent=2)

	# Print summary
	print("\n" + "=" * 80)
	print("COMPARISON SUMMARY")
	print("=" * 80)
	print("\n" + comparison_df.drop(columns=["True_Label"]).to_string(index=False))

	# Determine best model by IoU
	best_model = comparison_df.iloc[0]["Model"]
	best_iou = comparison_df.iloc[0]["IoU"]
	print(f"\nBest model (by IoU): {best_model} (IoU = {best_iou:.4f})")

	print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
	main()
