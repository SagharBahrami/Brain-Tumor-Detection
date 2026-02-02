from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.explainability import (
    TumorGradCAM,
    compute_iou_with_mask,
    sanity_check_data_randomization,
    sanity_check_model_randomization,
    visualize_gradcam_overlay,
)
from src.models import create_model
from src.preprocessing import (
    get_modality_paths,
    load_nifti,
    normalize_volume,
    stack_modalities,
)
from src.utils import ensure_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations for trained tumor detection model")

	parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint")
	parser.add_argument("--model", type=str, required=True, help="Model architecture (resnet50, efficientnet_b2, etc)")
	parser.add_argument("--patient-dir", type=Path, required=True, help="Path to patient directory (e.g., BraTS2021_00000)")
	parser.add_argument("--slice-idx", type=int, required=True, help="Slice index to visualize")
	parser.add_argument(
		"--modalities",
		nargs="+",
		default=["t1ce", "t2", "flair"],
		help="MRI modalities used during training",
	)
	parser.add_argument("--true-label", type=int, choices=[0, 1], required=True, help="Ground truth tumor label")
	parser.add_argument("--output-dir", type=Path, default=Path("results/explanations"), help="Output directory")
	parser.add_argument("--threshold", type=float, default=0.5, help="Heatmap threshold for IoU computation")
	parser.add_argument("--sanity-checks", action="store_true", help="Run sanity checks (model/data randomization)")
	parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (must match training)")
	parser.add_argument("--image-size", type=int, default=224, help="Image size (must match training)")
	parser.add_argument("--figure-width", type=float, default=18.0, help="Figure width for Grad-CAM output")
	parser.add_argument("--figure-height", type=float, default=6.0, help="Figure height for Grad-CAM output")
	parser.add_argument("--title-font-size", type=float, default=14.0, help="Title font size for Grad-CAM figures")
	parser.add_argument("--label-font-size", type=float, default=12.0, help="Label font size for Grad-CAM figures")
	parser.add_argument("--tick-font-size", type=float, default=10.0, help="Tick font size for Grad-CAM figures")

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
	# Load modalities
	modality_paths = get_modality_paths(patient_dir, modalities)
	seg_path = get_modality_paths(patient_dir, ["seg"])["seg"]

	volumes = {}
	for modality, path in modality_paths.items():
		vol = load_nifti(path)
		vol_norm = normalize_volume(vol)
		volumes[modality] = vol_norm

	segmentation = load_nifti(seg_path)

	# Stack modalities with resizing (matches training preprocessing)
	stacked = stack_modalities(volumes, slice_idx, size=image_size)  # Shape: (C, H, W)
	image = torch.from_numpy(stacked.astype(np.float32))

	seg_slice = segmentation[:, :, slice_idx]

	return image, seg_slice


def main(argv: Sequence[str] | None = None) -> None:
	args = parse_args(argv)

	patient_id = args.patient_dir.name.split("_")[-1]  # Extract ID from BraTS2021_00000
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print(f"Generating explanations for patient {patient_id}, slice {args.slice_idx}")
	print(f"Device: {device}")

	# Load model
	model = create_model(
		name=args.model,
		in_channels=len(args.modalities),
		pretrained=False,
		dropout=args.dropout,
	)
	model.to(device)

	checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
	state_dict = checkpoint.get("state_dict", checkpoint)
	state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}
	model.load_state_dict(state_dict)
	model.eval()

	# Load patient data
	print(f"Loading patient data from {args.patient_dir}")
	image, seg_slice = load_patient_slice(
		args.patient_dir,
		args.slice_idx,
		args.modalities,
		image_size=(args.image_size, args.image_size),
	)
	print(f"  Image shape: {image.shape}")
	print(f"  Segmentation shape: {seg_slice.shape}")
	print(f"  Tumor pixels in slice: {(seg_slice > 0).sum()}")

	# Create output directory
	output_dir = ensure_dir(args.output_dir / args.model / f"patient_{patient_id}")

	# Generate Grad-CAM
	print("\nGenerating Grad-CAM...")
	cam = TumorGradCAM(model, args.model, device)
	result = cam.generate_with_prediction(
		image=image,
		patient_id=patient_id,
		slice_idx=args.slice_idx,
		true_label=args.true_label,
	)

	print(f"  Prediction: {result.prediction:.4f}")
	print(f"  True label: {result.true_label}")
	print(f"  Predicted class: {int(result.prediction > 0.5)}")

	# Compute IoU with segmentation
	iou = compute_iou_with_mask(result.heatmap, seg_slice, threshold=args.threshold)
	print(f"  IoU with tumor mask: {iou:.4f} (target > 0.3)")

	# Visualize
	print("\nGenerating visualization...")
	viz_path = output_dir / f"slice_{args.slice_idx}_gradcam.png"
	visualize_gradcam_overlay(
		result=result,
		original_slice=image.cpu().numpy(),
		output_path=viz_path,
		show_segmentation=seg_slice,
		figure_size=(args.figure_width, args.figure_height),
		title_fontsize=args.title_font_size,
		label_fontsize=args.label_font_size,
		tick_fontsize=args.tick_font_size,
	)
	print(f"  Saved to: {viz_path}")

	# Save results
	results_data = {
		"patient_id": patient_id,
		"slice_idx": args.slice_idx,
		"model": args.model,
		"checkpoint": str(args.checkpoint),
		"prediction": float(result.prediction),
		"predicted_class": int(result.prediction > 0.5),
		"true_label": result.true_label,
		"correct": (result.prediction > 0.5) == result.true_label,
		"iou": float(iou),
		"threshold": args.threshold,
		"modalities": args.modalities,
	}

	results_path = output_dir / f"slice_{args.slice_idx}_results.json"
	with open(results_path, "w") as f:
		json.dump(results_data, f, indent=2)
	print(f"  Results saved to: {results_path}")

	# Sanity checks (optional)
	if args.sanity_checks:
		print("\nRunning sanity checks...")

		# Model randomization check
		print("  1. Model randomization test...")
		model_check = sanity_check_model_randomization(model, image, args.model, device)
		print(f"     Correlation: {model_check['correlation']:.4f} (should be < 0.3)")
		print(f"     Pass: {model_check['pass']}")

		# Data randomization check
		print("  2. Data randomization test...")
		data_check = sanity_check_data_randomization(model, image, args.model, device)
		print(f"     Correlation: {data_check['correlation']:.4f} (should be < 0.3)")
		print(f"     Pass: {data_check['pass']}")

		# Save sanity check results
		sanity_results = {
			"model_randomization": {
				"correlation": float(model_check["correlation"]),
				"pass": model_check["pass"],
			},
			"data_randomization": {
				"correlation": float(data_check["correlation"]),
				"pass": data_check["pass"],
			},
		}

		sanity_path = output_dir / f"slice_{args.slice_idx}_sanity_checks.json"
		with open(sanity_path, "w") as f:
			json.dump(sanity_results, f, indent=2)
		print(f"  Sanity check results saved to: {sanity_path}")

	print(f"\nDone! All outputs in: {output_dir}")


if __name__ == "__main__":
	main()