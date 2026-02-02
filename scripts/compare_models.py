from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DatasetConfig, build_dataloaders  # noqa: E402
from src.evaluation import evaluate_model  # noqa: E402
from src.models import create_model  # noqa: E402
from src.utils import ensure_dir, load_patient_manifest  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compare multiple trained tumor detection models")

	parser.add_argument(
		"--configs",
		type=Path,
		required=True,
		help="Path to JSON file with model configurations (see example below)",
	)
	parser.add_argument("--task1-dir", type=Path, required=True, help="Path to BraTS 2021 Task 1 directory")
	parser.add_argument("--labels-csv", type=Path, required=True, help="Path to RSNA-MICCAI labels CSV")
	parser.add_argument(
		"--modalities",
		nargs="+",
		default=["t1ce", "t2", "flair"],
		help="MRI modalities to use",
	)
	parser.add_argument("--manifest", type=Path, default=None, help="Optional patient manifest")

	parser.add_argument("--image-size", type=int, default=224, help="Image size")
	parser.add_argument("--num-slices", type=int, default=8, help="Slices per patient")
	parser.add_argument("--min-tumor-pixels", type=int, default=10, help="Minimum tumor pixels per slice")
	parser.add_argument("--no-prefer-middle", action="store_true", help="Disable middle-slice preference")
	parser.add_argument("--include-empty-slices", action="store_true", help="Include contextual empty slices")
	parser.add_argument("--max-empty-slices", type=int, default=2, help="Maximum empty slices per patient")
	parser.add_argument(
		"--background-slice-margin",
		type=int,
		default=2,
		help="Margin around tumor region for background slices",
	)

	parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
	parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker processes")
	parser.add_argument("--val-frac", type=float, default=0.15, help="Validation split fraction")
	parser.add_argument("--test-frac", type=float, default=0.15, help="Test split fraction")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splits")
	parser.add_argument(
		"--split",
		choices=["val", "test"],
		default="test",
		help="Dataset split to evaluate",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("results/comparisons"),
		help="Output directory for comparison results",
	)

	return parser.parse_args(argv)


def load_model_configs(config_path: Path) -> List[Dict]:
	"""
	Load model configurations from JSON file.

	Expected format:
	[
		{
			"name": "ResNet50",
			"model": "resnet50",
			"checkpoint": "results/models/20231109_120000/checkpoints/best.ckpt",
			"dropout": 0.2
		},
		{
			"name": "EfficientNet-B2",
			"model": "efficientnet_b2",
			"checkpoint": "results/models/20231109_130000/checkpoints/best.ckpt",
			"dropout": 0.2
		}
	]
	"""
	with open(config_path, "r") as f:
		configs = json.load(f)

	for config in configs:
		if "checkpoint" not in config or "model" not in config:
			raise ValueError(f"Each config must have 'checkpoint' and 'model' fields: {config}")
		config["checkpoint"] = Path(config["checkpoint"])

	return configs


def plot_roc_comparison(results: Dict[str, dict], output_path: Path) -> None:
	"""Generate comparison ROC curve plot."""
	fig, ax = plt.subplots(figsize=(10, 8))

	colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

	for idx, (model_name, result) in enumerate(results.items()):
		ax.plot(
			result["fpr"],
			result["tpr"],
			label=f"{model_name} (AUC = {result['auc']:.4f})",
			linewidth=2,
			color=colors[idx],
		)

	ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
	ax.set_xlabel("False Positive Rate", fontsize=12)
	ax.set_ylabel("True Positive Rate", fontsize=12)
	ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
	ax.legend(loc="lower right", fontsize=10)
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def plot_metrics_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
	"""Generate bar chart comparing metrics across models."""
	metrics = ["AUC-ROC", "Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]
	metrics_present = [m for m in metrics if m in comparison_df.columns]

	fig, ax = plt.subplots(figsize=(12, 6))

	x = np.arange(len(metrics_present))
	width = 0.8 / len(comparison_df)

	colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_df)))

	for idx, (_, row) in enumerate(comparison_df.iterrows()):
		values = [row[m] for m in metrics_present]
		offset = (idx - len(comparison_df) / 2) * width + width / 2
		ax.bar(x + offset, values, width, label=row["Model"], color=colors[idx])

	ax.set_xlabel("Metrics", fontsize=12)
	ax.set_ylabel("Score", fontsize=12)
	ax.set_title("Performance Metrics Comparison", fontsize=14, fontweight="bold")
	ax.set_xticks(x)
	ax.set_xticklabels(metrics_present, rotation=45, ha="right")
	ax.set_ylim([0, 1.05])
	ax.legend(loc="lower right", fontsize=10)
	ax.grid(True, alpha=0.3, axis="y")

	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
	args = parse_args(argv)

	print("=" * 80)
	print("MODEL COMPARISON TOOL")
	print("=" * 80)

	# Load model configurations
	model_configs = load_model_configs(args.configs)
	print(f"\nLoaded {len(model_configs)} model configurations:")
	for config in model_configs:
		print(f"  - {config.get('name', config['model'])}: {config['checkpoint']}")

	# Load manifest if provided
	manifest_list = None
	if args.manifest is not None:
		loaded_manifest = load_patient_manifest(args.manifest)
		manifest_list = list(loaded_manifest) if loaded_manifest is not None else None

	# Build dataset
	dataset_config = DatasetConfig(
		task1_dir=args.task1_dir,
		labels_csv=args.labels_csv,
		modalities=args.modalities,
		image_size=(args.image_size, args.image_size),
		num_slices=args.num_slices,
		min_tumor_pixels=args.min_tumor_pixels,
		prefer_middle=not args.no_prefer_middle,
		include_empty_slices=args.include_empty_slices,
		max_empty_slices=args.max_empty_slices,
		background_slice_margin=args.background_slice_margin,
	)

	train_loader, val_loader, test_loader = build_dataloaders(
		config=dataset_config,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		val_frac=args.val_frac,
		test_frac=args.test_frac,
		seed=args.seed,
		manifest=manifest_list,
	)

	split_loader = val_loader if args.split == "val" else test_loader
	print(f"\nEvaluating on {args.split} split ({len(split_loader.dataset)} samples)")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	# Evaluate each model
	results = {}
	comparison_data = []

	print("\n" + "=" * 80)
	print("EVALUATION")
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

		print(f"[{model_name}] Evaluating...")
		eval_result = evaluate_model(model, split_loader, device)

		results[model_name] = {
			"auc": eval_result.auc,
			"accuracy": eval_result.accuracy,
			"precision": eval_result.precision,
			"recall": eval_result.recall,
			"f1": eval_result.f1,
			"specificity": eval_result.specificity,
			"fpr": eval_result.fpr,
			"tpr": eval_result.tpr,
			"confusion_matrix": eval_result.confusion_matrix.tolist(),
			"checkpoint": str(config["checkpoint"]),
		}

		comparison_data.append({
			"Model": model_name,
			"AUC-ROC": eval_result.auc,
			"Accuracy": eval_result.accuracy,
			"Precision": eval_result.precision,
			"Recall": eval_result.recall,
			"F1-Score": eval_result.f1,
			"Specificity": eval_result.specificity,
			"Checkpoint": str(config["checkpoint"]),
		})

		print(f"[{model_name}] AUC-ROC: {eval_result.auc:.4f}")

	# Create output directory
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	output_dir = ensure_dir(args.output_dir / f"comparison_{args.split}_{timestamp}")

	# Save comparison table
	comparison_df = pd.DataFrame(comparison_data)
	comparison_df = comparison_df.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)

	csv_path = output_dir / "comparison_table.csv"
	comparison_df.to_csv(csv_path, index=False)

	# Generate plots
	roc_path = output_dir / "roc_comparison.png"
	plot_roc_comparison(results, roc_path)

	metrics_path = output_dir / "metrics_comparison.png"
	plot_metrics_comparison(comparison_df, metrics_path)

	# Save full results JSON
	results_json_path = output_dir / "comparison_results.json"
	with open(results_json_path, "w") as f:
		json.dump(results, f, indent=2)

	# Print summary
	print("\n" + "=" * 80)
	print("COMPARISON SUMMARY")
	print("=" * 80)
	print("\n" + comparison_df.drop(columns=["Checkpoint"]).to_string(index=False))

	# Determine best model
	best_model = comparison_df.iloc[0]["Model"]
	best_auc = comparison_df.iloc[0]["AUC-ROC"]
	print(f"\nBest model (by AUC-ROC): {best_model} ({best_auc:.4f})")

	# Save metadata
	metadata = {
		"timestamp": datetime.now().isoformat(),
		"split": args.split,
		"num_models": len(model_configs),
		"dataset_size": len(split_loader.dataset),
		"modalities": args.modalities,
		"best_model": best_model,
		"best_auc": float(best_auc),
		"artifacts": {
			"comparison_table": str(csv_path),
			"roc_comparison": str(roc_path),
			"metrics_comparison": str(metrics_path),
			"full_results": str(results_json_path),
		},
	}

	metadata_path = output_dir / "comparison_metadata.json"
	with open(metadata_path, "w") as f:
		json.dump(metadata, f, indent=2)

	print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
	main()
