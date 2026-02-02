#!/usr/bin/env python3
"""
Train tumor detection model on BraTS 2021 data.

Patient-centric approach: Loads all slices from one patient per training step.
Uses batch_size=1 and num_workers=0 to avoid multiprocessing memory issues.

Supports multiple architectures for comparison:
  - efficientnet_b2 (fast baseline)
  - resnet50 (standard baseline)
  - deit_small (transformer, 12 layers)
  - deit_base (transformer, 26 layers - larger model)

Usage:
  uv run python scripts/train.py --model efficientnet_b2 --max-epochs 50
  uv run python scripts/train.py --model resnet50 --max-epochs 50
  uv run python scripts/train.py --model deit_small --max-epochs 50
  uv run python scripts/train.py --model deit_base --max-epochs 50
  uv run python scripts/train.py --model efficientnet_b2 --max-epochs 5 --num-patients 15  # Quick test
"""

import sys
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import timm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DatasetConfig, create_patient_centric_data_loaders
from src.training import train_model, TrainingConfig


def create_model(model_name: str, num_classes: int = 1) -> nn.Module:
	"""Create a standard 3-channel model (treating MRI as RGB). 
	
	No weight averaging or first-layer modifications. Uses pretrained ImageNet weights directly.
	Supports: efficientnet_b2, resnet50, deit_small, deit_base
	"""
	
	if model_name == "efficientnet_b2":
		model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
		model.classifier = nn.Sequential(
			nn.Dropout(p=0.4, inplace=True),
			nn.Linear(model.classifier[1].in_features, num_classes),
		)
		return model
	
	elif model_name == "resnet50":
		model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		return model
	
	elif model_name == "deit_small":
		model = timm.create_model("deit_small_patch16_224", pretrained=True, in_chans=3)
		model.head = nn.Linear(model.head.in_features, num_classes)
		return model
	
	elif model_name == "deit_base":
		model = timm.create_model("deit_base_patch16_224", pretrained=True, in_chans=3)
		model.head = nn.Linear(model.head.in_features, num_classes)
		return model
	
	else:
		raise ValueError(f"Unknown model: {model_name}")


def find_patient_dirs(task1_dir: Path, num_patients: int = None) -> List[Path]:
	"""Find all BraTS patient directories."""
	patient_dirs = sorted([p for p in task1_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS2021_")])
	
	if num_patients is not None:
		patient_dirs = patient_dirs[:num_patients]
	
	print(f"Found {len(patient_dirs)} patients")
	return patient_dirs


def main():
	parser = argparse.ArgumentParser(description="Train tumor detection on BraTS data (patient-centric)")
	parser.add_argument("--model", type=str, default="efficientnet_b2", 
	                    choices=["efficientnet_b2", "resnet50", "deit_small", "deit_base"],
	                    help="Model architecture")
	parser.add_argument("--max-epochs", type=int, default=50, help="Number of epochs")
	parser.add_argument("--freeze-epochs", type=int, default=5, help="Freeze backbone epochs")
	parser.add_argument("--num-patients", type=int, default=None, help="Limit to N patients (for testing)")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
	parser.add_argument("--fast-dev-run", action="store_true", help="Run 1 batch per train/val/test for validation")
	parser.add_argument("--task1-dir", type=Path, default=Path("data/BraTS2021_Training_Data"), help="BraTS Task 1 directory")
	args = parser.parse_args()
	
	print(f"\n{'='*70}")
	print(f"TRAINING: Tumor Detection ({args.model.upper()})")
	print(f"{'='*70}")
	print(f"Model: {args.model}")
	print(f"Epochs: {args.max_epochs} | Freeze: {args.freeze_epochs}")
	print(f"LR: {args.lr} | Weight decay: {args.weight_decay}")
	print(f"Data: {args.task1_dir}")
	print(f"Batch size: 1 | Workers: 0 (patient-centric)")
	if args.num_patients:
		print(f"Limit: {args.num_patients} patients (testing mode)")
	print()
	
	# Verify data directory exists
	if not args.task1_dir.exists():
		print(f"ERROR: Data directory not found: {args.task1_dir}")
		return 1
	
	# Find patient directories
	print("Scanning for patients...")
	patient_dirs = find_patient_dirs(args.task1_dir, args.num_patients)
	if not patient_dirs:
		print(f"ERROR: No patients found in {args.task1_dir}")
		return 1
	
	# Create data loaders (patient-centric)
	print("\nCreating data loaders...")
	config = DatasetConfig(
		modalities=["t1ce", "t2", "flair"],  # 3 channels: most informative for tumor detection
		image_size=(224, 224),
	)
	
	train_loader, val_loader, test_loader = create_patient_centric_data_loaders(
		patient_dirs,
		config,
		val_frac=0.15,
		test_frac=0.15,
	)
	
	# Create model
	print(f"\nCreating model: {args.model}...")
	model = create_model(args.model)
	
	# Train
	training_config = TrainingConfig(
		max_epochs=args.max_epochs,
		freeze_backbone_epochs=args.freeze_epochs,
		lr=args.lr,
		weight_decay=args.weight_decay,
	)
	
	print("\n🚀 Starting training...\n")
	trainer, run_dir = train_model(
		model, 
		train_loader, 
		val_loader, 
		training_config,
		model_name=args.model,
		fast_dev_run=args.fast_dev_run,
	)
	print(f"\n✓ Training complete! Results: {run_dir}\n")
	
	# Test evaluation (skip if fast_dev_run since no checkpoints are saved)
	if not args.fast_dev_run:
		print("\n🧪 Running test evaluation...\n")
		test_results = trainer.test(dataloaders=test_loader, ckpt_path="best")
		print(f"\n✓ Test complete! Results saved to: {run_dir}\n")
	else:
		print("\n⚠️  Skipping test evaluation (fast_dev_run mode)\n")
	
	return 0


if __name__ == "__main__":
	sys.exit(main())

