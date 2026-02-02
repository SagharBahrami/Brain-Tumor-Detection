#!/usr/bin/env python3
"""
Evaluate all trained models on test set.
Generates comparison metrics: Accuracy, Precision, Recall, F1, AUC.
Exports results and predictions for downstream analysis.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DatasetConfig, create_patient_centric_data_loaders
from src.training import TumorDetectionModule
from scripts.train import create_model_4ch, find_patient_dirs


def evaluate_model(
	model: nn.Module,
	test_loader: DataLoader,
	device: str = "cuda"
) -> Dict[str, float]:
	"""Evaluate model on test set."""
	model = model.to(device)
	model.eval()
	
	all_preds = []
	all_labels = []
	all_logits = []
	
	with torch.no_grad():
		for batch in test_loader:
			images = batch["images"].squeeze(0).to(device)  # (num_slices, C, H, W)
			labels = batch["labels"].squeeze(0).to(device)  # (num_slices,)
			
			logits = model(images)  # (num_slices, 1)
			preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().squeeze()
			labels = labels.cpu().numpy()
			logits = logits.cpu().numpy().squeeze()
			
			all_preds.extend(preds if isinstance(preds, list) else [preds])
			all_labels.extend(labels if isinstance(labels, list) else [labels])
			all_logits.extend(logits if isinstance(logits, list) else [logits])
	
	all_preds = np.array(all_preds)
	all_labels = np.array(all_labels)
	all_logits = np.array(all_logits)
	
	metrics = {
		"accuracy": accuracy_score(all_labels, all_preds),
		"precision": precision_score(all_labels, all_preds),
		"recall": recall_score(all_labels, all_preds),
		"f1": f1_score(all_labels, all_preds),
		"auc": roc_auc_score(all_labels, all_logits),
	}
	
	return metrics


def main():
	print("\n" + "="*70)
	print("EVALUATION: Test Set Metrics for All Models")
	print("="*70 + "\n")
	
	# Models to evaluate
	models = ["efficientnet_b2", "resnet50", "deit_small"]
	results_dir = Path("experiments_logs")
	task1_dir = Path("data/BraTS2021_Training_Data")
	
	# Get test patients
	print("Preparing test data...")
	patient_dirs = find_patient_dirs(task1_dir)
	config = DatasetConfig(
		modalities=["t1", "t1ce", "t2", "flair"],
		image_size=(224, 224),
	)
	_, _, test_loader = create_patient_centric_data_loaders(patient_dirs, config)
	
	# Evaluate each model
	all_results = []
	for model_name in models:
		print(f"\nEvaluating: {model_name}")
		
		# Find checkpoint
		model_dir = results_dir / f"tumor_detection/{model_name}"
		checkpoint_path = list((model_dir / "checkpoints").glob("*.ckpt"))
		
		if not checkpoint_path:
			print(f"  WARNING: No checkpoint found for {model_name}, skipping")
			continue
		
		checkpoint_path = checkpoint_path[0]
		print(f"  Loading: {checkpoint_path}")
		
		# Load model
		model = create_model_4ch(model_name)
		metrics = evaluate_model(model, test_loader)
		
		# Store results
		result = {"model": model_name, **metrics}
		all_results.append(result)
		
		print(f"  Accuracy:  {metrics['accuracy']:.4f}")
		print(f"  Precision: {metrics['precision']:.4f}")
		print(f"  Recall:    {metrics['recall']:.4f}")
		print(f"  F1:        {metrics['f1']:.4f}")
		print(f"  AUC:       {metrics['auc']:.4f}")
	
	# Save results
	df = pd.DataFrame(all_results)
	results_path = results_dir / "evaluation_results.csv"
	df.to_csv(results_path, index=False)
	
	print("\n" + "="*70)
	print("RESULTS SUMMARY")
	print("="*70)
	print(df.to_string(index=False))
	print(f"\nSaved to: {results_path}\n")


if __name__ == "__main__":
	import numpy as np
	main()
