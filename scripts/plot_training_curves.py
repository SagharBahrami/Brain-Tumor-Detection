#!/usr/bin/env python3
"""
Plot training curves (loss and accuracy) from Lightning metrics.csv files.
Generates a publication-quality figure for the thesis.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Paths
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments_logs" / "tumor_detection"
OUTPUT_PATH = Path(__file__).parent.parent / "thesis_draft" / "figures" / "training_curves.png"

# Model configurations
MODELS = {
    "EfficientNet-B2": "efficientnet_b2_20260114_230340",
    "ResNet-50": "resnet50_20260115_003141",
    "DeiT-Base": "deit_base_20260115_015332"
}

COLORS = {
    "EfficientNet-B2": "#1f77b4",  # Blue
    "ResNet-50": "#ff7f0e",       # Orange
    "DeiT-Base": "#2ca02c"         # Green
}


def load_metrics(model_name, experiment_name):
    """Load and process metrics CSV for a model."""
    metrics_path = EXPERIMENTS_DIR / experiment_name / "lightning_logs" / "version_0" / "metrics.csv"
    
    if not metrics_path.exists():
        print(f"Warning: {metrics_path} not found")
        return None
    
    df = pd.read_csv(metrics_path)
    
    # Group by epoch to get one row per epoch
    # For each epoch, take the last recorded value (end of epoch)
    epoch_metrics = df.groupby('epoch').agg({
        'train_loss': 'last',
        'train_accuracy': 'last',
        'validation_loss': 'last',
        'validation_accuracy': 'last'
    }).reset_index()
    
    # Remove NaN rows (some epochs may not have all metrics)
    epoch_metrics = epoch_metrics.dropna(subset=['train_loss', 'validation_loss'])
    
    return epoch_metrics


def plot_training_curves():
    """Generate training curves figure - 3 separate plots, one per model."""
    # Load data for all models
    all_metrics = {}
    for model_name, exp_name in MODELS.items():
        metrics = load_metrics(model_name, exp_name)
        if metrics is not None:
            all_metrics[model_name] = metrics
    
    # Create 3 separate figures (one per model)
    for model_name, metrics in all_metrics.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        color = COLORS[model_name]
        epochs = metrics['epoch'].values
        
        # Plot 1: Training and Validation Loss
        ax1 = axes[0]
        ax1.plot(epochs, metrics['train_loss'], 
                label="Training Loss",
                color=color, linewidth=2.5, linestyle='-', marker='o', markersize=5)
        ax1.plot(epochs, metrics['validation_loss'], 
                label="Validation Loss",
                color=color, linewidth=2.5, linestyle='--', marker='s', markersize=5)
        
        ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Binary Cross-Entropy Loss', fontsize=13, fontweight='bold')
        ax1.set_title(f'{model_name}: Training and Validation Loss', fontsize=15, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(-0.5, 9.5)
        ax1.set_xticks(range(0, 10))
        
        # Plot 2: Training and Validation Accuracy
        ax2 = axes[1]
        ax2.plot(epochs, metrics['train_accuracy'], 
                label="Training Accuracy",
                color=color, linewidth=2.5, linestyle='-', marker='o', markersize=5)
        ax2.plot(epochs, metrics['validation_accuracy'], 
                label="Validation Accuracy",
                color=color, linewidth=2.5, linestyle='--', marker='s', markersize=5)
        
        ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax2.set_title(f'{model_name}: Training and Validation Accuracy', fontsize=15, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(-0.5, 9.5)
        ax2.set_xticks(range(0, 10))
        ax2.set_ylim(0.5, 1.0)
        
        plt.tight_layout()
        
        # Save individual model figure
        output_path = OUTPUT_PATH.parent / f"training_curves_{model_name.lower().replace('-', '_')}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ {model_name} training curves saved to {output_path}")
        plt.close()
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    for model_name, metrics in all_metrics.items():
        final_epoch = metrics.iloc[-1]
        print(f"\n{model_name} (Epoch {int(final_epoch['epoch'])}):")
        print(f"  Train Loss: {final_epoch['train_loss']:.4f} | Val Loss: {final_epoch['validation_loss']:.4f}")
        print(f"  Train Acc:  {final_epoch['train_accuracy']:.4f} | Val Acc:  {final_epoch['validation_accuracy']:.4f}")


if __name__ == "__main__":
    plot_training_curves()
