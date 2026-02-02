#!/usr/bin/env python3
"""Create training curves visualization for final document."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def plot_training_curves() -> None:
    """Create 2x3 subplot of training curves for both models including F1 score."""

    # Load metrics history - use more recent experiments
    experiment_dirs = list((PROJECT_ROOT / "experiments_logs").glob("*"))
    if not experiment_dirs:
        print("No experiment directories found in experiments_logs/")
        return
    
    # Sort by modification time, get most recent
    experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Try to find models from recent experiments
    resnet_path = None
    effnet_path = None
    simple_cnn_path = None
    vit_path = None
    
    for exp_dir in experiment_dirs[:10]:  # Check last 10 experiments
        if "resnet50" in exp_dir.name.lower():
            resnet_path = exp_dir / "metrics_history.csv"
        elif "efficientnet" in exp_dir.name.lower():
            effnet_path = exp_dir / "metrics_history.csv"
        elif "simple_cnn" in exp_dir.name.lower():
            simple_cnn_path = exp_dir / "metrics_history.csv"
        elif "vit" in exp_dir.name.lower():
            vit_path = exp_dir / "metrics_history.csv"
    
    # Load available metrics
    metrics_data = {}
    model_names = []
    
    if resnet_path and resnet_path.exists():
        metrics_data['ResNet50'] = pd.read_csv(resnet_path)
        model_names.append('ResNet50')
    
    if effnet_path and effnet_path.exists():
        metrics_data['EfficientNet-B2'] = pd.read_csv(effnet_path)
        model_names.append('EfficientNet-B2')
        
    if simple_cnn_path and simple_cnn_path.exists():
        metrics_data['SimpleCNN'] = pd.read_csv(simple_cnn_path)
        model_names.append('SimpleCNN')
        
    if vit_path and vit_path.exists():
        metrics_data['ViT-Small'] = pd.read_csv(vit_path)
        model_names.append('ViT-Small')
    
    if not metrics_data:
        print("No metrics_history.csv files found in recent experiments")
        return
    
    # Use first two available models for plotting
    if len(model_names) >= 2:
        model1, model2 = model_names[0], model_names[1]
        metrics1 = metrics_data[model1]
        metrics2 = metrics_data[model2]
    else:
        model1 = model_names[0]
        metrics1 = metrics_data[model1]
        model2 = model1  # Same model for both columns
        metrics2 = metrics1

    # Create figure with 3x2 layout (Loss, AUC, F1 for each model)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Training Progression Comparison (Including F1 Score)", fontsize=16, fontweight='bold')

    # Plot 1: Loss - Model 1
    ax = axes[0, 0]
    ax.plot(metrics1['epoch'], metrics1['train_loss_epoch'],
            label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(metrics1['epoch'], metrics1['val_loss'],
            label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{model1} - Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)

    # Plot 2: Loss - Model 2
    ax = axes[0, 1]
    ax.plot(metrics2['epoch'], metrics2['train_loss_epoch'],
            label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(metrics2['epoch'], metrics2['val_loss'],
            label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{model2} - Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)

    # Plot 3: AUC - Model 1
    ax = axes[1, 0]
    ax.plot(metrics1['epoch'], metrics1['train_auc_epoch'],
            label='Train AUC', linewidth=2, marker='o', markersize=4)
    ax.plot(metrics1['epoch'], metrics1['val_auc'],
            label='Val AUC', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(f'{model1} - AUC-ROC', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_ylim([0.45, 1.0])

    # Plot 4: AUC - Model 2
    ax = axes[1, 1]
    ax.plot(metrics2['epoch'], metrics2['train_auc_epoch'],
            label='Train AUC', linewidth=2, marker='o', markersize=4)
    ax.plot(metrics2['epoch'], metrics2['val_auc'],
            label='Val AUC', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(f'{model2} - AUC-ROC', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_ylim([0.45, 1.0])

    # Plot 5: F1 - Model 1
    ax = axes[2, 0]
    if 'train_f1_epoch' in metrics1.columns and 'val_f1_epoch' in metrics1.columns:
        ax.plot(metrics1['epoch'], metrics1['train_f1_epoch'],
                label='Train F1', linewidth=2, marker='o', markersize=4)
        ax.plot(metrics1['epoch'], metrics1['val_f1'],
                label='Val F1', linewidth=2, marker='s', markersize=4)
    else:
        ax.text(0.5, 0.5, 'F1 metrics not available\nin this experiment', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'{model1} - F1 Score', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_ylim([0.0, 1.0])

    # Plot 6: F1 - Model 2
    ax = axes[2, 1]
    if 'train_f1_epoch' in metrics2.columns and 'val_f1_epoch' in metrics2.columns:
        ax.plot(metrics2['epoch'], metrics2['train_f1_epoch'],
                label='Train F1', linewidth=2, marker='o', markersize=4)
        ax.plot(metrics2['epoch'], metrics2['val_f1'],
                label='Val F1', linewidth=2, marker='s', markersize=4)
    else:
        ax.text(0.5, 0.5, 'F1 metrics not available\nin this experiment', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'{model2} - F1 Score', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_ylim([0.0, 1.0])

    plt.tight_layout()

    # Save figure
    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_curves_with_f1.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves (with F1) saved to: {output_path}")

    # Print summary statistics including F1
    print("\n=== Training Summary ===")
    for model_name, metrics in metrics_data.items():
        print(f"\n{model_name}:")
        if 'val_auc_epoch' in metrics.columns:
            print(f"  Best Val AUC: {metrics['val_auc_epoch'].max():.4f} at epoch {metrics['val_auc_epoch'].idxmax()}")
            print(f"  Final Train AUC: {metrics['train_auc_epoch'].iloc[-1]:.4f}")
            print(f"  Final Val AUC: {metrics['val_auc_epoch'].iloc[-1]:.4f}")
        if 'val_f1_epoch' in metrics.columns:
            print(f"  Best Val F1: {metrics['val_f1_epoch'].max():.4f} at epoch {metrics['val_f1_epoch'].idxmax()}")
            print(f"  Final Train F1: {metrics['train_f1_epoch'].iloc[-1]:.4f}")
            print(f"  Final Val F1: {metrics['val_f1_epoch'].iloc[-1]:.4f}")


if __name__ == "__main__":
    plot_training_curves()
