#!/usr/bin/env python3
"""
Generate confusion matrices for trained models.
Uses the EXACT same model creation logic as training (scripts/train.py).
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DatasetConfig, create_patient_centric_data_loaders


def create_model(model_name: str, num_classes: int = 1) -> nn.Module:
    """EXACT copy of create_model from scripts/train.py - this is what was used for training."""
    
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
    
    elif model_name == "deit_base":
        model = timm.create_model("deit_base_patch16_224", pretrained=True, in_chans=3)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def find_patient_dirs(task1_dir: Path) -> list:
    """Find all BraTS patient directories."""
    patient_dirs = sorted([p for p in task1_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS2021_")])
    return patient_dirs


def load_model_from_checkpoint(checkpoint_path: Path, model_name: str, device: str) -> nn.Module:
    """Load model weights from Lightning checkpoint."""
    
    # Create model with EXACT same structure as training
    model = create_model(model_name, num_classes=1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    
    # Remove 'model.' prefix from Lightning wrapper
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key.replace('model.', '', 1)
            model_state[new_key] = value
    
    # Load weights
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    
    return model


def generate_confusion_matrix(model_name: str, checkpoint_path: Path, test_loader, device: str, output_dir: Path):
    """Generate and save confusion matrix for a model."""
    
    print(f"\n{'='*70}")
    print(f"Processing {model_name.upper()}")
    print(f"{'='*70}")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load model
    print("📂 Loading model...")
    model = load_model_from_checkpoint(checkpoint_path, model_name, device)
    
    # Run inference
    print("🔮 Running inference on test set...")
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing patients"):
            # Batch is 1 patient -> dict with 'images' and 'labels'
            images = batch["images"].squeeze(0).to(device)  # [N_slices, 3, 224, 224]
            labels = batch["labels"].squeeze(0)  # [N_slices]
            
            # Forward pass
            logits = model(images).squeeze(-1)  # [N_slices]
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
    
    # Convert to arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"\n📊 Slice-Level Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'{model_name.upper()} - Slice-Level Confusion Matrix\n' +
              f'Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f"cm_{model_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved confusion matrix to {output_path}")
    
    return {'accuracy': accuracy, 'f1_score': f1}


def main():
    print("\n" + "="*70)
    print("CONFUSION MATRIX GENERATION - Slice-Level Tumor Detection")
    print("="*70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data paths
    task1_dir = PROJECT_ROOT / "data" / "BraTS2021_Training_Data"
    output_dir = PROJECT_ROOT / "thesis_draft" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print("\n📊 Loading test dataset...")
    patient_dirs = find_patient_dirs(task1_dir)
    config = DatasetConfig(modalities=["t1ce", "t2", "flair"], image_size=(224, 224))
    _, _, test_loader = create_patient_centric_data_loaders(
        patient_dirs, 
        config, 
        seed=42
    )
    print(f"Test set: {len(test_loader)} patients")
    
    # Model checkpoints
    checkpoints = {
        'efficientnet_b2': PROJECT_ROOT / 'experiments_logs/tumor_detection/efficientnet_b2_20260114_230340/checkpoints/efficientnet_b2_best-epoch=01-validation_accuracy=0.9264-validation_loss=0.1940.ckpt',
        'resnet50': PROJECT_ROOT / 'experiments_logs/tumor_detection/resnet50_20260115_003141/checkpoints/resnet50_best-epoch=09-validation_accuracy=0.9404-validation_loss=0.1720.ckpt',
        'deit_base': PROJECT_ROOT / 'experiments_logs/tumor_detection/deit_base_20260115_015332/checkpoints/deit_base_best-epoch=09-validation_accuracy=0.9244-validation_loss=0.2165.ckpt',
    }
    
    # Generate confusion matrices
    results = {}
    for model_name, checkpoint_path in checkpoints.items():
        results[model_name] = generate_confusion_matrix(
            model_name, checkpoint_path, test_loader, device, output_dir
        )
        
        # Free memory
        torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name.upper():20s} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
    
    print(f"\n✅ All confusion matrices saved to {output_dir}/")


if __name__ == "__main__":
    main()
