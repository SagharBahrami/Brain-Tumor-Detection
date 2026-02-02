#!/usr/bin/env python3
"""
Generate confusion matrices for all three models on the test set.
Saves publication-quality figures for the thesis.
Uses slice-level metrics (not patient-level aggregation).
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DatasetConfig, SliceTumorPatientDataset
from src.models import create_model

# Paths
CHECKPOINTS_BASE = PROJECT_ROOT / "experiments_logs" / "tumor_detection"
DATA_DIR = PROJECT_ROOT / "data" / "BraTS2021_Training_Data"
OUTPUT_DIR = PROJECT_ROOT / "thesis_draft" / "figures"

# Model configurations
MODELS = {
    "EfficientNet-B2": {
        "experiment_dir": "efficientnet_b2_20260114_230340",
        "arch": "efficientnet_b2",
        "color": "#1f77b4"
    },
    "ResNet-50": {
        "experiment_dir": "resnet50_20260115_003141",
        "arch": "resnet50",
        "color": "#ff7f0e"
    },
    "DeiT-Base": {
        "experiment_dir": "deit_base_20260115_015332",
        "arch": "deit_base",
        "color": "#2ca02c"
    }
}


def load_model_and_predict(checkpoint_path, model_arch, test_dataset, device='cuda'):
    """Load model from checkpoint and generate slice-level predictions on test set."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract the wrapped model directly from checkpoint
    # The state dict has keys like 'model.features.X.Y' which means
    # we need to load the full model with the 'model.' wrapper
    from timm import create_model as timm_create_model
    from torchvision import models as tv_models
    
    # Create the base model from timm or torchvision
    if model_arch == 'efficientnet_b2':
        base_model = timm_create_model('efficientnet_b2', pretrained=False, num_classes=0)
        # Add classifier
        model = torch.nn.Sequential(
            base_model,
            torch.nn.Flatten(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(base_model.num_features, 1)
        )
    elif model_arch == 'resnet50':
        base_model = tv_models.resnet50(pretrained=False)
        model = torch.nn.Sequential(
            torch.nn.Sequential(*list(base_model.children())[:-1]),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 1)
        )
    elif model_arch == 'deit_base':
        base_model = timm_create_model('deit_base_patch16_224', pretrained=False, num_classes=0)
        model = torch.nn.Sequential(
            base_model,
            torch.nn.Linear(base_model.num_features, 1)
        )
    else:
        raise ValueError(f"Unknown model: {model_arch}")
    
    # Load state_dict - handle the 'model.' prefix
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('model.'):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"   Processing {len(test_dataset)} test patients (slice-by-slice)...")
    
    with torch.no_grad():
        for patient_idx in range(len(test_dataset)):
            # Load one patient (returns all slices)
            patient_data = test_dataset[patient_idx]
            images = patient_data['images'].to(device)  # [num_slices, 3, 224, 224]
            labels = patient_data['labels'].to(device)  # [num_slices]
            
            # Forward pass on all slices from this patient
            outputs = model(images)
            
            # Convert logits to slice-level predictions
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs > 0.5).float()
            
            # Store slice-level predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (patient_idx + 1) % 50 == 0:
                print(f"      Processed {patient_idx + 1}/{len(test_dataset)} patients...")
    
    print(f"   Total slices evaluated: {len(all_preds)}")
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, model_name, output_path, color):
    """Generate and save a confusion matrix plot."""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar=True, square=True, ax=ax,
                annot_kws={'size': 16, 'weight': 'bold'},
                cbar_kws={'label': 'Count'})
    
    # Labels
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} - Test Set Confusion Matrix\n' +
                 f'Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}',
                 fontsize=15, fontweight='bold', pad=20)
    
    # Set tick labels
    ax.set_xticklabels(['Non-Tumor (0)', 'Tumor (1)'], fontsize=12)
    ax.set_yticklabels(['Non-Tumor (0)', 'Tumor (1)'], fontsize=12, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    """Generate confusion matrices for all models."""
    print("="*70)
    print("Generating Confusion Matrices for Tumor Detection Models")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Build dataset
    print(f"\n📊 Loading BraTS 2021 dataset...")
    
    # Load patient directories
    patient_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('BraTS2021_')])
    print(f"   Found {len(patient_dirs)} patients")
    
    # Split patients (same as training)
    np.random.seed(42)
    indices = np.random.permutation(len(patient_dirs))
    n_val = int(len(patient_dirs) * 0.15)
    n_test = int(len(patient_dirs) * 0.15)
    test_indices = indices[-n_test:]
    test_dirs = [patient_dirs[i] for i in test_indices]
    
    print(f"   Test patients: {len(test_dirs)}")
    
    dataset_config = DatasetConfig(
        modalities=['t1ce', 't2', 'flair'],
        image_size=(224, 224)
    )
    
    # Create patient-centric test dataset
    test_dataset = SliceTumorPatientDataset(test_dirs, dataset_config, augment=False)
    
    # Generate confusion matrices for each model
    results = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for model_name, config in MODELS.items():
        print(f"\n{'='*70}")
        print(f"Processing: {model_name}")
        print(f"{'='*70}")
        
        # Find the best checkpoint
        checkpoint_dir = CHECKPOINTS_BASE / config['experiment_dir'] / "checkpoints"
        best_checkpoints = list(checkpoint_dir.glob(f"{config['arch']}_best*.ckpt"))
        
        if not best_checkpoints:
            print(f"❌ No best checkpoint found in: {checkpoint_dir}")
            continue
        
        checkpoint_path = best_checkpoints[0]
        print(f"📂 Checkpoint: {checkpoint_path.name}")
        
        # Generate predictions
        print("🔮 Generating slice-level predictions on test set...")
        y_pred, y_true = load_model_and_predict(
            checkpoint_path, 
            config['arch'], 
            test_dataset, 
            device
        )
        
        # Create confusion matrix
        output_path = OUTPUT_DIR / f"cm_{config['arch']}.png"
        print(f"📊 Creating confusion matrix...")
        
        metrics = plot_confusion_matrix(
            y_true, y_pred, 
            model_name, 
            output_path, 
            config['color']
        )
        
        results[model_name] = metrics
        
        print(f"✅ Saved to: {output_path}")
        print(f"\n📈 Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"   {metrics['confusion_matrix']}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
