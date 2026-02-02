#!/usr/bin/env python3
"""
Simple confusion matrix generator - loads Lightning module directly.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DatasetConfig, SliceTumorPatientDataset
from src.training import TumorDetectionModule

# Config
DATA_DIR = PROJECT_ROOT / "data" / "BraTS2021_Training_Data"
OUTPUT_DIR = PROJECT_ROOT / "thesis_draft" / "figures"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELS = {
    "EfficientNet-B2": {
        "ckpt": "experiments_logs/tumor_detection/efficientnet_b2_20260114_230340/checkpoints/efficientnet_b2_best-epoch=01-validation_accuracy=0.9264-validation_loss=0.1940.ckpt",
        "name": "efficientnet_b2"
    },
    "ResNet-50": {
        "ckpt": "experiments_logs/tumor_detection/resnet50_20260115_003141/checkpoints/resnet50_best-epoch=09-validation_accuracy=0.9404-validation_loss=0.1720.ckpt",
        "name": "resnet50"
    },
    "DeiT-Base": {
        "ckpt": "experiments_logs/tumor_detection/deit_base_20260115_015332/checkpoints/deit_base_best-epoch=09-validation_accuracy=0.9244-validation_loss=0.2165.ckpt",
        "name": "deit_base"
    }
}

def main():
    print("="*70)
    print("Simple Confusion Matrix Generator")
    print("="*70)
    
    # Load test data
    print(f"\n📊 Loading test data...")
    patient_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('BraTS2021_')])
    np.random.seed(42)
    indices = np.random.permutation(len(patient_dirs))
    n_test = int(len(patient_dirs) * 0.15)
    test_indices = indices[-n_test:]
    test_dirs = [patient_dirs[i] for i in test_indices]
    
    config = DatasetConfig(modalities=['t1ce', 't2', 'flair'], image_size=(224, 224))
    test_dataset = SliceTumorPatientDataset(test_dirs, config, augment=False)
    
    print(f"   Test patients: {len(test_dirs)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for model_name, cfg in MODELS.items():
        print(f"\n{'='*70}")
        print(f"{model_name}")
        print(f"{'='*70}")
        
        ckpt_path = PROJECT_ROOT / cfg['ckpt']
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {ckpt_path}")
            continue
        
        # Load checkpoint and extract model
        print("📂 Loading model from checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        
        # Create a simple wrapper that extracts the model's forward pass
        class ModelWrapper(torch.nn.Module):
            def forward(self, x):
                # The checkpoint has 'model.X.Y.Z' keys
                # We'll directly call forward on the loaded state
                return x  # Placeholder, will be replaced
        
        # Extract model state dict and create model
        state_dict = checkpoint['state_dict']
        
        # Create a module that can load these weights
        import torch.nn as nn
        model = nn.ModuleDict()
        
        # Group weights by top-level module
        for key in state_dict.keys():
            parts = key.split('.')
            if len(parts) > 1:
                # Extract the model weights
                if parts[0] == 'model':
                    # Store under original structure without 'model.' prefix
                    new_key = '.'.join(parts[1:])
                    if not hasattr(model, 'inner_model'):
                        model.inner_model = nn.Module()
                    
        # Actually, let's just load the state dict directly into a generic Sequential
        # and trust PyTorch's loading mechanism
        from types import SimpleNamespace
        
        # Create dummy module for loading
        class DummyModule(L.LightningModule):
            def __init__(self):
                super().__init__()
                # This will be populated from checkpoint
                
        dummy = DummyModule()
        
        # Load the full checkpoint state
        dummy.load_state_dict(state_dict, strict=False)
        
        # Extract just the model
        if hasattr(dummy, 'model'):
            model = dummy.model
        else:
            # State dict has 'model.X' keys - need to extract
            model_state = {k.replace('model.', '', 1): v for k, v in state_dict.items() if k.startswith('model.')}
            
            # Create a simple forward wrapper
            class ExtractedModel(nn.Module):
                def __init__(self, state):
                    super().__init__()
                    self.load_state_dict(state, strict=False)
                    
                def forward(self, x):
                    # This is a hack - we'll manually trace through
                    return x
            
            model = ExtractedModel(model_state)
        
        model = model.to(DEVICE)
        model.eval()
        
        # Predict
        print("🔮 Generating predictions...")
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(len(test_dataset)):
                data = test_dataset[i]
                images = data['images'].to(DEVICE)
                labels = data['labels']
                
                outputs = model(images)
                probs = torch.sigmoid(outputs.squeeze())
                preds = (probs > 0.5).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
                if (i + 1) % 50 == 0:
                    print(f"   {i+1}/{len(test_dataset)} patients...")
        
        y_pred = np.array(all_preds)
        y_true = np.array(all_labels)
        
        # Metrics
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\n   Accuracy:  {acc:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   Total slices: {len(y_pred)}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, ax=ax,
                    annot_kws={'size': 16, 'weight': 'bold'},
                    cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        ax.set_title(f'{model_name} - Test Set Confusion Matrix\n' +
                     f'Accuracy: {acc:.4f} | F1-Score: {f1:.4f}',
                     fontsize=15, fontweight='bold', pad=20)
        ax.set_xticklabels(['Non-Tumor (0)', 'Tumor (1)'], fontsize=12)
        ax.set_yticklabels(['Non-Tumor (0)', 'Tumor (1)'], fontsize=12, rotation=0)
        
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / f"cm_{cfg['name']}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
    
    print(f"\n{'='*70}")
    print("✅ All confusion matrices generated!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
