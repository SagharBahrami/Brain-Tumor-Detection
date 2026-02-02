"""Generate confusion matrices by loading Lightning checkpoints properly."""

import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training import TumorDetectionModule, TrainingConfig
from src.models import create_model
from src.data_loader import SliceTumorPatientDataset, DatasetConfig

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = PROJECT_ROOT / "data" / "BraTS2021_Training_Data"
OUTPUT_DIR = PROJECT_ROOT / "thesis_draft" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model checkpoints
CHECKPOINTS = {
    'efficientnet_b2': PROJECT_ROOT / 'experiments_logs/tumor_detection/efficientnet_b2_20260114_230340/checkpoints/efficientnet_b2_best-epoch=01-validation_accuracy=0.9264-validation_loss=0.1940.ckpt',
    'resnet50': PROJECT_ROOT / 'experiments_logs/tumor_detection/resnet50_20260115_003141/checkpoints/resnet50_best-epoch=09-validation_accuracy=0.9404-validation_loss=0.1720.ckpt',
    'deit_base': PROJECT_ROOT / 'experiments_logs/tumor_detection/deit_base_20260115_015332/checkpoints/deit_base_best-epoch=09-validation_accuracy=0.9244-validation_loss=0.2165.ckpt',
}


def load_model_for_inference(checkpoint_path: Path, model_arch: str):
    """Load model from Lightning checkpoint for inference."""
    print(f"📂 Loading {model_arch} from checkpoint...")
    
    # Create a dummy model and config just for loading
    # The actual weights will come from checkpoint
    dummy_model = create_model(model_arch, in_channels=3, pretrained=False, dropout=0.2)
    dummy_config = TrainingConfig(
        model_name=model_arch,
        batch_size=16,
        num_epochs=10,
        learning_rate=0.001,
    )
    
    # Load the checkpoint using Lightning's method
    # This will override the dummy_model with actual trained weights
    loaded_module = TumorDetectionModule.load_from_checkpoint(
        str(checkpoint_path),
        model=dummy_model,  # Required arg
        config=dummy_config,  # Required arg
        strict=False
    )
    
    loaded_module.eval()
    loaded_module.to(DEVICE)
    
    return loaded_module


def predict_on_dataset(model, dataset):
    """Run inference and collect predictions."""
    print(f"🔮 Running inference on {len(dataset)} patients...")
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing patients"):
            batch = dataset[idx]
            images = batch['images'].to(DEVICE)  # [N_slices, 3, 224, 224]
            labels = batch['labels'].cpu().numpy()  # [N_slices]
            
            # Forward pass
            logits = model(images)  # [N_slices, 1]
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int).flatten()
            
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
    
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_path: Path):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'{model_name} - Test Set Confusion Matrix\n' +
              f'Accuracy: {acc:.4f} | F1-Score: {f1:.4f}',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved confusion matrix to {output_path}")
    print(f"   Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    
    return acc, f1


def main():
    print("=" * 60)
    print("Confusion Matrix Generation - Slice-Level Metrics")
    print("=" * 60)
    
    # Load test dataset
    print("\n📊 Loading test dataset...")
    dataset_config = DatasetConfig(
        modalities=['t1ce', 't2', 'flair'],
        image_size=224
    )
    
    test_dataset = SliceTumorPatientDataset(
        data_dir=str(DATA_DIR),
        split='test',
        config=dataset_config,
        seed=42
    )
    print(f"Test set: {len(test_dataset)} patients")
    
    # Process each model
    results = {}
    for model_arch, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {model_arch.upper()}")
        print(f"{'=' * 60}")
        
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {ckpt_path}")
            continue
        
        # Load model
        model = load_model_for_inference(ckpt_path, model_arch)
        
        # Get predictions
        y_true, y_pred = predict_on_dataset(model, test_dataset)
        
        # Generate confusion matrix
        output_file = OUTPUT_DIR / f"cm_{model_arch}.png"
        acc, f1 = plot_confusion_matrix(y_true, y_pred, model_arch.upper(), output_file)
        
        results[model_arch] = {'accuracy': acc, 'f1_score': f1}
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY - SLICE-LEVEL METRICS")
    print(f"{'=' * 60}")
    for model_arch, metrics in results.items():
        print(f"{model_arch.upper():20s} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
    
    print(f"\n✅ All confusion matrices saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
