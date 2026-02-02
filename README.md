# Brain Tumor Slice-Level Detection

Binary classification of 2D MRI slices for tumor presence detection using deep learning with Grad-CAM explainability. Built on BraTS 2021 dataset (1,251 patients).

## Setup

```bash
git clone <repo>
cd brain-tumor-detection
uv sync
```

Download BraTS 2021 Task 1 data to `data/BraTS2021_Training_Data/`.

## Training

```bash
# Quick test (5 min)
uv run python scripts/train.py --model efficientnet_b2 --max-epochs 5 --num-patients 15

# Full training - all models
bash scripts/run_all_experiments.sh
```

**Models & Results**:
| Model | Test Accuracy | Test F1 |
|-------|---------------|---------|
| ResNet-50 | 94.69% | 93.72% |
| EfficientNet-B2 | 93.67% | 92.70% |
| DeiT-Base | 93.46% | 92.35% |

## Evaluation

```bash
uv run python scripts/evaluate_all.py
```

## Explainability (Grad-CAM)

```bash
uv run python scripts/xai_pipeline.py
```

Generates visualizations for TP/FP/TN/FN predictions with IoU-based sample selection.

## Key Details

- **Input**: 3-channel MRI (T1ce, T2, FLAIR) at 224x224
- **Data split**: Patient-centric (70/15/15 train/val/test)
- **Training**: AdamW, CosineAnnealingLR, 10 epochs with 5-epoch backbone freeze

## Project Structure

```
src/                  # Core modules (data loading, training, preprocessing)
scripts/              # Training, evaluation, XAI scripts
experiments_logs/     # Checkpoints, metrics, visualizations
thesis/               # LaTeX thesis document
```

## References

- BraTS 2021: Baid et al. (2021)
- Grad-CAM: Selvaraju et al. (2017)
