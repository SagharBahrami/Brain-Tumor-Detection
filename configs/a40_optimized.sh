#!/usr/bin/env bash
# Optimized hyperparameters for NVIDIA A40 GPU Server
# Server specs:
# - GPU: NVIDIA A40 (48GB VRAM, 10752 CUDA cores)
# - CPU: Intel Xeon Gold 6342 @ 2.80GHz (48 cores, 96 threads)
# - RAM: 46.57GB
# - OS: Ubuntu 22.04.5 LTS

# ============================================
# Optimized Batch Sizes for A40 (48GB VRAM)
# ============================================

# SimpleCNN (small model, ~1M params)
SIMPLE_CNN_BATCH=128        # Can go much larger
SIMPLE_CNN_WORKERS=24      # Use 50% of CPU cores
SIMPLE_CNN_ACCUM=1         # No accumulation needed

# ResNet50 (~23M params)
RESNET50_BATCH=96          # Large batch for faster training
RESNET50_WORKERS=24        # Parallel data loading
RESNET50_ACCUM=1

# EfficientNet-B2 (~7.7M params)
EFFICIENTNET_BATCH=96      # Memory efficient architecture
EFFICIENTNET_WORKERS=24
EFFICIENTNET_ACCUM=1

# DeiT-Small (~22M params)
DEIT_SMALL_BATCH=64        # Transformers use more memory
DEIT_SMALL_WORKERS=20
DEIT_SMALL_ACCUM=1

# DeiT-Base (~86M params)
DEIT_BASE_BATCH=32         # Largest model, more conservative
DEIT_BASE_WORKERS=16
DEIT_BASE_ACCUM=2          # Use gradient accumulation for effective batch=64

# ============================================
# Training Settings
# ============================================

EPOCHS=30                  # Increase from 20 (you have time!)
DROPOUT=0.2               # Standard for transfer learning
LEARNING_RATE=0.001       # Default AdamW
WEIGHT_DECAY=0.0001       # L2 regularization
FREEZE_EPOCHS=5           # Warmup classifier for 5 epochs

# ============================================
# Mixed Precision Training (Faster on A40)
# ============================================

# A40 supports Tensor Cores with mixed precision
# This can give 2-3x speedup with minimal accuracy loss
# Add to training command: --precision 16-mixed

# ============================================
# Data Loading Optimization
# ============================================

# With 48 cores, we can use many workers
# Rule of thumb: 4-8 workers per GPU, but with 48GB data we can go higher
# Set persistent_workers=True in DataLoader (already done)

# Enable pinned memory for faster GPU transfer (already done)
# This is critical with large datasets

# ============================================
# Memory Optimization Tips
# ============================================

# 1. If OOM occurs, reduce batch size by 50%
# 2. Enable gradient checkpointing for transformers (saves memory)
# 3. Use gradient accumulation to simulate larger batches

# ============================================
# Expected Training Times (with optimized batches)
# ============================================

# Dataset: ~40,000 training samples (all tumor slices)

# Model            | Batch | Time/Epoch | Total (30 epochs) | VRAM Usage
# -----------------|-------|------------|-------------------|------------
# SimpleCNN        |  128  |   ~90 sec  |   ~45 min         |   ~8 GB
# ResNet50         |   96  |   ~4 min   |   ~2 hours        |  ~18 GB
# EfficientNet-B2  |   96  |   ~5 min   |   ~2.5 hours      |  ~16 GB
# DeiT-Small       |   64  |   ~8 min   |   ~4 hours        |  ~24 GB
# DeiT-Base        |   32  |  ~15 min   |   ~7.5 hours      |  ~38 GB

# Total time (all 5 models sequentially): ~16-17 hours
# With optimizations: ~12-14 hours

# ============================================
# Usage Examples
# ============================================

# SimpleCNN (fastest)
: '
python scripts/train_model.py \
  --task1-dir data/BraTS2021_Training_Data \
  --labels-csv data/train_labels.csv \
  --model simple_cnn \
  --batch-size 128 \
  --num-workers 24 \
  --max-epochs 30 \
  --dropout 0.2
'

# ResNet50 (standard benchmark)
: '
python scripts/train_model.py \
  --task1-dir data/BraTS2021_Training_Data \
  --labels-csv data/train_labels.csv \
  --model resnet50 \
  --batch-size 96 \
  --num-workers 24 \
  --max-epochs 30 \
  --dropout 0.2
'

# EfficientNet-B2 (efficient)
: '
python scripts/train_model.py \
  --task1-dir data/BraTS2021_Training_Data \
  --labels-csv data/train_labels.csv \
  --model efficientnet_b2 \
  --batch-size 96 \
  --num-workers 24 \
  --max-epochs 30 \
  --dropout 0.2
'

# DeiT-Small (transformer)
: '
python scripts/train_model.py \
  --task1-dir data/BraTS2021_Training_Data \
  --labels-csv data/train_labels.csv \
  --model deit_small \
  --batch-size 64 \
  --num-workers 20 \
  --max-epochs 30 \
  --dropout 0.1
'

# DeiT-Base (largest model)
: '
python scripts/train_model.py \
  --task1-dir data/BraTS2021_Training_Data \
  --labels-csv data/train_labels.csv \
  --model deit_base \
  --batch-size 32 \
  --num-workers 16 \
  --max-epochs 30 \
  --dropout 0.1 \
  --accumulate-grad-batches 2
'

echo "A40 Optimized configurations loaded!"
echo "Use these batch sizes to maximize GPU utilization."
