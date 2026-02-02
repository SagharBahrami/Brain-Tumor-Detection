#!/usr/bin/env bash
set -euo pipefail

# Clean Slate 3-Channel Orchestrator
# Sequential training of 3 models with standard 3-channel input (T1ce, T2, FLAIR)
# No weight hacking - pure ImageNet pretrained weights
# Models: EfficientNet-B2, ResNet-50, DeiT-Base (DeiT-Small dropped)

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

EPOCHS=10
FREEZE_EPOCHS=5
LR=0.001
WEIGHT_DECAY=0.0001

LOG_DIR=experiments_logs/model_logs
mkdir -p "$LOG_DIR"

ORCHESTRATOR_LOG=experiments_logs/orchestrator_3ch_retrain.log

# Log orchestrator invocation
{
  echo "========================================"
  echo "3-Channel Clean Slate Orchestrator"
  echo "Started: $(date)"
  echo "Models: efficientnet_b2, resnet50, deit_base (3 models)"
  echo "Modalities: T1ce, T2, FLAIR (3 channels, standard RGB)"
  echo "Epochs: $EPOCHS (freeze: $FREEZE_EPOCHS)"
  echo "Learning Rate: $LR"
  echo "Weight Decay: $WEIGHT_DECAY"
  echo "========================================"
} | tee "$ORCHESTRATOR_LOG"

# Function to train a model
train_model() {
  local MODEL=$1
  local LOG_FILE="$LOG_DIR/${MODEL}.log"
  
  echo "========================================" | tee -a "$ORCHESTRATOR_LOG"
  echo "Training: $MODEL" | tee -a "$ORCHESTRATOR_LOG"
  echo "Started: $(date)" | tee -a "$ORCHESTRATOR_LOG"
  echo "Modalities: T1ce, T2, FLAIR (3 channels)" | tee -a "$ORCHESTRATOR_LOG"
  echo "========================================" | tee -a "$ORCHESTRATOR_LOG"
  
  # Run training
  uv run python scripts/train.py \
    --model "$MODEL" \
    --max-epochs "$EPOCHS" \
    --freeze-epochs "$FREEZE_EPOCHS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    2>&1 | tee "$LOG_FILE"
  
  local EXIT_CODE=${PIPESTATUS[0]}
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ $MODEL completed successfully at $(date)" | tee -a "$ORCHESTRATOR_LOG"
  else
    echo "❌ $MODEL FAILED with exit code $EXIT_CODE at $(date)" | tee -a "$ORCHESTRATOR_LOG"
    return $EXIT_CODE
  fi
}

# Train all models sequentially
train_model "efficientnet_b2"
train_model "resnet50"
train_model "deit_base"

# Final summary
{
  echo ""
  echo "========================================"
  echo "All models trained successfully!"
  echo "Completed: $(date)"
  echo "========================================"
} | tee -a "$ORCHESTRATOR_LOG"

echo ""
echo "Check results in: experiments_logs/"
echo "Training logs: experiments_logs/model_logs/"
echo "Orchestrator log: $ORCHESTRATOR_LOG"
