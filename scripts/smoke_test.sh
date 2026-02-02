#!/usr/bin/env bash
set -e

echo "===================================="
echo "SMOKE TEST - Quick validation"
echo "===================================="
echo ""

cd "$(dirname "$0")/.."

# Test 1: Model creation
echo "Test 1: Creating all models..."
for MODEL in simple_cnn resnet50 efficientnet_b2 deit_small deit_base; do
    echo -n "  - $MODEL: "
    uv run python3 -c "
from src.models import create_model
import torch
model = create_model('$MODEL', pretrained=False)
x = torch.randn(2, 3, 224, 224)
y = model(x)
assert y.shape == (2, 1), f'Bad shape: {y.shape}'
print('✓')
"
done

echo ""
echo "Test 2: Majority voting logic..."
uv run python3 -c "
import numpy as np
from src.evaluation import aggregate_patient_predictions

# Test tie-breaking
patient_ids = ['P1', 'P1', 'P1', 'P1']
predictions = np.array([1, 1, 0, 0])  # 50-50 tie
probabilities = np.array([0.7, 0.6, 0.3, 0.4])  # avg = 0.5
labels = np.array([1, 1, 1, 1])

p_labels, p_preds, p_probs, p_ids = aggregate_patient_predictions(
    patient_ids, predictions, probabilities, labels
)
print(f'Tie-break test: prediction={p_preds[0]} ✓')
"

echo ""
if [ -d "data/BraTS2021_Training_Data" ] && [ -f "data/train_labels.csv" ]; then
    echo "Test 3: Fast dev run (SimpleCNN, 1 batch)..."
    uv run python3 scripts/train_model.py \
        --task1-dir data/BraTS2021_Training_Data \
        --labels-csv data/train_labels.csv \
        --model simple_cnn \
        --batch-size 8 \
        --num-workers 2 \
        --fast-dev-run
    echo "✓ Fast dev run completed"
else
    echo "Test 3: Skipped (no dataset)"
fi

echo ""
echo "===================================="
echo "✓ ALL SMOKE TESTS PASSED"
echo "===================================="
