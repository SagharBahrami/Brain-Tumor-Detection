#!/usr/bin/env bash
set -e  # Exit on any error

echo "=================================================="
echo "PRE-DEPLOYMENT TEST SUITE"
echo "Testing all components before GPU deployment"
echo "=================================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_passed=0
test_failed=0

run_test() {
    local test_name="$1"
    shift
    echo -e "${YELLOW}► Running: $test_name${NC}"
    if "$@"; then
        echo -e "${GREEN}✓ PASSED: $test_name${NC}"
        ((test_passed++))
    else
        echo -e "${RED}✗ FAILED: $test_name${NC}"
        ((test_failed++))
        return 1
    fi
    echo ""
}

# ====================================
# Test 1: Unit Tests
# ====================================
echo "============================================"
echo "TEST 1: Running Unit Tests"
echo "============================================"

run_test "Model Tests" uv run pytest tests/test_models.py -v || true
run_test "Data Loader Tests" uv run pytest tests/test_data_loader.py -v || true
run_test "Evaluation Tests" uv run pytest tests/test_evaluation.py -v || true

# ====================================
# Test 2: Model Creation (All 5 Models)
# ====================================
echo "============================================"
echo "TEST 2: Model Creation & Forward Pass"
echo "============================================"

run_test "SimpleCNN Creation" uv run python3 -c "
import torch
from src.models import create_model

model = create_model('simple_cnn', in_channels=3, pretrained=False)
x = torch.randn(2, 3, 224, 224)
y = model(x)
assert y.shape == (2, 1), f'Expected (2,1), got {y.shape}'
print(f'SimpleCNN output shape: {y.shape} ✓')
" || true

run_test "ResNet50 Creation" uv run python3 -c "
import torch
from src.models import create_model

model = create_model('resnet50', in_channels=3, pretrained=False)
x = torch.randn(2, 3, 224, 224)
y = model(x)
assert y.shape == (2, 1), f'Expected (2,1), got {y.shape}'
print(f'ResNet50 output shape: {y.shape} ✓')
" || true

run_test "EfficientNet-B2 Creation" uv run python3 -c "
import torch
from src.models import create_model

model = create_model('efficientnet_b2', in_channels=3, pretrained=False)
x = torch.randn(2, 3, 224, 224)
y = model(x)
assert y.shape == (2, 1), f'Expected (2,1), got {y.shape}'
print(f'EfficientNet-B2 output shape: {y.shape} ✓')
" || true

run_test "DeiT-Small Creation" uv run python3 -c "
import torch
from src.models import create_model

model = create_model('deit_small', in_channels=3, pretrained=False)
x = torch.randn(2, 3, 224, 224)
y = model(x)
assert y.shape == (2, 1), f'Expected (2,1), got {y.shape}'
print(f'DeiT-Small output shape: {y.shape} ✓')
" || true

run_test "DeiT-Base Creation" uv run python3 -c "
import torch
from src.models import create_model

model = create_model('deit_base', in_channels=3, pretrained=False)
x = torch.randn(2, 3, 224, 224)
y = model(x)
assert y.shape == (2, 1), f'Expected (2,1), got {y.shape}'
print(f'DeiT-Base output shape: {y.shape} ✓')
" || true

# ====================================
# Test 3: Data Loading
# ====================================
echo "============================================"
echo "TEST 3: Data Loading Pipeline"
echo "============================================"

if [ -d "data/BraTS2021_Training_Data" ] && [ -f "data/train_labels.csv" ]; then
    run_test "Data Loader Initialization" uv run python3 -c "
from pathlib import Path
from src.data_loader import DatasetConfig, build_dataloaders

config = DatasetConfig(
    task1_dir=Path('data/BraTS2021_Training_Data'),
    labels_csv=Path('data/train_labels.csv'),
    modalities=['t1ce', 't2', 'flair'],
    image_size=(224, 224)
)

try:
    train_loader, val_loader, test_loader = build_dataloaders(
        config=config,
        batch_size=8,
        num_workers=2,
        val_frac=0.15,
        test_frac=0.15,
        seed=42
    )
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')

    # Test batch retrieval
    batch = next(iter(train_loader))
    print(f'Batch image shape: {batch[\"image\"].shape}')
    print(f'Batch label shape: {batch[\"label\"].shape}')
    assert batch['image'].shape[0] == 8, 'Batch size mismatch'
    assert batch['image'].shape[1:] == (3, 224, 224), 'Image shape mismatch'
    print('Data loader working correctly ✓')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" || true
else
    echo -e "${YELLOW}⚠ Dataset not found. Skipping data loader tests.${NC}"
    echo "Expected: data/BraTS2021_Training_Data/ and data/train_labels.csv"
    echo ""
fi

# ====================================
# Test 4: Majority Voting Logic
# ====================================
echo "============================================"
echo "TEST 4: Patient-Level Majority Voting"
echo "============================================"

run_test "Majority Voting Function" uv run python3 -c "
import numpy as np
from src.evaluation import aggregate_patient_predictions

# Test case 1: Clear majority
patient_ids = ['P1', 'P1', 'P1', 'P2', 'P2', 'P2']
predictions = np.array([1, 1, 0, 0, 0, 1])
probabilities = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.6])
labels = np.array([1, 1, 1, 0, 0, 0])

p_labels, p_preds, p_probs, p_ids = aggregate_patient_predictions(
    patient_ids, predictions, probabilities, labels
)

assert len(p_labels) == 2, 'Should have 2 patients'
assert p_preds[0] == 1, 'P1 should predict 1 (2 out of 3)'
assert p_preds[1] == 0, 'P2 should predict 0 (2 out of 3)'
print('Clear majority case: ✓')

# Test case 2: Tie-breaking with probability
patient_ids = ['P3', 'P3', 'P3', 'P3']
predictions = np.array([1, 1, 0, 0])  # 50-50 tie
probabilities = np.array([0.7, 0.6, 0.3, 0.4])  # avg = 0.5
labels = np.array([1, 1, 1, 1])

p_labels, p_preds, p_probs, p_ids = aggregate_patient_predictions(
    patient_ids, predictions, probabilities, labels
)

assert len(p_labels) == 1, 'Should have 1 patient'
# Tie-break uses probability > 0.5, avg is 0.5, so should be 0
print(f'Tie-breaking case (50-50): prediction={p_preds[0]} ✓')

print('Majority voting logic working correctly ✓')
" || true

# ====================================
# Test 5: Fast Dev Run (CPU)
# ====================================
echo "============================================"
echo "TEST 5: Fast Dev Run on CPU (SimpleCNN)"
echo "============================================"

if [ -d "data/BraTS2021_Training_Data" ] && [ -f "data/train_labels.csv" ]; then
    run_test "Fast Dev Run Training" uv run python3 scripts/train_model.py \
        --task1-dir data/BraTS2021_Training_Data \
        --labels-csv data/train_labels.csv \
        --model simple_cnn \
        --batch-size 4 \
        --num-workers 2 \
        --fast-dev-run || true
else
    echo -e "${YELLOW}⚠ Dataset not found. Skipping fast dev run.${NC}"
    echo ""
fi

# ====================================
# Test 6: Training Module Shape Handling
# ====================================
echo "============================================"
echo "TEST 6: Training Module Shape Robustness"
echo "============================================"

run_test "Lightning Forward Pass" uv run python3 -c "
import torch
from src.models import create_model
from src.training import MGMTLightningModule, TrainingConfig

config = TrainingConfig(max_epochs=1, lr=0.001)
model = create_model('simple_cnn', pretrained=False)
lightning_module = MGMTLightningModule(model, config)

# Test with (N, 1) output
x = torch.randn(4, 3, 224, 224)
output = lightning_module(x)
assert output.shape == (4,), f'Expected (4,), got {output.shape}'
print(f'Shape handling works correctly: {output.shape} ✓')
" || true

# ====================================
# Summary
# ====================================
echo ""
echo "=================================================="
echo "TEST SUMMARY"
echo "=================================================="
echo -e "${GREEN}PASSED: $test_passed${NC}"
echo -e "${RED}FAILED: $test_failed${NC}"
echo ""

if [ $test_failed -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED! Ready for GPU deployment.${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED. Fix issues before GPU deployment.${NC}"
    exit 1
fi
