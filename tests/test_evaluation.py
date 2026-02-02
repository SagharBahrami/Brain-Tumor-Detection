from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation import EvaluationResult, evaluate_model


class _DummyDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [
            {"logit": -2.0, "label": 0, "patient_id": "00000", "slice_idx": 0},
            {"logit": -1.0, "label": 0, "patient_id": "00001", "slice_idx": 1},
            {"logit": 1.5, "label": 1, "patient_id": "00002", "slice_idx": 2},
            {"logit": 2.2, "label": 1, "patient_id": "00003", "slice_idx": 3},
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = torch.zeros(3, 224, 224, dtype=torch.float32)
        image[0, 0, 0] = sample["logit"]
        return {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "patient_id": sample["patient_id"],
            "slice_idx": torch.tensor(sample["slice_idx"], dtype=torch.int64),
        }


class _SimpleModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = x[:, 0, 0, 0]
        return logits.unsqueeze(-1)


def test_evaluation_result_to_dict() -> None:
    result = EvaluationResult(
        auc=0.85,
        accuracy=0.8,
        precision=0.75,
        recall=0.82,
        f1=0.78,
        specificity=0.77,
        confusion_matrix=np.array([[10, 2], [3, 15]]),
        fpr=np.array([0.0, 0.5, 1.0]),
        tpr=np.array([0.0, 0.8, 1.0]),
        thresholds=np.array([1.0, 0.5, 0.1]),
        labels=np.array([0, 1]),
        predictions=np.array([0, 1]),
        probabilities=np.array([0.1, 0.9]),
        patient_ids=["00000", "00001"],
        slice_indices=np.array([0, 1]),
    )

    metrics = result.to_dict()
    # Updated for new nested format (patient_level and slice_level)
    assert metrics["slice_level"]["F1-Score"] == pytest.approx(0.78)
    assert metrics["slice_level"]["Accuracy"] == pytest.approx(0.8)
    # Patient-level metrics exist but may be 0.0 if not computed
    assert "patient_level" in metrics
    assert "F1-Score" in metrics["patient_level"]
    assert "Accuracy" in metrics["patient_level"]

    predictions_df = result.to_predictions_dataframe()
    assert set(predictions_df.columns) == {"patient_id", "slice_idx", "label", "prediction", "probability"}
    assert len(predictions_df) == 2


def test_evaluate_model_runs_and_exports(tmp_path) -> None:
    dataset = _DummyDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = _SimpleModel()
    device = torch.device("cpu")

    result = evaluate_model(model, dataloader, device)

    assert result.accuracy == pytest.approx(1.0)
    assert result.confusion_matrix.shape == (2, 2)
    assert result.fpr.ndim == 1

    output_dir = tmp_path / "plots"
    result.save_plots(output_dir, "dummy")
    assert (output_dir / "roc_curve.png").exists()
    assert (output_dir / "confusion_matrix.png").exists()

    predictions_df = result.to_predictions_dataframe()
    assert len(predictions_df) == len(dataset)
    assert set(predictions_df["patient_id"]) == {"00000", "00001", "00002", "00003"}