from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_confusion_matrix,
    binary_f1_score,
    binary_precision,
    binary_recall,
    binary_roc,
)

from .utils import move_to_device


@dataclass
class EvaluationResult:
    """Container for evaluation metrics and per-slice predictions."""

    # Slice-level metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    confusion_matrix: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    patient_ids: List[str]
    slice_indices: np.ndarray
    balanced_accuracy: float = 0.0  # Slice-level balanced accuracy (handles class imbalance)

    def to_dict(self) -> Dict[str, Any]:
        """Return slice-level F1 and metrics for thesis."""
        result = {
            "slice_level": {
                "F1-Score": round(float(self.f1), 4),
                "Accuracy": round(float(self.accuracy), 4),
                "Balanced-Accuracy": round(float(self.balanced_accuracy), 4),
                "Precision": round(float(self.precision), 4),
                "Recall": round(float(self.recall), 4),
                "Specificity": round(float(self.specificity), 4),
            }
        }
        return result

    def to_predictions_dataframe(self) -> pd.DataFrame:
        """Return predictions (per slice) for downstream analysis."""
        return pd.DataFrame(
            {
                "patient_id": self.patient_ids,
                "slice_idx": self.slice_indices.astype(int),
                "label": self.labels.astype(int),
                "prediction": self.predictions.astype(int),
                "probability": self.probabilities.astype(float),
            }
        )

    def save_plots(
        self,
        output_dir: Path,
        model_name: str,
        *,
        roc_size: tuple[float, float] = (9.0, 7.0),
        cm_size: tuple[float, float] = (8.0, 7.0),
        title_fontsize: float = 14.0,
        label_fontsize: float = 12.0,
        tick_fontsize: float = 10.0,
        legend_fontsize: float = 10.0,
        dpi: int = 300,
    ) -> None:
        """Persist ROC curve and confusion matrix figures to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        roc_path = output_dir / "roc_curve.png"
        plt.figure(figsize=roc_size)
        plt.plot(self.fpr, self.tpr, label=f"F1 = {self.f1:.3f}", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        ax = plt.gca()
        ax.set_xlabel("False Positive Rate", fontsize=label_fontsize)
        ax.set_ylabel("True Positive Rate", fontsize=label_fontsize)
        ax.set_title(f"ROC Curve - {model_name}", fontsize=title_fontsize)
        legend = ax.legend(loc="lower right", fontsize=legend_fontsize)
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(legend_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(roc_path, dpi=dpi)
        plt.close()

        cm_path = output_dir / "confusion_matrix.png"
        plt.figure(figsize=cm_size)
        ax = sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Unmethylated", "Methylated"],
            yticklabels=["Unmethylated", "Methylated"],
        )
        ax.set_xlabel("Predicted", fontsize=label_fontsize)
        ax.set_ylabel("True", fontsize=label_fontsize)
        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=title_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=dpi)
        plt.close()


def _extract_scalar(value: float) -> float:
    """Return a plain Python float for serialisation."""
    return float(value)


def _normalise_patient_batch(patient_batch: Any, batch_size: int) -> List[str]:
    if patient_batch is None:
        return ["unknown"] * batch_size
    if isinstance(patient_batch, list):
        return [str(item) for item in patient_batch]
    if isinstance(patient_batch, tuple):
        return [str(item) for item in patient_batch]
    if isinstance(patient_batch, torch.Tensor):
        return [str(item.item()) for item in patient_batch]
    return [str(patient_batch)] * batch_size


def _normalise_slice_indices(slice_batch: Any, batch_size: int) -> List[int]:
    if slice_batch is None:
        return [-1] * batch_size
    if isinstance(slice_batch, torch.Tensor):
        return slice_batch.cpu().long().tolist()
    if isinstance(slice_batch, list):
        return [int(item) for item in slice_batch]
    if isinstance(slice_batch, tuple):
        return [int(item) for item in slice_batch]
    return [int(slice_batch)] * batch_size


def aggregate_patient_predictions(
    patient_ids: List[str],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Aggregate slice-level predictions to patient-level using majority voting.

    For each patient, if more than 50% of their slices predict class 1,
    the patient prediction is 1, otherwise 0.

    Tie-breaking: In case of exact 50%-50% tie, use average probability > 0.5.

    Returns:
        patient_labels: True labels for each patient (unique)
        patient_predictions: Majority vote predictions per patient
        patient_probabilities: Average probability per patient
        unique_patient_ids: List of unique patient IDs
    """
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'prediction': predictions,
        'probability': probabilities,
        'label': labels,
    })

    # Group by patient and aggregate
    def _patient_level_agg(group: pd.DataFrame) -> pd.Series:
        """Aggregate slice-level predictions to the patient level."""
        vote_mean = group["prediction"].mean()
        prob_mean = group["probability"].mean()

        # Majority vote with probability-based tie-breaking
        if vote_mean != 0.5:
            prediction = int(vote_mean > 0.5)
        else:
            prediction = int(prob_mean > 0.5)

        return pd.Series({
            "prediction": prediction,
            "probability": prob_mean,
            "label": group["label"].iloc[0],  # All labels for a patient are the same
        })

    patient_df = df.groupby("patient_id").apply(_patient_level_agg).reset_index()

    return (
        patient_df['label'].values,
        patient_df['prediction'].values,
        patient_df['probability'].values,
        patient_df['patient_id'].tolist(),
    )


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> EvaluationResult:
    """Run inference on dataloader and collect slice-level evaluation metrics.
    
    NOTE: No patient-level aggregation. All BraTS patients have tumors,
    so patient-level labels are meaningless. We focus on slice-level metrics.
    """
    model.eval()

    if len(dataloader.dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")

    all_probs: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_patient_ids: List[str] = []
    all_slice_indices: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)

            images = batch["image"]
            labels = batch["label"].float()
            batch_size = labels.size(0)

            logits = model(images)
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            logits = logits.view(-1)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())
            all_patient_ids.extend(_normalise_patient_batch(batch.get("patient_id"), batch_size))
            all_slice_indices.extend(_normalise_slice_indices(batch.get("slice_idx"), batch_size))

    y_true_tensor = torch.tensor(all_labels, dtype=torch.int64)
    y_pred_tensor = torch.tensor(all_preds, dtype=torch.int64)
    y_prob_tensor = torch.tensor(all_probs, dtype=torch.float32)
    slice_indices = np.asarray(all_slice_indices, dtype=np.int64)

    if y_true_tensor.numel() == 0:
        raise ValueError("No samples collected during evaluation.")

    # Slice-level metrics only
    cm_tensor = binary_confusion_matrix(y_pred_tensor, y_true_tensor)
    cm = cm_tensor.cpu().numpy().astype(int)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Balanced accuracy: (recall + specificity) / 2
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2.0

    if torch.unique(y_true_tensor).numel() > 1:
        fpr_tensor, tpr_tensor, thresholds_tensor = binary_roc(y_prob_tensor, y_true_tensor)
        fpr = fpr_tensor.cpu().numpy()
        tpr = tpr_tensor.cpu().numpy()
        thresholds = thresholds_tensor.cpu().numpy()
    else:
        fpr = np.array([0.0, 1.0], dtype=np.float32)
        tpr = np.array([0.0, 1.0], dtype=np.float32)
        thresholds = np.array([1.0, 0.0], dtype=np.float32)

    return EvaluationResult(
        # Slice-level metrics only
        accuracy=_extract_scalar(binary_accuracy(y_pred_tensor, y_true_tensor).item()),
        precision=_extract_scalar(binary_precision(y_pred_tensor, y_true_tensor, zero_division=0.0).item()),
        recall=_extract_scalar(binary_recall(y_pred_tensor, y_true_tensor, zero_division=0.0).item()),
        f1=_extract_scalar(binary_f1_score(y_pred_tensor, y_true_tensor, zero_division=0.0).item()),
        specificity=_extract_scalar(specificity),
        balanced_accuracy=balanced_acc,
        confusion_matrix=cm,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        labels=y_true_tensor.cpu().numpy(),
        predictions=y_pred_tensor.cpu().numpy(),
        probabilities=y_prob_tensor.cpu().numpy(),
        patient_ids=all_patient_ids,
        slice_indices=slice_indices,
    )

