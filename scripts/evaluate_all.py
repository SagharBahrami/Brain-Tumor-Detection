#!/usr/bin/env python3
"""Evaluate trained three-channel tumor-detection models on the held-out test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import create_model, find_patient_dirs  # noqa: E402
from src.data_loader import (  # noqa: E402
    DatasetConfig,
    create_patient_centric_data_loaders,
)


DEFAULT_MODELS = ("efficientnet_b2", "resnet50", "deit_base")
MODALITIES = ("t1ce", "t2", "flair")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained tumor-detection models on the held-out test set."
    )
    parser.add_argument(
        "--task1-dir",
        type=Path,
        default=Path("data/BraTS2021_Training_Data"),
        help="Directory containing BraTS 2021 patient folders.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments_logs"),
        help="Root directory containing timestamped training runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (default: RESULTS_DIR/evaluation_results.csv).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["efficientnet_b2", "resnet50", "deit_small", "deit_base"],
        default=list(DEFAULT_MODELS),
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the original patient-level split.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Evaluation device.",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def find_best_checkpoint(results_dir: Path, model_name: str) -> Path:
    """Return the newest best checkpoint from timestamped training runs."""
    checkpoint_dir_pattern = f"tumor_detection/{model_name}_*/checkpoints"
    matches = list(results_dir.glob(f"{checkpoint_dir_pattern}/*_best-*.ckpt"))
    if not matches:
        matches = list(results_dir.glob(f"{checkpoint_dir_pattern}/*best*.ckpt"))
    if not matches:
        raise FileNotFoundError(
            f"No best checkpoint found for {model_name} under {results_dir}."
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_trained_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Create the training architecture and restore its weights from Lightning."""
    model = create_model(model_name)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)

    if any(key.startswith("model.") for key in state_dict):
        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float | int]:
    """Compute flat, slice-level metrics on the held-out patient split."""
    model.to(device)
    model.eval()

    all_predictions: list[int] = []
    all_labels: list[int] = []
    all_probabilities: list[float] = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].squeeze(0).to(device)
            labels = batch["labels"].squeeze(0).to(device).reshape(-1)

            logits = model(images).reshape(-1)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).long()

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    if not all_labels:
        raise ValueError("No test samples were collected.")

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_pred = np.asarray(all_predictions, dtype=np.int64)
    y_score = np.asarray(all_probabilities, dtype=np.float64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auc = float("nan")
    if np.unique(y_true).size == 2:
        auc = float(roc_auc_score(y_true, y_score))

    return {
        "samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    output_path = args.output or args.results_dir / "evaluation_results.csv"

    print("\n" + "=" * 70)
    print("EVALUATION: Held-out test metrics for trained models")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Modalities: {', '.join(MODALITIES)}")
    print(f"Patient split seed: {args.seed}\n")

    if not args.task1_dir.exists():
        raise FileNotFoundError(f"BraTS data directory not found: {args.task1_dir}")

    patient_dirs = find_patient_dirs(args.task1_dir)
    if not patient_dirs:
        raise ValueError(f"No BraTS patient directories found in {args.task1_dir}")

    config = DatasetConfig(modalities=MODALITIES, image_size=(224, 224))
    _, _, test_loader = create_patient_centric_data_loaders(
        patient_dirs,
        config,
        val_frac=0.15,
        test_frac=0.15,
        seed=args.seed,
    )

    all_results: list[dict[str, float | int | str]] = []
    for model_name in args.models:
        print(f"Evaluating: {model_name}")
        try:
            checkpoint_path = find_best_checkpoint(args.results_dir, model_name)
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}")
            continue

        print(f"  Checkpoint: {checkpoint_path}")
        model = load_trained_model(model_name, checkpoint_path, device)
        metrics = evaluate_model(model, test_loader, device)
        all_results.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint_path),
                **metrics,
            }
        )
        print(
            f"  Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}\n"
        )

    if not all_results:
        raise FileNotFoundError(
            f"No evaluable checkpoints found under {args.results_dir}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(all_results)
    results_frame.to_csv(output_path, index=False)

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_frame.drop(columns=["checkpoint"]).to_string(index=False))
    print(f"\nSaved to: {output_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
