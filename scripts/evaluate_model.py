from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.serialization

# Ensure the project root (which contains the "src" package) is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DatasetConfig, build_dataloaders  # noqa: E402
from src.evaluation import evaluate_model  # noqa: E402
from src.models import create_model  # noqa: E402
from src.utils import ensure_dir, load_patient_manifest  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained tumor detection model")

    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .ckpt checkpoint file")
    parser.add_argument("--task1-dir", type=Path, required=True, help="Path to BraTS 2021 Task 1 directory")
    parser.add_argument("--labels-csv", type=Path, required=True, help="Path to RSNA-MICCAI labels CSV")
    parser.add_argument("--model", type=str, required=True, help="Model architecture identifier")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["t1ce", "t2", "flair"],
        help="MRI modalities included during training",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional patient manifest to filter dataset")

    parser.add_argument("--image-size", type=int, default=224, help="Image size used during training")

    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker processes")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Hold-out test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for dataset splits")
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Directory for evaluation outputs",
    )
    parser.add_argument("--roc-width", type=float, default=9.0, help="Width of the ROC figure")
    parser.add_argument("--roc-height", type=float, default=7.0, help="Height of the ROC figure")
    parser.add_argument("--cm-width", type=float, default=8.0, help="Width of the confusion matrix figure")
    parser.add_argument("--cm-height", type=float, default=7.0, help="Height of the confusion matrix figure")
    parser.add_argument("--title-font-size", type=float, default=14.0, help="Font size for plot titles")
    parser.add_argument("--label-font-size", type=float, default=12.0, help="Font size for axis labels")
    parser.add_argument("--tick-font-size", type=float, default=10.0, help="Font size for tick labels")
    parser.add_argument("--legend-font-size", type=float, default=10.0, help="Font size for legends")

    return parser.parse_args(argv)


def _summarize_dataset(dataset) -> Dict[str, object]:
    summary: Dict[str, object] = {"num_samples": len(dataset)}
    if hasattr(dataset, "records"):
        records = dataset.records  # type: ignore[attr-defined]
        patient_ids = {record.get("patient_id") for record in records if record.get("patient_id") is not None}
        summary["num_patients"] = len(patient_ids)
        label_counts: Dict[str, int] = {}
        for record in records:
            label = str(int(record.get("label", 0)))
            label_counts[label] = label_counts.get(label, 0) + 1
        summary["label_distribution"] = label_counts
        summary["tumor_slices"] = sum(1 for record in records if record.get("has_tumor"))
    return summary


def _build_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        task1_dir=args.task1_dir,
        labels_csv=args.labels_csv,
        modalities=args.modalities,
        image_size=(args.image_size, args.image_size),
    )


def _resolve_training_metadata(checkpoint_path: Path) -> Path | None:
    # Expect ".../run_dir/checkpoints/<file>.ckpt". Metadata lives beside checkpoints' parent.
    candidate = checkpoint_path.parent.parent / "training_metadata.json"
    return candidate if candidate.exists() else None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    manifest_list = None
    if args.manifest is not None:
        loaded_manifest = load_patient_manifest(args.manifest)
        manifest_list = list(loaded_manifest) if loaded_manifest is not None else None

    dataset_config = _build_dataset_config(args)

    train_loader, val_loader, test_loader = build_dataloaders(
        config=dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        manifest=manifest_list,
    )

    split_loader = val_loader if args.split == "val" else test_loader
    split_dataset = split_loader.dataset
    dataset_summary = _summarize_dataset(split_dataset)

    print("Evaluation dataset summary:")
    print(f"  Split: {args.split}")
    print(f"  Samples: {dataset_summary.get('num_samples', 'n/a')}")
    if "num_patients" in dataset_summary:
        print(f"  Patients: {dataset_summary['num_patients']}")
    if "label_distribution" in dataset_summary:
        print(f"  Label distribution: {dataset_summary['label_distribution']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_model(
        name=args.model,
        in_channels=len(args.modalities),
        pretrained=False,
        dropout=args.dropout,
    )
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    results = evaluate_model(model, split_loader, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = ensure_dir(args.output_dir / f"{args.model}_{args.split}_{timestamp}")

    metrics_path = eval_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(results.to_dict(), fh, indent=2)

    predictions_path = eval_dir / "predictions.csv"
    results.to_predictions_dataframe().to_csv(predictions_path, index=False)

    results.save_plots(
        eval_dir,
        args.model,
        roc_size=(args.roc_width, args.roc_height),
        cm_size=(args.cm_width, args.cm_height),
        title_fontsize=args.title_font_size,
        label_fontsize=args.label_font_size,
        tick_fontsize=args.tick_font_size,
        legend_fontsize=args.legend_font_size,
    )

    metadata: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "dataset": {
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in dataset_config.__dict__.items()
            },
            "summary": dataset_summary,
        },
        "dataloader": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
        },
        "artifacts": {
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "roc_curve": str(eval_dir / "roc_curve.png"),
            "confusion_matrix": str(eval_dir / "confusion_matrix.png"),
        },
    }
    if manifest_list is not None:
        metadata["manifest_count"] = len(manifest_list)

    training_metadata_path = _resolve_training_metadata(args.checkpoint)
    if training_metadata_path is not None:
        metadata["training_metadata"] = str(training_metadata_path)

    metadata_path = eval_dir / "evaluation_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("\nEvaluation metrics:")
    for metric, value in results.to_dict().items():
        if metric == "Confusion_Matrix":
            continue
        print(f"  {metric:20s}: {value}")
    print("  Confusion matrix:\n", results.confusion_matrix)
    print(f"\nArtifacts saved to: {eval_dir}")


if __name__ == "__main__":
    main()
