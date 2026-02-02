#!/usr/bin/env python3
"""Generate Grad-CAM visualizations for a batch of patient samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for multiple patients")
    parser.add_argument("--predictions-csv", type=Path, required=True, help="Path to predictions.csv from evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")
    parser.add_argument("--task1-dir", type=Path, required=True, help="Path to BraTS2021 Task 1 directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results/gradcam_batch"), help="Output directory")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of samples to visualize")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (must match training)")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["t1ce", "t2", "flair"],
        help="MRI modalities used during training",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Image size (must match training)")
    return parser.parse_args(argv)


def select_diverse_samples(predictions_df: pd.DataFrame, num_samples: int = 12) -> pd.DataFrame:
    """Select diverse samples: TP, TN, FP, FN with varying confidence."""

    # Classify predictions
    predictions_df["correct"] = predictions_df["label"] == predictions_df["prediction"]
    predictions_df["category"] = "other"

    # True Positives
    tp_mask = (predictions_df["label"] == 1) & (predictions_df["prediction"] == 1)
    predictions_df.loc[tp_mask, "category"] = "TP"

    # True Negatives
    tn_mask = (predictions_df["label"] == 0) & (predictions_df["prediction"] == 0)
    predictions_df.loc[tn_mask, "category"] = "TN"

    # False Positives
    fp_mask = (predictions_df["label"] == 0) & (predictions_df["prediction"] == 1)
    predictions_df.loc[fp_mask, "category"] = "FP"

    # False Negatives
    fn_mask = (predictions_df["label"] == 1) & (predictions_df["prediction"] == 0)
    predictions_df.loc[fn_mask, "category"] = "FN"

    # Select samples per category
    samples_per_category = max(2, num_samples // 4)
    selected = []

    for category in ["TP", "TN", "FP", "FN"]:
        category_df = predictions_df[predictions_df["category"] == category]
        if len(category_df) > 0:
            # Sort by confidence (distance from 0.5) and select top samples
            category_df = category_df.copy()
            category_df["confidence"] = abs(category_df["probability"] - 0.5)
            category_df = category_df.sort_values("confidence", ascending=False)
            selected.append(category_df.head(samples_per_category))

    if selected:
        result = pd.concat(selected, ignore_index=True)
        return result.head(num_samples)
    return pd.DataFrame()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # Load predictions
    predictions_df = pd.read_csv(args.predictions_csv)
    print(f"Loaded {len(predictions_df)} predictions from {args.predictions_csv}")

    # Select diverse samples
    selected_df = select_diverse_samples(predictions_df, args.num_samples)
    print(f"\nSelected {len(selected_df)} samples for Grad-CAM visualization:")
    print(selected_df[["patient_id", "slice_idx", "label", "prediction", "probability", "category"]].to_string(index=False))

    # Save selection metadata
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / f"{args.model}_selected_samples.csv"
    selected_df.to_csv(selection_path, index=False)
    print(f"\nSaved selection to: {selection_path}")

    # Generate Grad-CAM for each sample
    import os
    from subprocess import run

    success_count = 0
    failed = []
    generated_files = []

    for idx, row in selected_df.iterrows():
        patient_id = str(row["patient_id"]).zfill(5)
        patient_dir = args.task1_dir / f"BraTS2021_{patient_id}"
        slice_idx = int(row["slice_idx"])
        true_label = int(row["label"])
        category = row["category"]

        if not patient_dir.exists():
            print(f"Warning: Patient directory not found: {patient_dir}")
            failed.append(patient_id)
            continue

        print(f"\n[{idx + 1}/{len(selected_df)}] Generating Grad-CAM for patient {patient_id}, slice {slice_idx} ({category})...")

        cmd = [
            "uv", "run", "python", "scripts/generate_explanations.py",
            "--checkpoint", str(args.checkpoint),
            "--model", args.model,
            "--patient-dir", str(patient_dir),
            "--slice-idx", str(slice_idx),
            "--true-label", str(true_label),
            "--output-dir", str(args.output_dir),
            "--dropout", str(args.dropout),
            "--modalities", *args.modalities,
            "--image-size", str(args.image_size),
        ]

        # Inherit current environment and add PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = "."

        result = run(cmd, capture_output=True, text=True, env=env)

        if result.returncode == 0:
            success_count += 1
            viz_path = args.output_dir / args.model / f"patient_{patient_id}" / f"slice_{slice_idx}_gradcam.png"
            generated_files.append({
                "patient_id": patient_id,
                "slice_idx": slice_idx,
                "category": category,
                "label": true_label,
                "prediction": int(row["prediction"]),
                "probability": float(row["probability"]),
                "file": str(viz_path),
            })
            print(f"  ✓ Success: {viz_path}")
        else:
            print(f"  ✗ Failed")
            # Log full stderr to file
            error_log = args.output_dir / f"error_{patient_id}_slice{slice_idx}.log"
            error_log.write_text(result.stderr)
            print(f"  Error logged to: {error_log}")
            failed.append(patient_id)

    # Save generated files manifest
    if generated_files:
        manifest_path = args.output_dir / f"{args.model}_gradcam_manifest.json"
        with manifest_path.open("w") as f:
            json.dump({"files": generated_files, "model": args.model, "total": len(generated_files)}, f, indent=2)
        print(f"\nGenerated files manifest saved to: {manifest_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Grad-CAM generation complete!")
    print(f"  Success: {success_count}/{len(selected_df)}")
    if failed:
        print(f"  Failed: {len(failed)} - {failed}")
    print(f"  Output directory: {args.output_dir}")
    print(f"\nGenerated visualizations by category:")
    for item in generated_files:
        print(f"  {item['category']:3s} | Patient {item['patient_id']} slice {item['slice_idx']:3d} | {item['file']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
