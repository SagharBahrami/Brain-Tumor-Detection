#!/usr/bin/env python3
"""Quick diagnostic to test data loader throughput and num_workers stability.
Usage: python scripts/test_data_loading.py --num-workers 4 --batch-size 16 --batches 10
"""
import argparse
import time
from pathlib import Path

from src.data_loader import DatasetConfig, build_dataloaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task1-dir", type=Path, default=Path("data/BraTS2021_Training_Data"))
    p.add_argument("--labels-csv", type=Path, default=Path("data/train_labels.csv"))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batches", type=int, default=20)
    args = p.parse_args()

    config = DatasetConfig(
        task1_dir=args.task1_dir,
        labels_csv=args.labels_csv,
        modalities=["t1ce", "t2", "flair"],
        image_size=(224, 224),
    )

    print(f"Building dataloaders (workers={args.num_workers}, batch={args.batch_size})")
    train_loader, val_loader, test_loader = build_dataloaders(
        config, batch_size=args.batch_size, num_workers=args.num_workers
    )

    it = iter(train_loader)
    t0 = time.time()
    for i in range(args.batches):
        try:
            b = next(it)
        except StopIteration:
            print("Iterator exhausted")
            break
        t1 = time.time()
        print(f"Batch {i+1}/{args.batches}: images {b['image'].shape}, labels {b['label'].shape}, time {(t1-t0):.3f}s")
        t0 = t1

    print("Done")


if __name__ == '__main__':
    main()
