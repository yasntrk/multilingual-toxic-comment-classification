#!/usr/bin/env python3
"""Optional: write cleaned train/test splits to disk as CSV files.

This is a convenience script. The training scripts call ``data_utils.load_splits``
directly and do not require preprocessed files on disk, but having them can be
useful for quick inspection in a spreadsheet or for sharing intermediate data.

Usage:
    python src/preprocess.py
    python src/preprocess.py --data-dir /path/to/raw/csvs
"""

import os
import argparse

import data_utils

PROCESSED_DIR = os.path.join(data_utils.DATA_DIR, "processed")


def preprocess_and_save(data_dir=None):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    splits = data_utils.load_splits(
        clean_fn=data_utils.light_clean,
        data_dir=data_dir or data_utils.DATA_DIR,
    )

    splits["X_train"].to_csv(os.path.join(PROCESSED_DIR, "train_split.csv"), index=False)
    splits["X_test_en"].to_csv(os.path.join(PROCESSED_DIR, "test_en_split.csv"), index=False)

    print(f"Processed splits saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Jigsaw dataset")
    parser.add_argument("--data-dir", default=None,
                        help="Override the raw data directory (default: data/)")
    args = parser.parse_args()
    preprocess_and_save(args.data_dir)
