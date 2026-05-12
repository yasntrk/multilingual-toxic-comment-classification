#!/usr/bin/env python3
"""Data loading and preprocessing for Multilingual Toxic Comment Classification."""

import os
import re
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "jigsaw-toxic-comment-train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_df, val_df, test_df


def preprocess_and_save():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading raw data...")
    train_df, val_df, test_df = load_raw_data()

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("Cleaning text...")
    train_df["comment_text_clean"] = train_df["comment_text"].apply(clean_text)
    val_df["comment_text_clean"] = val_df["comment_text"].apply(clean_text)
    test_df["comment_text_clean"] = test_df["comment_text"].apply(clean_text)

    X_train, X_test_en, y_train, y_test_en = train_test_split(
        train_df, train_df["toxic"],
        test_size=0.2, random_state=42, stratify=train_df["toxic"]
    )

    print(f"Train split: {len(X_train)}, English holdout: {len(X_test_en)}")
    print(f"Toxic ratio (train): {y_train.mean():.4f}")

    X_train.to_csv(os.path.join(PROCESSED_DIR, "train_split.csv"), index=False)
    X_test_en.to_csv(os.path.join(PROCESSED_DIR, "test_en_split.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "validation_clean.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test_clean.csv"), index=False)

    print(f"Processed data saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Jigsaw dataset")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to raw data directory")
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    preprocess_and_save()
