#!/usr/bin/env python3
"""Shared data loading, cleaning, and splitting utilities.

All training scripts import from this module so that the train/test split and
text-cleaning behaviour stay identical across baselines, the LSTM, and the
transformer models.

Two cleaning functions are provided on purpose:

- ``aggressive_clean`` strips every non-ASCII character. This is what the
  TF-IDF baselines use. It is intentionally bad for non-English text and is the
  reason the classical baselines collapse on the multilingual validation set.
- ``light_clean`` keeps Unicode letters (Turkish/Spanish/Italian accents, etc.)
  and only removes URLs, HTML and excess whitespace. Transformer and LSTM models
  should use this, since they rely on sub-word/character information.
"""

import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TRAIN_FILE = "jigsaw-toxic-comment-train.csv"
VALIDATION_FILE = "validation.csv"

# Same seed/ratio everywhere so every model is scored on the same English split.
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42


def aggressive_clean(text):
    """ASCII-only cleaning used by the TF-IDF baselines."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def light_clean(text):
    """Unicode-preserving cleaning used by the LSTM and transformer models."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_splits(clean_fn=light_clean, data_dir=DATA_DIR, max_train_samples=None):
    """Load the dataset and return the standard splits used in this project.

    Args:
        clean_fn: Function applied to every comment string.
        data_dir: Directory holding the raw Kaggle CSV files.
        max_train_samples: If set, randomly subsample the English training split
            to at most this many rows (stratified by label). Useful for running
            on a laptop in a reasonable amount of time.

    Returns:
        dict with keys:
            X_train, y_train          - English training comments / labels
            X_test_en, y_test_en      - 20% English holdout (in-language test)
            X_val, y_val, val_langs   - multilingual validation set + language tags
    """
    train_path = os.path.join(data_dir, TRAIN_FILE)
    val_path = os.path.join(data_dir, VALIDATION_FILE)

    print(f"Loading training data from {train_path} ...")
    train_df = pd.read_csv(train_path)
    print(f"Loading validation data from {val_path} ...")
    val_df = pd.read_csv(val_path)

    train_df["comment_text"] = train_df["comment_text"].apply(clean_fn)
    val_df["comment_text"] = val_df["comment_text"].apply(clean_fn)

    X_full = train_df["comment_text"].reset_index(drop=True)
    y_full = train_df["toxic"].reset_index(drop=True)

    X_train, X_test_en, y_train, y_test_en = train_test_split(
        X_full, y_full,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_full,
    )

    if max_train_samples is not None and max_train_samples < len(X_train):
        # Stratified subsample of the training split only; the test split is left
        # untouched so results stay comparable to full-data runs.
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=max_train_samples,
            random_state=RANDOM_SEED,
            stratify=y_train,
        )
        print(f"Subsampled training set to {len(X_train)} rows.")

    splits = {
        "X_train": X_train.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "X_test_en": X_test_en.reset_index(drop=True),
        "y_test_en": y_test_en.reset_index(drop=True),
        "X_val": val_df["comment_text"].reset_index(drop=True),
        "y_val": val_df["toxic"].reset_index(drop=True),
        "val_langs": val_df["lang"].reset_index(drop=True),
    }

    print(
        f"  Train: {len(splits['X_train'])} | "
        f"English test: {len(splits['X_test_en'])} | "
        f"Multilingual val: {len(splits['X_val'])}"
    )
    print(f"  Toxic ratio (train): {splits['y_train'].mean():.3f}")
    return splits
