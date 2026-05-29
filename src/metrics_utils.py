#!/usr/bin/env python3
"""Shared evaluation helpers for the neural models (LSTM, mBERT, adapters).

Keeping these in one place guarantees that every model reports AUC-ROC, F1,
accuracy, precision and recall the same way, and that per-language results are
collected consistently.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Return the standard metric dict for one split.

    AUC-ROC is threshold-independent; the remaining metrics use ``threshold``.
    When a split contains a single class (can happen for tiny per-language
    slices) AUC-ROC is reported as NaN instead of crashing.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "auc_roc": round(float(auc), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }


def best_threshold(y_true, y_prob, metric="f1"):
    """Search probability thresholds in [0.05, 0.95] for the best F1.

    The training data is ~10% toxic, so the default 0.5 threshold is rarely
    optimal. We tune it on the multilingual validation set and reuse it on the
    test splits.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t), round(float(best_score), 4)


def evaluate_all_splits(name, predict_fn, splits, threshold=0.5):
    """Evaluate a model on every labelled split and return a list of result rows.

    Args:
        name: Model name to record in each row.
        predict_fn: Callable mapping a sequence of texts -> 1D array of toxic
            probabilities.
        splits: Dict produced by ``data_utils.load_splits``.
        threshold: Decision threshold applied to all splits.

    Returns:
        (rows, predictions) where ``rows`` is a list of metric dicts and
        ``predictions`` maps split name -> (y_true, y_prob, langs-or-None) so the
        caller can dump probabilities for ROC/PR plotting.
    """
    rows = []
    predictions = {}

    prob_en = predict_fn(splits["X_test_en"])
    rows.append({"model": name, "split": "English Test (20%)",
                 **compute_metrics(splits["y_test_en"], prob_en, threshold)})
    predictions["English Test (20%)"] = (
        np.asarray(splits["y_test_en"]), prob_en, None)

    prob_val = predict_fn(splits["X_val"])
    rows.append({"model": name, "split": "Multilingual Val",
                 **compute_metrics(splits["y_val"], prob_val, threshold)})
    predictions["Multilingual Val"] = (
        np.asarray(splits["y_val"]), prob_val, np.asarray(splits["val_langs"]))

    langs = splits["val_langs"]
    for lang in sorted(langs.unique()):
        mask = (langs == lang).to_numpy()
        rows.append({"model": name, "split": f"Val-{lang}",
                     **compute_metrics(np.asarray(splits["y_val"])[mask],
                                       prob_val[mask], threshold)})

    return rows, predictions


def print_rows(rows):
    """Pretty-print a list of metric rows."""
    for r in rows:
        print(f"  [{r['split']:<20}] "
              f"AUC={r['auc_roc']:.4f}  F1={r['f1']:.4f}  "
              f"Acc={r['accuracy']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")
