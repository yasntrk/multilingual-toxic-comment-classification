#!/usr/bin/env python3
"""Evaluation utilities for Multilingual Toxic Comment Classification."""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    return {k: round(v, 4) for k, v in metrics.items()}


def plot_confusion_matrix(y_true, y_pred, model_name, split_name, save_dir=FIGURES_DIR):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-toxic", "Toxic"])
    ax.set_yticklabels(["Non-toxic", "Toxic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {model_name}\n({split_name})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    fname = f"cm_{model_name.replace(' ', '_').replace('+', '')}_{split_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    return fname


def plot_roc_curves(results_dict, save_dir=FIGURES_DIR):
    """results_dict: {model_name: (y_true, y_prob)}"""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_prob) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Baseline Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=150)
    plt.close()


def summarize_results(results_csv_path):
    df = pd.read_csv(results_csv_path)
    print("\n=== Results Summary ===")
    for split in df["split"].unique():
        print(f"\n--- {split} ---")
        subset = df[df["split"] == split]
        print(subset[["model", "auc_roc", "f1", "accuracy", "precision", "recall"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "baseline_results.csv"))
    args = parser.parse_args()
    summarize_results(args.results)
