#!/usr/bin/env python3
"""Exploratory data analysis for the Jigsaw Multilingual Toxic Comment dataset.

Prints summary statistics and saves four figures to ``figures/``:

  1. Label distribution (training) + toxic ratio per language (validation)
  2. Comment length distribution
  3. Comment length split by toxicity label
  4. Multi-label correlation heatmap

Usage:
    python src/eda.py
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import data_utils

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    print("Loading data ...")
    train_df = pd.read_csv(
        os.path.join(data_utils.DATA_DIR, data_utils.TRAIN_FILE))
    val_df = pd.read_csv(
        os.path.join(data_utils.DATA_DIR, data_utils.VALIDATION_FILE))

    # --- Summary statistics ---
    print(f"\n=== Training Data ===")
    print(f"Shape: {train_df.shape}")
    print(f"\nLabel distribution:")
    print(train_df["toxic"].value_counts())
    print(f"Toxic ratio: {train_df['toxic'].mean():.4f}")

    train_df["text_length"] = train_df["comment_text"].str.len()
    print(f"\nText length stats:")
    print(train_df["text_length"].describe())

    print(f"\n=== Validation Data (Multilingual) ===")
    print(f"Shape: {val_df.shape}")
    print(f"\nLanguage distribution:")
    print(val_df["lang"].value_counts())
    print(f"\nToxic ratio per language:")
    print(val_df.groupby("lang")["toxic"].mean())

    # --- Plots ---
    # 1. Label distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    train_df["toxic"].value_counts().plot(
        kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Training Set: Label Distribution")
    axes[0].set_xticklabels(["Non-toxic", "Toxic"], rotation=0)
    axes[0].set_ylabel("Count")

    val_df.groupby("lang")["toxic"].mean().plot(
        kind="bar", ax=axes[1], color="#3498db")
    axes[1].set_title("Validation Set: Toxic Ratio by Language")
    axes[1].set_ylabel("Toxic Ratio")
    axes[1].set_xlabel("Language")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "label_distribution.png"), dpi=150)
    plt.close()
    print(f"\nSaved: figures/label_distribution.png")

    # 2. Text length distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    train_df["text_length"].clip(upper=2000).hist(
        bins=100, ax=ax, color="#3498db", alpha=0.7)
    ax.set_title("Training Set: Comment Length Distribution (clipped at 2000 chars)")
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "text_length_distribution.png"), dpi=150)
    plt.close()
    print("Saved: figures/text_length_distribution.png")

    # 3. Text length by toxicity
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [(0, "#2ecc71", "Non-toxic"),
                                (1, "#e74c3c", "Toxic")]:
        subset = train_df[train_df["toxic"] == label]
        subset["text_length"].clip(upper=2000).hist(
            bins=80, ax=ax, color=color, alpha=0.5, label=name)
    ax.set_title("Comment Length by Toxicity Label")
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "text_length_by_toxicity.png"), dpi=150)
    plt.close()
    print("Saved: figures/text_length_by_toxicity.png")

    # 4. Multi-label correlation heatmap
    label_cols = [
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
    ]
    corr = train_df[label_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
                ax=ax, vmin=-1, vmax=1)
    ax.set_title("Label Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "label_correlation.png"), dpi=150)
    plt.close()
    print("Saved: figures/label_correlation.png")

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
