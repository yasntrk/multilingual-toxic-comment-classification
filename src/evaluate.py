#!/usr/bin/env python3
"""Evaluation, comparison and visualization for all models.

This script reads the artifacts written by the training scripts into
``results/`` and produces the figures planned in the progress report:

  - Combined results table across every model.
  - Per-language AUC-ROC bar chart (cross-lingual transfer view).
  - ROC and precision-recall curves.
  - Per-language confusion matrices.
  - Trainable-parameter and inference-time comparison.
  - TF-IDF top-feature importance.
  - t-SNE of mBERT embeddings (optional; needs a saved model).

Each task is skipped gracefully if its input files are not present yet, so you
can run it after the baselines alone or after the full pipeline.

Examples:
    python src/evaluate.py                       # everything available
    python src/evaluate.py --task per_language
    python src/evaluate.py --task tsne --model-dir models/mBERT-adapter-64
"""

import os
import re
import glob
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Map a "split" label to a short language code for the per-language chart.
SPLIT_TO_LANG = {
    "English Test (20%)": "en",
    "Val-tr": "tr",
    "Val-es": "es",
    "Val-it": "it",
}
LANG_ORDER = ["en", "tr", "es", "it"]


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def load_all_results():
    """Concatenate every ``*_results.csv`` in results/ into one DataFrame."""
    frames = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_results.csv"))):
        df = pd.read_csv(path)
        if {"model", "split", "auc_roc"}.issubset(df.columns):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_pred_files(split_tag):
    """Return {model_name: DataFrame} for all ``preds_<model>_<split_tag>.csv``."""
    out = {}
    pattern = os.path.join(RESULTS_DIR, f"preds_*_{split_tag}.csv")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        model = re.sub(r"^preds_", "", fname)
        model = re.sub(rf"_{re.escape(split_tag)}\.csv$", "", model)
        out[model.replace("_", " ")] = pd.read_csv(path)
    return out


def model_threshold(model_name):
    """Look up a tuned threshold from a model's summary JSON, else 0.5."""
    safe = model_name.replace(" ", "*")
    for path in glob.glob(os.path.join(RESULTS_DIR, "*_results.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and data.get("model", "").replace("_", " ") == model_name:
            return float(data.get("threshold", 0.5))
    return 0.5


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def task_summary(df):
    if df.empty:
        print("No *_results.csv files found; run the training scripts first.")
        return
    combined = os.path.join(RESULTS_DIR, "all_results.csv")
    df.to_csv(combined, index=False)
    for split in df["split"].unique():
        print(f"\n--- {split} ---")
        sub = df[df["split"] == split]
        print(sub[["model", "auc_roc", "f1", "accuracy", "precision", "recall"]]
              .to_string(index=False))
    print(f"\nCombined table saved to {combined}")


def task_per_language(df):
    if df.empty:
        print("Per-language chart skipped: no results found.")
        return
    df = df.copy()
    df["lang"] = df["split"].map(SPLIT_TO_LANG)
    df = df.dropna(subset=["lang"])
    if df.empty:
        print("Per-language chart skipped: no per-language splits found.")
        return

    pivot = df.pivot_table(index="model", columns="lang", values="auc_roc")
    langs = [l for l in LANG_ORDER if l in pivot.columns]
    pivot = pivot[langs]

    models = list(pivot.index)
    x = np.arange(len(langs))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        ax.bar(x + i * width, pivot.loc[model].values, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([l.upper() for l in langs])
    ax.set_ylabel("AUC-ROC")
    ax.set_xlabel("Language")
    ax.set_title("Per-language AUC-ROC (cross-lingual transfer)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="random")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "per_language_auc.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def _curve_plot(preds, kind, split_label, out_name):
    if not preds:
        print(f"{kind} curves skipped: no prediction files for '{split_label}'.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, d in preds.items():
        y_true, y_prob = d["y_true"].to_numpy(), d["y_prob"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        if kind == "ROC":
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax.plot(fpr, tpr, label=f"{model} (AUC={roc_auc_score(y_true, y_prob):.3f})")
        else:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ax.plot(rec, prec,
                    label=f"{model} (AP={average_precision_score(y_true, y_prob):.3f})")
    if kind == "ROC":
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    ax.set_title(f"{kind} curves - {split_label}")
    ax.legend(fontsize=8, loc="lower left" if kind == "ROC" else "upper right")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, out_name)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def task_curves():
    for split_tag, split_label in [("English_Test_20", "English Test"),
                                   ("Multilingual_Val", "Multilingual Val")]:
        preds = load_pred_files(split_tag)
        _curve_plot(preds, "ROC", split_label, f"roc_{split_tag}.png")
        _curve_plot(preds, "PR", split_label, f"pr_{split_tag}.png")


def task_confusion():
    preds = load_pred_files("Multilingual_Val")
    if not preds:
        print("Confusion matrices skipped: no multilingual prediction files.")
        return
    for model, d in preds.items():
        if "lang" not in d.columns:
            continue
        thr = model_threshold(model)
        langs = sorted(d["lang"].unique())
        fig, axes = plt.subplots(1, len(langs), figsize=(4 * len(langs), 4))
        if len(langs) == 1:
            axes = [axes]
        for ax, lang in zip(axes, langs):
            sub = d[d["lang"] == lang]
            y_pred = (sub["y_prob"].to_numpy() >= thr).astype(int)
            cm = confusion_matrix(sub["y_true"], y_pred, labels=[0, 1])
            ax.imshow(cm, cmap="Blues")
            ax.set_title(f"{lang.upper()}")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Non-tox", "Tox"]); ax.set_yticklabels(["Non-tox", "Tox"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.suptitle(f"Confusion matrices per language: {model} (thr={thr:.2f})")
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f"confusion_{model.replace(' ', '_')}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")


def task_params():
    rows = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*_results.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "trainable_params" in data:
            rows.append(data)
    if not rows:
        print("Parameter/inference chart skipped: no transformer summaries found.")
        return

    rows.sort(key=lambda r: r["trainable_params"])
    names = [r["model"] for r in rows]
    trainable = [r["trainable_params"] / 1e6 for r in rows]
    ms = [r.get("ms_per_sample", 0) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.barh(names, trainable, color="#3498db")
    ax1.set_xlabel("Trainable parameters (millions)")
    ax1.set_title("Trainable parameters")
    ax2.barh(names, ms, color="#e67e22")
    ax2.set_xlabel("Inference time (ms / sample)")
    ax2.set_title("Inference latency")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "param_inference_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def task_features():
    path = os.path.join(RESULTS_DIR, "tfidf_feature_importance.json")
    if not os.path.exists(path):
        print("Feature importance chart skipped: run train_baselines.py first.")
        return
    with open(path) as f:
        data = json.load(f)
    top_n = 20
    toxic = data["most_toxic"][:top_n][::-1]
    clean = data["least_toxic"][:top_n][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
    ax1.barh([d["feature"] for d in toxic], [d["weight"] for d in toxic], color="#e74c3c")
    ax1.set_title("Top features -> Toxic")
    ax1.set_xlabel("LR coefficient")
    ax2.barh([d["feature"] for d in clean], [d["weight"] for d in clean], color="#2ecc71")
    ax2.set_title("Top features -> Non-toxic")
    ax2.set_xlabel("LR coefficient")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "tfidf_feature_importance.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def task_tsne(model_dir, sample_size=600):
    """t-SNE of mBERT [CLS] embeddings on the multilingual validation set."""
    if not model_dir or not os.path.isdir(model_dir):
        print("t-SNE skipped: pass --model-dir pointing to a saved mBERT model.")
        return
    import torch
    from sklearn.manifold import TSNE
    from transformers import AutoTokenizer, AutoModel
    import data_utils

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()

    splits = data_utils.load_splits(clean_fn=data_utils.light_clean)
    idx = np.random.RandomState(42).choice(
        len(splits["X_val"]), size=min(sample_size, len(splits["X_val"])), replace=False)
    texts = splits["X_val"].iloc[idx].tolist()
    langs = splits["val_langs"].iloc[idx].to_numpy()
    labels = np.asarray(splits["y_val"])[idx]

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(device)
            cls = model(**enc).last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls.cpu().numpy())
    embeddings = np.concatenate(embeddings)

    coords = TSNE(n_components=2, random_state=42,
                  perplexity=30, init="pca").fit_transform(embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for lang in np.unique(langs):
        m = langs == lang
        ax1.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.6, label=lang)
    ax1.set_title("mBERT [CLS] embeddings by language")
    ax1.legend()
    for label, color, name in [(0, "#2ecc71", "Non-toxic"), (1, "#e74c3c", "Toxic")]:
        m = labels == label
        ax2.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.6, c=color, label=name)
    ax2.set_title("mBERT [CLS] embeddings by toxicity")
    ax2.legend()
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "tsne_mbert.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize results")
    parser.add_argument("--task", default="all",
                        choices=["all", "summary", "per_language", "curves",
                                 "confusion", "params", "features", "tsne"])
    parser.add_argument("--model-dir", default=None,
                        help="Saved mBERT directory, required for the t-SNE task.")
    args = parser.parse_args()

    df = load_all_results()

    if args.task in ("all", "summary"):
        task_summary(df)
    if args.task in ("all", "per_language"):
        task_per_language(df)
    if args.task in ("all", "curves"):
        task_curves()
    if args.task in ("all", "confusion"):
        task_confusion()
    if args.task in ("all", "params"):
        task_params()
    if args.task in ("all", "features"):
        task_features()
    if args.task == "tsne":  # opt-in only: needs a saved model + transformers
        task_tsne(args.model_dir)


if __name__ == "__main__":
    main()
