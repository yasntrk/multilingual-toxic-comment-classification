#!/usr/bin/env python3
"""TF-IDF Baseline Models for Multilingual Toxic Comment Classification."""

import os
import re
import json
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, classification_report, confusion_matrix
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data():
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "jigsaw-toxic-comment-train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))

    train_df["comment_text"] = train_df["comment_text"].apply(clean_text)
    val_df["comment_text"] = val_df["comment_text"].apply(clean_text)

    X_train_full = train_df["comment_text"]
    y_train_full = train_df["toxic"]

    X_train, X_test_en, y_train, y_test_en = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    X_val_multi = val_df["comment_text"]
    y_val_multi = val_df["toxic"]
    val_langs = val_df["lang"]

    print(f"  Train: {len(X_train)}, English test: {len(X_test_en)}, Multilingual val: {len(X_val_multi)}")
    print(f"  Toxic ratio (train): {y_train.mean():.3f}")
    return X_train, y_train, X_test_en, y_test_en, X_val_multi, y_val_multi, val_langs


def evaluate_model(name, model, X_test, y_test, tfidf, split_name="English Test"):
    X_tfidf = tfidf.transform(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_tfidf)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_tfidf)
    else:
        y_prob = model.predict(X_tfidf)

    y_pred = (y_prob >= 0.5).astype(int) if hasattr(model, "predict_proba") else model.predict(X_tfidf)

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"\n  [{name}] on {split_name}:")
    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")

    return {
        "model": name,
        "split": split_name,
        "auc_roc": round(auc, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


def main():
    X_train, y_train, X_test_en, y_test_en, X_val_multi, y_val_multi, val_langs = load_data()

    print("\nFitting TF-IDF vectorizer (50K features, unigrams+bigrams)...")
    t0 = time.time()
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    print(f"  TF-IDF shape: {X_train_tfidf.shape}, took {time.time()-t0:.1f}s")

    models = {
        "TF-IDF + Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="liblinear", random_state=42
        ),
        "TF-IDF + Naive Bayes": MultinomialNB(alpha=0.1),
        "TF-IDF + SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, random_state=42), cv=3
        ),
    }

    all_results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        res_en = evaluate_model(name, model, X_test_en, y_test_en, tfidf, "English Test (20%)")
        res_en["train_time_s"] = round(train_time, 1)
        all_results.append(res_en)

        res_multi = evaluate_model(name, model, X_val_multi, y_val_multi, tfidf, "Multilingual Val")
        all_results.append(res_multi)

        for lang in sorted(val_langs.unique()):
            mask = val_langs == lang
            res_lang = evaluate_model(name, model,
                X_val_multi[mask], y_val_multi[mask], tfidf, f"Val-{lang}")
            all_results.append(res_lang)

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")

    print("\n=== SUMMARY TABLE (English Test) ===")
    en_results = results_df[results_df["split"] == "English Test (20%)"]
    print(en_results[["model", "auc_roc", "f1", "accuracy", "precision", "recall"]].to_string(index=False))

    print("\n=== SUMMARY TABLE (Multilingual Val) ===")
    multi_results = results_df[results_df["split"] == "Multilingual Val"]
    print(multi_results[["model", "auc_roc", "f1", "accuracy", "precision", "recall"]].to_string(index=False))

    results_json = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON results saved to {results_json}")


if __name__ == "__main__":
    main()
