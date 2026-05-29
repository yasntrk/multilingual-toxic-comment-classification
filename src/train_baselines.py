#!/usr/bin/env python3
"""TF-IDF baseline models for multilingual toxic comment classification.

Three classical baselines share one TF-IDF representation:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM (probability-calibrated)

These use ASCII-only cleaning on purpose, so they are strong on English and
collapse on the multilingual validation set - quantifying the gap that the
mBERT models are meant to close.

Besides the metrics CSV/JSON, this script also dumps per-split prediction
probabilities (for ROC/PR curves) and the top TF-IDF features for the toxic
class (for the feature-importance plot in evaluate.py).
"""

import os
import json
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import data_utils
import metrics_utils

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def predicted_proba(model, X_tfidf):
    """Return a 1D toxic-probability (or score) array for any of the models."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_tfidf)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_tfidf)
    return model.predict(X_tfidf)


def save_feature_importance(tfidf, lr_model, top_k=30):
    """Export the most toxic / least toxic TF-IDF features from the LR model."""
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = lr_model.coef_[0]
    order = np.argsort(coefs)
    most_toxic = [{"feature": feature_names[i], "weight": round(float(coefs[i]), 4)}
                  for i in order[::-1][:top_k]]
    least_toxic = [{"feature": feature_names[i], "weight": round(float(coefs[i]), 4)}
                   for i in order[:top_k]]
    path = os.path.join(RESULTS_DIR, "tfidf_feature_importance.json")
    with open(path, "w") as f:
        json.dump({"most_toxic": most_toxic, "least_toxic": least_toxic}, f, indent=2)
    print(f"Feature importance saved to {path}")


def main():
    splits = data_utils.load_splits(clean_fn=data_utils.aggressive_clean)

    print("\nFitting TF-IDF vectorizer (50K features, unigrams+bigrams) ...")
    t0 = time.time()
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(splits["X_train"])
    X_test_tfidf = tfidf.transform(splits["X_test_en"])
    X_val_tfidf = tfidf.transform(splits["X_val"])
    print(f"  TF-IDF shape: {X_train_tfidf.shape}, took {time.time() - t0:.1f}s")

    models = {
        "TF-IDF + Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="liblinear", random_state=42),
        "TF-IDF + Naive Bayes": MultinomialNB(alpha=0.1),
        "TF-IDF + SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, random_state=42), cv=3),
    }

    all_rows = []
    all_predictions = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        t0 = time.time()
        model.fit(X_train_tfidf, splits["y_train"])
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        prob_en = predicted_proba(model, X_test_tfidf)
        prob_val = predicted_proba(model, X_val_tfidf)

        row_en = {"model": name, "split": "English Test (20%)",
                  **metrics_utils.compute_metrics(splits["y_test_en"], prob_en),
                  "train_time_s": round(train_time, 1)}
        row_val = {"model": name, "split": "Multilingual Val",
                   **metrics_utils.compute_metrics(splits["y_val"], prob_val)}
        all_rows.extend([row_en, row_val])

        langs = splits["val_langs"]
        for lang in sorted(langs.unique()):
            mask = (langs == lang).to_numpy()
            all_rows.append({
                "model": name, "split": f"Val-{lang}",
                **metrics_utils.compute_metrics(
                    np.asarray(splits["y_val"])[mask], prob_val[mask])})

        all_predictions[name] = {
            "English Test (20%)": (np.asarray(splits["y_test_en"]), prob_en, None),
            "Multilingual Val": (np.asarray(splits["y_val"]), prob_val,
                                 np.asarray(splits["val_langs"])),
        }
        print(f"  English Test  AUC={row_en['auc_roc']:.4f} F1={row_en['f1']:.4f}")
        print(f"  Multilingual  AUC={row_val['auc_roc']:.4f} F1={row_val['f1']:.4f}")

    # Save metrics.
    results_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    results_df.to_csv(csv_path, index=False)
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nResults saved to {csv_path}")

    # Save prediction probabilities for ROC/PR plotting.
    for name, splits_preds in all_predictions.items():
        short = name.replace("TF-IDF + ", "").replace(" ", "_")
        for split, (y_true, y_prob, lang) in splits_preds.items():
            out = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
            if lang is not None:
                out["lang"] = lang
            tag = split.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')
            out.to_csv(os.path.join(RESULTS_DIR, f"preds_{short}_{tag}.csv"), index=False)

    # Feature importance from the Logistic Regression model.
    save_feature_importance(tfidf, models["TF-IDF + Logistic Regression"])

    print("\n=== English Test summary ===")
    en = results_df[results_df["split"] == "English Test (20%)"]
    print(en[["model", "auc_roc", "f1", "accuracy", "precision", "recall"]]
          .to_string(index=False))
    print("\n=== Multilingual Val summary ===")
    ml = results_df[results_df["split"] == "Multilingual Val"]
    print(ml[["model", "auc_roc", "f1", "accuracy", "precision", "recall"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
