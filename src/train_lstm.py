#!/usr/bin/env python3
"""Bidirectional LSTM baseline for multilingual toxic comment classification.

This is the recurrent deep-learning baseline from the proposal. Unlike the
TF-IDF models it captures word order, but with a vocabulary learned only from
English training data it still cannot transfer well to other languages - which
is exactly the gap the mBERT models are meant to close.

The embedding layer is trained from scratch by default. If you have pretrained
multilingual word vectors (e.g. FastText `cc.*.300.vec`) you can load them with
``--embeddings path/to/vectors.vec`` to give the model a small cross-lingual
head start.

Designed to run on a laptop: pass ``--max-train-samples`` to subsample and use
``--device mps`` on Apple Silicon.

Example:
    python src/train_lstm.py --max-train-samples 40000 --epochs 4 --device mps
"""

import os
import re
import copy
import json
import time
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import data_utils
import metrics_utils

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PAD, UNK = "<pad>", "<unk>"


def tokenize(text):
    """Very small word tokenizer that keeps Unicode word characters."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def build_vocab(texts, min_freq=2, max_size=50000):
    """Build a word->index map from the training texts."""
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {PAD: 0, UNK: 1}
    for word, freq in counter.most_common(max_size):
        if freq < min_freq:
            break
        vocab[word] = len(vocab)
    print(f"  Vocabulary size: {len(vocab)}")
    return vocab


def encode(text, vocab, max_len):
    """Turn a string into a fixed-length list of token ids (padded/truncated)."""
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(text)][:max_len]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.ids = [encode(t, vocab, max_len) for t in texts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (torch.tensor(self.ids[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float))


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=128,
                 num_layers=1, dropout=0.3, pad_idx=0, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # *2 for the two LSTM directions.
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)                  # (B, T, E)
        out, _ = self.lstm(emb)                  # (B, T, 2H)
        pooled = torch.max(out, dim=1).values    # max-pooling over time
        return self.fc(self.dropout(pooled)).squeeze(-1)  # (B,) logits


def load_pretrained_embeddings(path, vocab, embed_dim):
    """Load word vectors in the common text format ``word v1 v2 ...``.

    Words missing from the file keep their random initialisation.
    """
    print(f"  Loading pretrained embeddings from {path} ...")
    matrix = torch.randn(len(vocab), embed_dim) * 0.1
    matrix[vocab[PAD]] = 0
    found = 0
    with open(path, encoding="utf-8") as f:
        first = f.readline().split()
        # Some files start with a "<count> <dim>" header line; skip if so.
        if len(first) != 2:
            f.seek(0)
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab and len(parts) == embed_dim + 1:
                matrix[vocab[word]] = torch.tensor([float(x) for x in parts[1:]])
                found += 1
    print(f"  Matched {found}/{len(vocab)} vocabulary words.")
    return matrix


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y)
    torch.set_grad_enabled(True)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_proba(model, texts, vocab, max_len, device, batch_size=256):
    """Return toxic probabilities for an iterable of raw strings."""
    model.eval()
    probs = []
    ids = [encode(t, vocab, max_len) for t in texts]
    for i in range(0, len(ids), batch_size):
        batch = torch.tensor(ids[i:i + batch_size], dtype=torch.long, device=device)
        probs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def main():
    parser = argparse.ArgumentParser(description="Train a BiLSTM toxicity classifier")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Subsample the training split for faster laptop runs.")
    parser.add_argument("--embeddings", default=None,
                        help="Optional path to pretrained word vectors (text format).")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    torch.manual_seed(data_utils.RANDOM_SEED)

    splits = data_utils.load_splits(
        clean_fn=data_utils.light_clean,
        max_train_samples=args.max_train_samples,
    )

    print("Building vocabulary from training data ...")
    vocab = build_vocab(splits["X_train"])

    pretrained = None
    if args.embeddings:
        pretrained = load_pretrained_embeddings(args.embeddings, vocab, args.embed_dim)

    train_ds = TextDataset(splits["X_train"], splits["y_train"], vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = BiLSTMClassifier(
        vocab_size=len(vocab), embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim, pretrained=pretrained,
    ).to(device)

    # Class imbalance: weight the positive (toxic) class by its inverse frequency.
    pos_ratio = float(np.mean(splits["y_train"]))
    pos_weight = torch.tensor([(1 - pos_ratio) / max(pos_ratio, 1e-6)], device=device)
    print(f"  Positive-class weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("\nTraining ...")
    # Keep the checkpoint with the best multilingual val AUC. The BiLSTM has an
    # English-only vocabulary, so it overfits to English and its cross-lingual
    # AUC peaks early then degrades - we must not report the final (worst) epoch.
    best_auc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_prob = predict_proba(model, splits["X_val"], vocab, args.max_len, device)
        val_metrics = metrics_utils.compute_metrics(splits["y_val"], val_prob)
        print(f"  Epoch {epoch}/{args.epochs}  "
              f"loss={loss:.4f}  val_AUC={val_metrics['auc_roc']:.4f}  "
              f"({time.time()-t0:.0f}s)")
        if val_metrics["auc_roc"] > best_auc:
            best_auc, best_epoch = val_metrics["auc_roc"], epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint from epoch {best_epoch} "
              f"(val_AUC={best_auc:.4f}).")

    # Tune the decision threshold on the multilingual validation set.
    val_prob = predict_proba(model, splits["X_val"], vocab, args.max_len, device)
    threshold, val_f1 = metrics_utils.best_threshold(splits["y_val"], val_prob)
    print(f"\nTuned threshold on multilingual val: {threshold:.2f} (F1={val_f1:.4f})")

    def predict_fn(texts):
        return predict_proba(model, texts, vocab, args.max_len, device)

    print("\n=== BiLSTM evaluation ===")
    rows, predictions = metrics_utils.evaluate_all_splits(
        "BiLSTM", predict_fn, splits, threshold=threshold)
    metrics_utils.print_rows(rows)

    save_results(rows, predictions, threshold)


def resolve_device(choice):
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_results(rows, predictions, threshold):
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "lstm_results.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(RESULTS_DIR, "lstm_results.json"), "w") as f:
        json.dump({"threshold": threshold, "results": rows}, f, indent=2)

    # Dump probabilities so evaluate.py can draw ROC/PR curves.
    for split, (y_true, y_prob, langs) in predictions.items():
        out = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
        if langs is not None:
            out["lang"] = langs
        fname = f"preds_BiLSTM_{split.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')}.csv"
        out.to_csv(os.path.join(RESULTS_DIR, fname), index=False)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
