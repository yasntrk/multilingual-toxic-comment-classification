#!/usr/bin/env python3
"""mBERT models for multilingual toxic comment classification.

Two modes:

- ``--mode full``    : fine-tune every parameter of mBERT (the strong baseline).
- ``--mode adapter`` : freeze mBERT and train only small bottleneck adapter
                       modules + the classification head + LayerNorms. This is
                       the parameter-efficient method proposed for the project,
                       following Houlsby et al. (2019).

The adapters are implemented by hand (no extra library) so the mechanism is
fully visible: a bottleneck adapter is inserted after the projection of each
attention and feed-forward sub-layer, with its own residual connection.

The model is set up as a single-logit binary classifier (``num_labels=1``) and
trained with a class-weighted ``BCEWithLogitsLoss`` to counter the ~10% toxic
class imbalance. The decision threshold is tuned on the multilingual validation
set.

Laptop-friendly: use ``--device mps`` on Apple Silicon and ``--max-train-samples``
to keep runtime to a couple of hours. Adapter mode trains far fewer parameters
and is the faster of the two.

Examples:
    python src/train_transformer.py --mode adapter --adapter-size 64 \
        --max-train-samples 30000 --epochs 2 --device mps
    python src/train_transformer.py --mode full \
        --max-train-samples 20000 --epochs 1 --device mps
"""

import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import data_utils
import metrics_utils

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_NAME = "bert-base-multilingual-cased"


# --------------------------------------------------------------------------- #
# Adapter modules
# --------------------------------------------------------------------------- #
class BottleneckAdapter(nn.Module):
    """Houlsby-style bottleneck adapter with an internal residual connection.

    down-project -> non-linearity -> up-project, then add the input back.
    Weights are initialised near zero so that, at the start of training, the
    adapter is approximately the identity and does not disrupt the pretrained
    representations.
    """

    def __init__(self, hidden_size, bottleneck):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class OutputWithAdapter(nn.Module):
    """Drop-in replacement for BertSelfOutput / BertOutput with an adapter.

    Both original modules share the same structure and forward signature
    ``(hidden_states, input_tensor)``:
        dense -> dropout -> (adapter) -> LayerNorm(x + input_tensor)
    The adapter is inserted just before the residual add + LayerNorm, exactly
    where Houlsby et al. place it.
    """

    def __init__(self, original, bottleneck):
        super().__init__()
        self.dense = original.dense
        self.dropout = original.dropout
        self.LayerNorm = original.LayerNorm
        self.adapter = BottleneckAdapter(self.dense.out_features, bottleneck)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def inject_adapters(model, bottleneck):
    """Insert adapters into every transformer block of an mBERT model."""
    for layer in model.bert.encoder.layer:
        layer.attention.output = OutputWithAdapter(layer.attention.output, bottleneck)
        layer.output = OutputWithAdapter(layer.output, bottleneck)
    return model


def freeze_for_adapters(model):
    """Freeze pretrained weights, leaving adapters + classifier + LayerNorms trainable."""
    trainable_keys = ("adapter", "classifier", "LayerNorm")
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in trainable_keys)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


# --------------------------------------------------------------------------- #
# Training / inference
# --------------------------------------------------------------------------- #
def resolve_device(choice):
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_proba(model, tokenizer, texts, max_length, device, batch_size=64):
    """Return toxic probabilities for an iterable of raw strings."""
    model.eval()
    texts = list(texts)
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(
            [str(t) for t in texts[i:i + batch_size]],
            max_length=max_length, padding=True, truncation=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**batch).logits.squeeze(-1)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def train(args):
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup,
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    torch.manual_seed(data_utils.RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    if args.mode == "adapter":
        model = inject_adapters(model, args.adapter_size)
        freeze_for_adapters(model)
        print(f"Adapter mode (bottleneck={args.adapter_size}).")
    else:
        print("Full fine-tuning mode.")

    trainable, total = count_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    model.to(device)

    # The ablation can disable text cleaning entirely (--no-clean).
    clean_fn = (lambda x: x if isinstance(x, str) else "") if args.no_clean \
        else data_utils.light_clean
    splits = data_utils.load_splits(
        clean_fn=clean_fn, max_train_samples=args.max_train_samples)

    train_ds = ToxicCommentDataset(
        splits["X_train"], splits["y_train"], tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    pos_ratio = float(np.mean(splits["y_train"]))
    pos_weight = torch.tensor([(1 - pos_ratio) / max(pos_ratio, 1e-6)], device=device)
    print(f"Positive-class weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    print("\nTraining ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits.squeeze(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)}  "
                      f"loss={running / step:.4f}")

        val_prob = predict_proba(model, tokenizer, splits["X_val"],
                                 args.max_len, device)
        val_metrics = metrics_utils.compute_metrics(splits["y_val"], val_prob)
        print(f"  Epoch {epoch} done: val_AUC={val_metrics['auc_roc']:.4f} "
              f"({time.time() - t0:.0f}s)")

    return model, tokenizer, splits, device, (trainable, total)


def main():
    parser = argparse.ArgumentParser(description="Train mBERT (full or adapter)")
    parser.add_argument("--mode", choices=["full", "adapter"], default="adapter")
    parser.add_argument("--adapter-size", type=int, default=64,
                        help="Adapter bottleneck dimension (try 64/128/256).")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None,
                        help="Defaults to 1e-4 for adapters, 2e-5 for full fine-tuning.")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--no-clean", action="store_true",
                        help="Ablation: feed raw text without cleaning.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    # Adapters tolerate (and need) a higher learning rate than full fine-tuning.
    if args.lr is None:
        args.lr = 1e-4 if args.mode == "adapter" else 2e-5

    model, tokenizer, splits, device, (trainable, total) = train(args)

    # Tune threshold on the multilingual validation set.
    val_prob = predict_proba(model, tokenizer, splits["X_val"], args.max_len, device)
    threshold, val_f1 = metrics_utils.best_threshold(splits["y_val"], val_prob)
    print(f"\nTuned threshold on multilingual val: {threshold:.2f} (F1={val_f1:.4f})")

    # Measure inference throughput on a fixed sample.
    sample = splits["X_val"][: min(512, len(splits["X_val"]))]
    t0 = time.time()
    predict_proba(model, tokenizer, sample, args.max_len, device)
    infer_time = time.time() - t0
    ms_per_sample = 1000 * infer_time / len(sample)

    model_label = f"mBERT-adapter-{args.adapter_size}" if args.mode == "adapter" else "mBERT-full"

    def predict_fn(texts):
        return predict_proba(model, tokenizer, texts, args.max_len, device)

    print(f"\n=== {model_label} evaluation ===")
    rows, predictions = metrics_utils.evaluate_all_splits(
        model_label, predict_fn, splits, threshold=threshold)
    metrics_utils.print_rows(rows)

    save_results(model_label, rows, predictions, threshold,
                 trainable, total, ms_per_sample, args)

    if args.save_model:
        os.makedirs(MODELS_DIR, exist_ok=True)
        out = os.path.join(MODELS_DIR, model_label)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print(f"Model saved to {out}")


def save_results(model_label, rows, predictions, threshold,
                 trainable, total, ms_per_sample, args):
    import pandas as pd
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, f"{model_label}_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary = {
        "model": model_label,
        "mode": args.mode,
        "adapter_size": args.adapter_size if args.mode == "adapter" else None,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": round(100 * trainable / total, 2),
        "ms_per_sample": round(ms_per_sample, 2),
        "max_len": args.max_len,
        "epochs": args.epochs,
        "threshold": round(threshold, 3),
        "results": rows,
    }
    with open(os.path.join(RESULTS_DIR, f"{model_label}_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for split, (y_true, y_prob, langs) in predictions.items():
        out = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
        if langs is not None:
            out["lang"] = langs
        tag = split.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')
        out.to_csv(os.path.join(RESULTS_DIR, f"preds_{model_label}_{tag}.csv"),
                   index=False)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
