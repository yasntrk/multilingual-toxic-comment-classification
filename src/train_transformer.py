#!/usr/bin/env python3
"""Transformer-based models (mBERT, mBERT+Adapter) for Multilingual Toxic Comment Classification.

TODO: Complete implementation for the final submission.
- mBERT full fine-tuning
- mBERT with adapter modules (Houlsby et al., 2019)
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.float),
        }


def train_mbert(use_adapters=False, adapter_size=128, epochs=3, batch_size=16, lr=2e-5):
    """Train mBERT with optional adapter modules."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if use_adapters:
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        print(f"Adapter mode: frozen base, trainable classifier")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "jigsaw-toxic-comment-train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))

    train_dataset = ToxicCommentDataset(train_df["comment_text"], train_df["toxic"], tokenizer)
    val_dataset = ToxicCommentDataset(val_df["comment_text"], val_df["toxic"], tokenizer)

    os.makedirs(MODELS_DIR, exist_ok=True)
    output_dir = os.path.join(MODELS_DIR, "mbert_adapter" if use_adapters else "mbert_full")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=lr,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer models")
    parser.add_argument("--model", choices=["mbert", "mbert-adapter"], default="mbert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adapter-size", type=int, default=128)
    args = parser.parse_args()

    use_adapters = args.model == "mbert-adapter"
    train_mbert(
        use_adapters=use_adapters,
        adapter_size=args.adapter_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
