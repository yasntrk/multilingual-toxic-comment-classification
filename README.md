# Multilingual Toxic Comment Classification

SEDS 537 - Machine Learning | Term Project | Spring 2026

Binary toxicity classification across multiple languages using TF-IDF baselines, LSTM, and adapter-based mBERT fine-tuning.

## Dataset

[Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/datasets/julian3833/jigsaw-multilingual-toxic-comment-classification) from Kaggle.

- **Training**: ~223K English comments with binary toxicity labels
- **Validation**: ~8K multilingual comments (Turkish, Spanish, Italian)
- **Test**: ~63K multilingual comments

### Download

```bash
kaggle datasets download -d julian3833/jigsaw-multilingual-toxic-comment-classification -p data/
unzip data/jigsaw-multilingual-toxic-comment-classification.zip -d data/
```

## Project Structure

```
multilingual-toxic-comment-classification/
  README.md
  requirements.txt
  .gitignore
  src/
    data_utils.py          # Shared loading, cleaning, train/test split
    metrics_utils.py       # Shared metrics + threshold tuning + per-language eval
    preprocess.py          # Optional: write cleaned splits to disk
    eda.py                 # Exploratory data analysis and plots
    train_baselines.py     # TF-IDF + LR/NB/SVM baselines
    train_lstm.py          # BiLSTM baseline (PyTorch)
    train_transformer.py   # mBERT full fine-tuning and mBERT + adapters
    evaluate.py            # Comparison tables and all visualizations
    predict_app.py         # Interactive demo: type a comment -> toxic / not toxic
  data/                    # Raw and processed data (gitignored)
  results/                 # Experiment results (CSV/JSON, predictions)
  figures/                 # Generated plots
  models/                  # Saved model checkpoints (gitignored)
  presentation/            # Term-project slide deck (build script + .pptx)
```

### Text cleaning

`data_utils` provides two cleaners, used deliberately:

- `aggressive_clean` (ASCII only) — used by the TF-IDF baselines. Strong on
  English, collapses on other languages, which quantifies the cross-lingual gap.
- `light_clean` (Unicode-preserving) — used by the LSTM and mBERT models.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Exploratory data analysis
python src/eda.py

# 2. Train TF-IDF baselines (LR / NB / SVM)
python src/train_baselines.py

# 3. Train the BiLSTM baseline
python src/train_lstm.py --max-train-samples 40000 --epochs 4 --device mps

# 4. Train mBERT with adapters (parameter-efficient, the proposed method)
python src/train_transformer.py --mode adapter --adapter-size 64 \
    --max-train-samples 30000 --epochs 2 --device mps --save-model

# 5. Train mBERT with full fine-tuning (the strong baseline)
python src/train_transformer.py --mode full \
    --max-train-samples 20000 --epochs 1 --device mps

# 6. Build comparison tables and all figures
python src/evaluate.py

# 7. (optional) t-SNE of mBERT embeddings — needs a model saved in step 4
python src/evaluate.py --task tsne --model-dir models/mBERT-adapter-64

# 8. Interactive demo — type a comment, get toxic / not toxic
#    Needs a model saved in step 4. Opens a browser UI if `gradio` is
#    installed, otherwise falls back to a terminal prompt.
python src/predict_app.py                  # web UI (or terminal if no gradio)
python src/predict_app.py --cli            # force terminal prompt
```

### Running on a laptop (Apple Silicon / CPU)

The full English training set has ~223K rows, so for a few-hour budget on a
MacBook use `--device mps` and `--max-train-samples` to subsample. Adapter mode
trains <2% of mBERT's parameters and is the fastest transformer option. The
English test split is always evaluated on the full holdout regardless of
subsampling, so results stay comparable.

### Ablation knobs

- `--adapter-size {64,128,256}` — adapter bottleneck capacity.
- `--max-len {64,128,256}` — sequence length vs. speed trade-off.
- `--no-clean` — feed raw text to mBERT (cleaning ablation).
- `--mode {full,adapter}` — full fine-tuning vs. parameter-efficient adapters.

## Baseline Results (English Test Split)

| Model | AUC-ROC | F1 | Accuracy |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9693 | 0.6993 | 0.9527 |
| TF-IDF + Naive Bayes | 0.9434 | 0.6759 | 0.9445 |
| TF-IDF + SVM | 0.9649 | 0.7299 | 0.9555 |
| LSTM (BiLSTM) | - | - | - |
| mBERT (full fine-tuning) | - | - | - |
| mBERT + Adapter | - | - | - |
