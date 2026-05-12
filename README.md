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
    preprocess.py          # Data loading and text cleaning
    eda.py                 # Exploratory data analysis and plots
    train_baselines.py     # TF-IDF + LR/NB/SVM baselines
    train_transformer.py   # mBERT and mBERT+adapter training
    evaluate.py            # Metrics computation and visualization
  data/                    # Raw and processed data (gitignored)
  results/                 # Experiment results (CSV/JSON)
  figures/                 # Generated plots
  models/                  # Saved model checkpoints (gitignored)
```

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

# 2. Preprocess data
python src/preprocess.py

# 3. Train TF-IDF baselines
python src/train_baselines.py

# 4. Train mBERT (full fine-tuning)
python src/train_transformer.py --model mbert --epochs 3

# 5. Train mBERT with adapters
python src/train_transformer.py --model mbert-adapter --epochs 3 --adapter-size 128

# 6. View results summary
python src/evaluate.py
```

## Baseline Results (English Test Split)

| Model | AUC-ROC | F1 | Accuracy |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9693 | 0.6993 | 0.9527 |
| TF-IDF + Naive Bayes | 0.9434 | 0.6759 | 0.9445 |
| TF-IDF + SVM | 0.9649 | 0.7299 | 0.9555 |
| LSTM (BiLSTM) | - | - | - |
| mBERT (full fine-tuning) | - | - | - |
| mBERT + Adapter | - | - | - |
