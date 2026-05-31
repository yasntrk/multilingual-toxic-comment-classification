#!/usr/bin/env python3
"""Interactive toxicity checker for the trained mBERT model.

Type a comment in any language and the model tells you whether it looks
**TOXIC** or **NOT TOXIC**, together with the toxic probability.

It loads a model saved by ``train_transformer.py --save-model`` (default the
adapter model in ``models/mBERT-adapter-64``). It works for both the adapter
model and the full fine-tuned model - adapters are detected and rebuilt
automatically from the saved weights.

Two ways to run it:

    # 1. Web UI in the browser (needs `pip install gradio`):
    python src/predict_app.py

    # 2. Plain terminal prompt (no extra dependency):
    python src/predict_app.py --cli

Useful flags:
    --model-dir models/mBERT-full   # use a different saved model
    --threshold 0.5                 # override the tuned decision cut-off
    --device mps                    # force a device (auto by default)
"""

import os
import json
import argparse

import torch

import data_utils                 # light_clean (same cleaning used in training)
import train_transformer as T     # reuse adapter modules + predict_proba + device

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_MODEL_DIR = os.path.join(MODELS_DIR, "mBERT-adapter-64")

EXAMPLES = [
    "You are amazing, thank you so much for the help!",
    "Shut up, you absolute idiot.",
    "Aptal herif, defol git buradan.",          # Turkish (toxic)
    "Eres un completo imbécil.",                 # Spanish (toxic)
    "Grazie mille, sei stato molto utile!",      # Italian (clean)
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_state_dict(model_dir):
    """Load the saved weights from safetensors (preferred) or a .bin file."""
    safet = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(safet):
        from safetensors.torch import load_file
        return load_file(safet)
    binp = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(binp):
        return torch.load(binp, map_location="cpu")
    raise FileNotFoundError(
        f"No model weights found in {model_dir}. "
        "Train one first with: python src/train_transformer.py "
        "--mode adapter --save-model")


def _rename_legacy_keys(state):
    """Map old HuggingFace LayerNorm names (gamma/beta) to PyTorch's weight/bias.

    ``from_pretrained`` does this rename automatically, but we load the state
    dict by hand (to re-inject adapters first), so we must do it ourselves -
    otherwise every LayerNorm silently stays at its default init and the model
    produces garbage predictions.
    """
    renamed = {}
    for key, tensor in state.items():
        new_key = key.replace("LayerNorm.gamma", "LayerNorm.weight") \
                     .replace("LayerNorm.beta", "LayerNorm.bias")
        renamed[new_key] = tensor
    return renamed


def _detect_adapter_size(state):
    """If the weights contain adapter layers, return their bottleneck size."""
    for key, tensor in state.items():
        if key.endswith("adapter.down.weight"):
            return tensor.shape[0]   # (bottleneck, hidden)
    return None


def load_model(model_dir, device):
    """Rebuild the trained model (adapter or full) and load its weights."""
    from transformers import (
        AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    )

    print(f"Loading model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    state = _rename_legacy_keys(_load_state_dict(model_dir))
    adapter_size = _detect_adapter_size(state)

    # Build the architecture from the saved config (no re-download of weights),
    # inject adapters if needed, then load the trained weights into it.
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_config(config)
    if adapter_size is not None:
        T.inject_adapters(model, adapter_size)
        print(f"  Detected adapter model (bottleneck={adapter_size}).")
    else:
        print("  Detected full fine-tuned model.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  (note: {len(missing)} weights left at init - usually buffers)")
    model.to(device).eval()
    return model, tokenizer, adapter_size


def load_threshold(model_dir, override):
    """Decision cut-off: CLI override, else the value tuned during training."""
    if override is not None:
        return float(override)
    label = os.path.basename(os.path.normpath(model_dir))
    path = os.path.join(RESULTS_DIR, f"{label}_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return float(json.load(f).get("threshold", 0.5))
    print("  (no tuned threshold found, falling back to 0.50)")
    return 0.5


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def make_classifier(model, tokenizer, device, threshold, max_len, clean):
    """Return a function: text -> (label, probability)."""
    def classify(text):
        text = (text or "").strip()
        if not text:
            return "—", 0.0
        prepared = data_utils.light_clean(text) if clean else text
        prob = float(T.predict_proba(model, tokenizer, [prepared], max_len, device)[0])
        label = "TOXIC" if prob >= threshold else "NOT TOXIC"
        return label, prob
    return classify


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
def run_cli(classify, threshold):
    print("\n" + "=" * 60)
    print("  Toxic comment checker - type a comment, press Enter.")
    print("  Type 'quit' (or press Ctrl-D) to exit.")
    print(f"  Decision threshold: {threshold:.2f}")
    print("=" * 60)
    while True:
        try:
            text = input("\ncomment> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if text.strip().lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if not text.strip():
            continue
        label, prob = classify(text)
        mark = "[!]" if label == "TOXIC" else "[ok]"
        print(f"  {mark} {label}   (toxic probability {prob:.1%})")


def run_web(classify, threshold, model_label):
    import gradio as gr

    def predict(text):
        label, prob = classify(text)
        if label == "TOXIC":
            verdict = f"## 🚫 TOXIC\nToxic probability: **{prob:.1%}**"
        elif label == "—":
            verdict = "Please type a comment above."
        else:
            verdict = f"## ✅ NOT TOXIC\nToxic probability: **{prob:.1%}**"
        verdict += f"\n\n*(flagged toxic when probability ≥ {threshold:.0%})*"
        return verdict

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=3, label="Comment (any language)",
                          placeholder="Type a comment in English, Turkish, Spanish, Italian, ..."),
        outputs=gr.Markdown(label="Result"),
        title="Multilingual Toxic Comment Classifier",
        description=(f"mBERT model ({model_label}). Trained on English, works "
                     "across languages. Decision threshold tuned on the "
                     "multilingual validation set."),
        examples=[[e] for e in EXAMPLES],
        allow_flagging="never",
    )
    print("\nLaunching web UI - open the printed local URL in your browser.")
    demo.launch()


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Interactive toxicity checker")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Folder of a saved model (default: models/mBERT-adapter-64).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the tuned decision threshold.")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip text cleaning (feed raw text to the model).")
    parser.add_argument("--cli", action="store_true",
                        help="Force the terminal interface instead of the web UI.")
    args = parser.parse_args()

    device = T.resolve_device(args.device)
    print(f"Using device: {device}")

    model, tokenizer, _ = load_model(args.model_dir, device)
    threshold = load_threshold(args.model_dir, args.threshold)
    classify = make_classifier(
        model, tokenizer, device, threshold, args.max_len, clean=not args.no_clean)
    model_label = os.path.basename(os.path.normpath(args.model_dir))

    if args.cli:
        run_cli(classify, threshold)
        return

    try:
        run_web(classify, threshold, model_label)
    except ImportError:
        print("\n(gradio not installed - using the terminal interface instead.)")
        print("  For the browser UI, run:  pip install gradio")
        run_cli(classify, threshold)


if __name__ == "__main__":
    main()
