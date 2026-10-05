#!/usr/bin/env python3
"""
train_mac.py
Optimized ByT5 Training Pipeline for Apple Silicon (M-Series / MPS).

Features:
- Native Apple Silicon MPS (Metal Performance Shaders) acceleration.
- Automatically reads local .env configuration without Git exposure.
- Zero risk of GitHub leaks: outputs to checkpoints/ (git-ignored).
- Real-time progress logging and immediate post-training test inference.
"""

import os
import sys
import json
import argparse
from typing import List, Dict

# 1. Load .env if present (pure python, no dotenv package needed)
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class GeezDataset(Dataset if TRANSFORMERS_AVAILABLE else object):
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        model_inputs = self.tokenizer(
            item["input_text"],
            max_length=self.max_len,
            truncation=True
        )
        labels = self.tokenizer(
            text_target=item["target_text"],
            max_length=self.max_len,
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def load_data(dataset_dir: str, limit: int = 20000) -> List[Dict[str, str]]:
    train_file = os.path.join(dataset_dir, "output", "synthetic_morphology_train.jsonl")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found at: {train_file}")

    print(f"Reading synthetic dataset from {train_file}...")
    with open(train_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    import random
    rng = random.Random(42)
    if limit and limit < len(all_lines):
        print(f"Uniformly sampling {limit} pairs across all roots (from {len(all_lines)} total)...")
        selected_lines = rng.sample(all_lines, limit)
    else:
        selected_lines = all_lines
        rng.shuffle(selected_lines)

    samples = []
    for line in selected_lines:
        d = json.loads(line)
        feat = d.get("features", {})
        surface = d.get("surface_form", "")
        
        input_text = f"analyze: {surface}"
        target_text = f"root: {feat.get('root', '')} | tense: {feat.get('tense', '')} | person: {feat.get('person', '')} | gender: {feat.get('gender', '')} | number: {feat.get('number', '')} | stem: {feat.get('stem', '')}"
        samples.append({"input_text": input_text, "target_text": target_text})

    # Inject authentic Merho Ges training pairs
    gold_train_file = os.path.join(dataset_dir, "data", "paradigms", "merho_ges_gold_train.jsonl")
    if os.path.exists(gold_train_file):
        gold_count = 0
        with open(gold_train_file, "r", encoding="utf-8") as f_gold:
            for line in f_gold:
                rec = json.loads(line)
                root = rec.get("root", "")
                stem = rec.get("stem", "basic")
                for row in rec.get("table_data", []):
                    form_raw = row.get("geez_form", "")
                    if form_raw:
                        variants = [v.strip(" :፡።()[]") for v in form_raw.split("/") if v.strip(" :፡።()[]")]
                        for v in variants:
                            input_text = f"analyze: {v}"
                            target_text = f"root: {root} | tense: {row.get('tense', '')} | person: {row.get('person', '')} | gender: {row.get('gender', '')} | number: {row.get('number', '')} | stem: {stem}"
                            samples.append({"input_text": input_text, "target_text": target_text})
                            gold_count += 1
        print(f"Injected {gold_count} authentic Merho Ges gold training pairs into training mix.")

    rng.shuffle(samples)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train ByT5 on Apple Silicon Mac")
    parser.add_argument("--dataset_dir", default="../geez-dataset-v1", help="Path to geez-dataset-v1")
    parser.add_argument("--output_dir", default="./checkpoints", help="Local checkpoint dir (git-ignored)")
    parser.add_argument("--sample_limit", type=int, default=20000, help="Number of pairs to train on")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 is ideal on 24GB M4)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    args = parser.parse_args()

    if not TRANSFORMERS_AVAILABLE:
        print("[!] PyTorch or HuggingFace Transformers is not installed.")
        print("To install in your local virtual environment, run:")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install torch transformers datasets accelerate")
        sys.exit(1)

    # Check Apple Silicon MPS
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("[✓] Apple Silicon GPU (MPS) detected! Using hardware acceleration.")
    else:
        print("[i] Running on CPU.")

    print(f"Loading ByT5-small model on {device}...")
    model_name = "google/byt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"Loading {args.sample_limit} training pairs from {args.dataset_dir}...")
    samples = load_data(args.dataset_dir, limit=args.sample_limit)
    print(f"Loaded {len(samples)} pairs.")

    split = int(len(samples) * 0.9)
    train_ds = GeezDataset(samples[:split], tokenizer)
    eval_ds = GeezDataset(samples[split:], tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=False,
        fp16=False,
        report_to="none"
    )
    # Note: transformers 5.x removed use_mps_device; MPS is selected automatically when available.

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    print("\nStarting local training on your Mac...")
    trainer.train()

    print(f"\nTraining complete! Checkpoints safely saved in: {args.output_dir}")

    # Post-training inference test
    print("\nRunning test inference on trained weights:")
    model.eval()
    test_words = ["ይቀትሉ", "ተቀተለት", "ይባርኩ", "ይቀድሱ", "ቀተለ"]
    for tw in test_words:
        inp = tokenizer(f"analyze: {tw}", return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_length=128, num_beams=2, repetition_penalty=1.5)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  {tw} -> {pred}")


if __name__ == "__main__":
    main()
