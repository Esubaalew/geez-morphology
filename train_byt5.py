#!/usr/bin/env python3
"""
train_byt5.py
Training pipeline for fine-tuning ByT5 (Byte-Level Seq2Seq) on Classical Ge'ez morphology and translation.

Why ByT5?
Standard subword tokenizers (BPE) fragment Ethiopic syllables and obscure root-and-pattern
Semitic morphology. ByT5 operates directly on UTF-8 bytes with zero tokenizer vocabulary,
enabling direct character-level and root-level representation learning.
"""

import os
import json
import argparse
from typing import List, Dict, Any

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


class GeezMorphologyDataset(Dataset if TRANSFORMERS_AVAILABLE else object):
    """
    Dataset wrapper for Ge'ez morphological analysis and translation tasks.
    Uses dynamic tokenization without manual label masking bugs.
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        model_inputs = self.tokenizer(
            item["input_text"],
            max_length=self.max_length,
            truncation=True
        )
        labels = self.tokenizer(
            text_target=item["target_text"],
            max_length=self.max_length,
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


import geez_config

def to_ethiopic_root(root_str: str) -> str:
    if not root_str:
        return ""
    if "(" in root_str and any(0x1200 <= ord(c) <= 0x137F for c in root_str):
        eth_chars = [c for c in root_str if 0x1200 <= ord(c) <= 0x137F]
        return "-".join(eth_chars)
    if any(0x1200 <= ord(c) <= 0x137F for c in root_str):
        chars = [c for c in root_str if 0x1200 <= ord(c) <= 0x137F]
        return "-".join(chars)
    parts = root_str.split("-")
    eth_parts = []
    for p in parts:
        p_clean = p.strip().lower()
        if p_clean in geez_config.CONSONANTS:
            eth_parts.append(geez_config.render_unit(p_clean, "a"))
        elif p_clean == "ś":
            eth_parts.append("ሠ")
        elif p_clean == "ḥ":
            eth_parts.append("ሐ")
        elif p_clean == "ḫ":
            eth_parts.append("ኀ")
        elif p_clean == "ṣ":
            eth_parts.append("ጸ")
        elif p_clean == "ḍ":
            eth_parts.append("ፀ")
        elif p_clean == "ṭ":
            eth_parts.append("ጠ")
        else:
            eth_parts.append(p_clean)
    return "-".join(eth_parts)

def load_morphology_training_data(dataset_dir: str, limit: int = 30000) -> List[Dict[str, str]]:
    """
    Load bidirectional training samples from synthetic_morphology_train.jsonl.
    Uses uniform random sampling across all roots (fixed seed for reproducibility).
    """
    import geez_config
    train_file = os.path.join(dataset_dir, "output", "synthetic_morphology_train.jsonl")
    samples = []
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found at: {train_file}")

    with open(train_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    import random
    rng = random.Random(42)
    if limit and limit < len(all_lines):
        selected_lines = rng.sample(all_lines, limit)
    else:
        selected_lines = all_lines
        rng.shuffle(selected_lines)

    for line in selected_lines:
            d = json.loads(line)
            feat = d.get("features", {})
            surface = d.get("surface_form", "")
            
            root = to_ethiopic_root(feat.get("root", ""))
            tense = feat.get("tense", "")
            person = feat.get("person", "")
            gender = feat.get("gender", "")
            number = feat.get("number", "")
            stem = feat.get("stem", "")
            
            input_text = f"analyze: {surface}"
            target_text = f"root: {root} | tense: {tense} | person: {person} | gender: {gender} | number: {number} | stem: {stem}"
            
            samples.append({
                "input_text": input_text,
                "target_text": target_text
            })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Train ByT5 on Ge'ez Morphology")
    parser.add_argument("--model_name", default="google/byt5-small", help="Pretrained ByT5 model name")
    parser.add_argument("--dataset_dir", default="../geez-dataset-v1", help="Path to geez-dataset-v1")
    parser.add_argument("--output_dir", default="./byt5_geez_checkpoint", help="Checkpoint output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--sample_limit", type=int, default=25000, help="Number of training samples to load")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3 for ByT5)")
    args = parser.parse_args()

    if not TRANSFORMERS_AVAILABLE:
        print("PyTorch or HuggingFace Transformers is not installed in the active environment.")
        print("To install dependencies, run:")
        print("  pip install torch transformers datasets accelerate")
        return

    print(f"Loading ByT5 model & tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    print(f"Loading Ge'ez morphology dataset from: {args.dataset_dir} (limit: {args.sample_limit})...")
    raw_data = load_morphology_training_data(args.dataset_dir, limit=args.sample_limit)
    print(f"Loaded {len(raw_data)} training examples.")

    split_idx = int(len(raw_data) * 0.9)
    train_data = raw_data[:split_idx]
    eval_data = raw_data[split_idx:]

    train_dataset = GeezMorphologyDataset(train_data, tokenizer)
    eval_dataset = GeezMorphologyDataset(eval_data, tokenizer)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=False,
        fp16=False,   # Must be False on T5/ByT5 to avoid NaN underflow
        bf16=use_bf16,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Beginning ByT5 fine-tuning with active gradient tracking...")
    trainer.train()
    print(f"Model checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
