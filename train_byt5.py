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
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer, max_input_length: int = 128, max_target_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["input_text"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            item["target_text"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = targets["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }


def load_morphology_training_data(dataset_dir: str, limit: int = 50000) -> List[Dict[str, str]]:
    """
    Load bidirectional training samples from synthetic_morphology_train.jsonl.
    """
    train_file = os.path.join(dataset_dir, "output", "synthetic_morphology_train.jsonl")
    samples = []
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found at: {train_file}")

    with open(train_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            d = json.loads(line)
            feat = d.get("features", {})
            surface = d.get("surface_form", "")
            
            # Task: Analysis (surface -> root + features)
            root = feat.get("root", "")
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
    parser.add_argument("--sample_limit", type=int, default=20000, help="Number of training samples to load")
    args = parser.parse_args()

    if not TRANSFORMERS_AVAILABLE:
        print("PyTorch or HuggingFace Transformers is not installed in the active environment.")
        print("To install dependencies, run:")
        print("  pip install torch transformers datasets accelerate")
        return

    print(f"Loading ByT5 tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    print(f"Loading Ge'ez morphology dataset from: {args.dataset_dir} (limit: {args.sample_limit})...")
    raw_data = load_morphology_training_data(args.dataset_dir, limit=args.sample_limit)
    print(f"Loaded {len(raw_data)} training examples.")

    # Split train/eval
    split_idx = int(len(raw_data) * 0.9)
    train_data = raw_data[:split_idx]
    eval_data = raw_data[split_idx:]

    train_dataset = GeezMorphologyDataset(train_data, tokenizer)
    eval_dataset = GeezMorphologyDataset(eval_data, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=5e-4,
        num_train_epochs=args.epochs,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available() or torch.backends.mps.is_available(),
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Beginning ByT5 fine-tuning...")
    trainer.train()
    print(f"Model checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
