#!/usr/bin/env python3
"""Predict the root of each word in a file with the trained ByT5 model.

Input: a text file with one Ge'ez word per line (empty lines are skipped). Output: a TSV line
"word<TAB>root" per word, the root in Ethiopic letters joined by "-", "none" for a word the model says has no
root, and an empty root when the reply does not parse (analyzer.parse_neural_output).

--task root (the default) asks for the root alone ("root: <word>"); --task analyze asks for the
full analysis ("analyze: <word>") and keeps its root.

  python predict_roots.py words.txt --output roots.tsv [--model_dir byt5_morphology_model]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyzer  # noqa: E402
import dataset_loader  # noqa: E402
from train_byt5_morphology import DEFAULT_OUTPUT_DIR, model_to_load  # noqa: E402

TASKS = {"root": dataset_loader.ROOT_ONLY_TASK, "analyze": dataset_loader.ANALYZE_TASK}
MAX_NEW_TOKENS = {"root": 32, "analyze": 128}   # bytes


def read_words(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def predict_roots(words, generate, task="root", batch_size=32):
    """[(word, root)] for `words`. `generate` maps a list of model inputs to the list of replies."""
    out = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        replies = generate([dataset_loader.input_text(w, TASKS[task]) for w in batch])
        for word, reply in zip(batch, replies):
            out.append((word, root_of_reply(reply)))
    return out


def root_of_reply(reply):
    """The root of a model reply, its letters joined by "-"; "none" for no root; "" when the reply does not parse."""
    parsed = analyzer.parse_neural_output(reply)
    if parsed is None:
        return ""
    if parsed["root"] is None:
        return dataset_loader.NO_ROOT
    return "-".join(c for c in parsed["root"] if c not in "- ")


def load_generator(model_dir, max_new_tokens, device=None):
    """generate(texts) -> replies, greedy, with the model of model_dir (train_byt5_morphology.model_to_load)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    path = model_to_load(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()

    def generate(texts):
        enc = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        return tokenizer.batch_decode(ids, skip_special_tokens=True)
    return generate


def main(argv=None):
    p = argparse.ArgumentParser(description="Write word<TAB>root for each word of a file.")
    p.add_argument("words", help="Text file, one word per line")
    p.add_argument("--output", default=None, help="TSV file to write (default: standard output)")
    p.add_argument("--model_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--task", choices=sorted(TASKS), default="root",
                   help="root: ask for the root alone (default); analyze: ask for the full analysis")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    args = p.parse_args(argv)
    generate = load_generator(args.model_dir, MAX_NEW_TOKENS[args.task], args.device)
    rows = predict_roots(read_words(args.words), generate, args.task, args.batch_size)
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for word, root in rows:
            out.write(f"{word}\t{root}\n")
    finally:
        if args.output:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
