#!/usr/bin/env python3
"""
evaluate_gold.py
Evaluates ByT5 morphology model against the 15% held-out gold test set from Merho Ges.
Reports exact accuracy on root, stem, tense, person, gender, number, and overall match.
"""

import os
import sys
import json
import re
import argparse

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print("[!] PyTorch and Transformers required.")
    sys.exit(1)


def parse_prediction(pred_str):
    """Parse key-value target string into a dictionary."""
    out = {}
    parts = pred_str.split("|")
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def normalize_root(r):
    if not r:
        return ""
    # Remove parenthetical info or transliterations
    r = r.split("(")[0].strip()
    r = r.replace(" ", "")
    # Normalize Ethiopic dashes or ascii dashes
    r = re.sub(r"[-፡]", "-", r)
    return r.lower()


def load_gold_test_pairs(test_file):
    pairs = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            parent_root = rec.get("root", "")
            parent_stem = rec.get("stem", "basic")
            
            for row in rec.get("table_data", []):
                form_raw = row.get("geez_form", "")
                if not form_raw:
                    continue
                # If multiple variants given e.g. "ሃልሀ / ሀልሀ", take each
                variants = [v.strip() for v in form_raw.split("/") if v.strip()]
                for v in variants:
                    # Clean punctuation
                    v_clean = v.strip(" :፡።()[]")
                    if v_clean:
                        pairs.append({
                            "surface": v_clean,
                            "root": parent_root,
                            "stem": parent_stem,
                            "tense": row.get("tense", ""),
                            "person": str(row.get("person", "")),
                            "gender": row.get("gender", ""),
                            "number": row.get("number", "")
                        })
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Evaluate ByT5 on Gold Merho Ges Test Set")
    parser.add_argument("--model_dir", default="./checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--test_file", default="../geez-dataset-v1/data/paradigms/merho_ges_gold_test.jsonl", help="Gold test set path")
    parser.add_argument("--limit", type=int, default=100, help="Number of test pairs to evaluate (0 for all)")
    args = parser.parse_args()

    if not os.path.exists(args.test_file):
        print(f"[!] Test file not found: {args.test_file}")
        sys.exit(1)

    # Find latest checkpoint in model_dir
    chkpt = args.model_dir
    if os.path.exists(args.model_dir):
        subdirs = [os.path.join(args.model_dir, d) for d in os.listdir(args.model_dir) if d.startswith("checkpoint-")]
        if subdirs:
            chkpt = sorted(subdirs, key=lambda x: int(x.split("-")[-1]))[-1]

    print(f"Loading model checkpoint from: {chkpt}...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(chkpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(chkpt).to(device)
    model.eval()

    test_pairs = load_gold_test_pairs(args.test_file)
    if args.limit > 0 and args.limit < len(test_pairs):
        import random
        rng = random.Random(42)
        test_pairs = rng.sample(test_pairs, args.limit)
    print(f"Evaluating on {len(test_pairs)} authentic gold test pairs (random sample, seed 42)...")

    metrics = {
        "total": len(test_pairs),
        "exact_match": 0,
        "stem_match": 0,
        "tense_match": 0,
        "person_match": 0,
        "gender_match": 0,
        "number_match": 0,
        "root_match": 0
    }

    examples = []

    for item in test_pairs:
        surface = item["surface"]
        inp = tokenizer(f"analyze: {surface}", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_length=128, num_beams=2)
        pred_text = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = parse_prediction(pred_text)

        # Compare
        stem_ok = (pred.get("stem") == item["stem"])
        tense_ok = (pred.get("tense") == item["tense"])
        person_ok = (str(pred.get("person")) == str(item["person"]))
        gender_ok = (pred.get("gender") == item["gender"])
        number_ok = (pred.get("number") == item["number"])
        root_ok = normalize_root(pred.get("root")) == normalize_root(item["root"])

        if stem_ok: metrics["stem_match"] += 1
        if tense_ok: metrics["tense_match"] += 1
        if person_ok: metrics["person_match"] += 1
        if gender_ok: metrics["gender_match"] += 1
        if number_ok: metrics["number_match"] += 1
        if root_ok: metrics["root_match"] += 1

        all_ok = stem_ok and tense_ok and person_ok and gender_ok and number_ok
        if all_ok and root_ok:
            metrics["exact_match"] += 1

        if len(examples) < 10:
            examples.append({
                "surface": surface,
                "gold_root": item["root"],
                "gold_stem": item["stem"],
                "gold_tense": item["tense"],
                "pred": pred_text
            })

    print("\n" + "="*50)
    print("GOLD HELD-OUT BENCHMARK RESULTS (MERHO GES)")
    print("="*50)
    n = metrics["total"]
    if n > 0:
        print(f"Stem Accuracy:         {metrics['stem_match']/n*100:6.2f}% ({metrics['stem_match']}/{n})")
        print(f"Tense Accuracy:        {metrics['tense_match']/n*100:6.2f}% ({metrics['tense_match']}/{n})")
        print(f"Person Accuracy:       {metrics['person_match']/n*100:6.2f}% ({metrics['person_match']}/{n})")
        print(f"Gender Accuracy:       {metrics['gender_match']/n*100:6.2f}% ({metrics['gender_match']}/{n})")
        print(f"Number Accuracy:       {metrics['number_match']/n*100:6.2f}% ({metrics['number_match']}/{n})")
        print(f"Root Match Accuracy:   {metrics['root_match']/n*100:6.2f}% ({metrics['root_match']}/{n})")
        print(f"Overall Exact Match:   {metrics['exact_match']/n*100:6.2f}% ({metrics['exact_match']}/{n})")

    print("\nSample Predictions on Held-Out Authentic Forms:")
    for ex in examples:
        print(f"  {ex['surface']} (gold: {ex['gold_root']} | {ex['gold_tense']} | {ex['gold_stem']})")
        print(f"    -> {ex['pred']}")


if __name__ == "__main__":
    main()
