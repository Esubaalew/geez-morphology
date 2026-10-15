#!/usr/bin/env python3
"""Check an exported training set against the held-out words.

    python3 train_byt5_morphology.py --dataset_dir D --load_only --export_dir OUT [flags]
    python3 scripts/audit_train_leakage.py --dataset_dir D --train OUT/train_pairs.jsonl

Held-out words are the gold val and test pairs and the book rows of the unified file in split val or test. The
training pairs are compared after spelling folding (geez_script.fold_spelling: labiovelar letters plain, twin
letters merged):
  roots   no training root may equal a held-out root exactly, and no training pair may repeat a held-out form with
          the same root (exit status 1 otherwise). Training roots that differ from a held-out root only by
          folding (a spelling variant, such as ሠ-ሀ-የ and ሰ-ሐ-የ) are counted and reported, not failed: the splits
          are disjoint by exact root.
Words without root (target 'root: none') are left out of the root check. generate: pairs are read in reverse
(root from the input analysis, form from the target); pos: pairs are checked by form only. A training pair whose
form is a held-out form with any root also fails.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dataset_loader  # noqa: E402
import geez_script  # noqa: E402


def fold(s):
    return geez_script.fold_spelling(s or "")


def root_of(target):
    if not target.startswith("root:"):
        return None
    r = target.split("|", 1)[0][5:].strip()
    return None if r in ("none", "") else r


def form_of_analysis(inp):
    """The analysis of a generate: input ("generate: root: ... | ...")."""
    return inp.split(":", 1)[1].strip()


def form_of(inp):
    return inp.split(":", 1)[1].strip() if ":" in inp else inp.strip()


def heldout(dataset_dir):
    """{split: [(form, root)]} for val and test: gold pairs and unified book rows."""
    out = collections.defaultdict(list)
    for split in ("val", "test"):
        for p in dataset_loader.load_gold_split(dataset_dir, split):
            out[split].append((p["form"], p["root"]))
        for p in dataset_loader.load_unified(dataset_dir, split, ("book",)):
            out[split].append((p["form"], p["root"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", default=dataset_loader.default_dataset_dir())
    ap.add_argument("--train", required=True, help="train_pairs.jsonl written by the trainer's --export_dir")
    args = ap.parse_args()
    held = heldout(args.dataset_dir)
    exact_roots = {s: {r for _, r in v if r} for s, v in held.items()}
    exact_pairs = {s: {(f, r) for f, r in v} for s, v in held.items()}
    exact_forms = {s: {f for f, _ in v} for s, v in held.items()}
    form_any = collections.Counter()
    folded_roots = {s: {fold(r) for r in rs} for s, rs in exact_roots.items()}
    n = 0
    root_exact, root_variant, form_exact, form_variant = (collections.Counter() for _ in range(4))
    examples = []
    train_roots = set()
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            n += 1
            if d["input_text"].startswith("generate:"):        # analysis -> form
                r, form = root_of(form_of_analysis(d["input_text"])), d["target_text"].strip()
            elif d["input_text"].startswith("pos:"):            # form -> class
                r, form = None, form_of(d["input_text"])
            else:
                r = root_of(d["target_text"])
                form = form_of(d["input_text"])
            for s in held:
                if r:
                    train_roots.add(r)
                    if r in exact_roots[s]:
                        root_exact[s] += 1
                        if len(examples) < 10:
                            examples.append((s, d["input_text"], d["target_text"][:60]))
                    elif fold(r) in folded_roots[s]:
                        root_variant[s] += 1
                if (form, r) in exact_pairs[s]:
                    form_exact[s] += 1
                if form in exact_forms[s]:
                    form_any[s] += 1
    print(f"training pairs: {n}; distinct training roots: {len(train_roots)}")
    for s in sorted(held):
        print(f"held-out {s}: {len(held[s])} pairs, {len(exact_roots[s])} roots")
        print(f"   training pairs with the same root: {root_exact[s]}   same form and root: {form_exact[s]}   (must be 0)")
        print(f"   training pairs with a {s} form (any root): {form_any[s]}   (reported; 0 with the trainer's leak gate)")
        print(f"   training pairs whose root is only a spelling variant of a {s} root (folded): {root_variant[s]}   (reported)")
    for e in examples:
        print("  overlap:", e)
    bad = sum(root_exact.values()) + sum(form_exact.values()) + sum(form_any.values())
    print("RESULT:", "CLEAN (exact roots and pairs)" if not bad else f"{bad} overlap(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
