#!/usr/bin/env python3
"""Hold out the synthetic lines whose root is a gold val or test root.

The generator (generate_synthetic_fixed.py) writes the full synthetic pair of files. This one split step
moves every line whose root is a root of the current gold val or test file out of the training file into
a separate held-out file in data/synthetic/; nothing is deleted, the lines move, both files keep the
generator's order. The paired analyze file is split on the same lines.

  data/paradigms/synthetic_morphology_train.jsonl        synthetic TRAIN (the trainer reads only this)
  data/paradigms/synthetic_morphology_analyze.jsonl      its analyze mirror
  data/synthetic/synthetic_forms_for_val_test_roots.jsonl            held out: lines with a val/test root
  data/synthetic/synthetic_forms_for_val_test_roots_analyze.jsonl    their analyze mirror

Gold val/test roots: for every row of merho_ges_gold_val.jsonl and merho_ges_gold_test.jsonl, the row's
root and the record's root, each "/" alternative, in the 1st order (geez_config.first_order_root); only
roots of Ethiopic letters. A synthetic line is held out when its root, written in Ethiopic letters of
the 1st order, is one of them.

Usage:
  python synthetic_heldout.py --full_dir <generator output dir> --dataset_dir <dataset repo> [--write | --check]
    --write   write the four files into the dataset repo
    --check   compare what the split gives with the four files in the dataset repo (exit 1 if different)
  The generator calls split_into_dataset() itself when it mirrors into the dataset (generate_synthetic_fixed.py).
"""
import argparse
import collections
import json
import os
import re
import sys

import geez_config

TRAIN, ANALYZE = "synthetic_morphology_train.jsonl", "synthetic_morphology_analyze.jsonl"
HELDOUT_DIR = os.path.join("data", "synthetic")
HELDOUT_TRAIN = "synthetic_forms_for_val_test_roots.jsonl"
HELDOUT_ANALYZE = "synthetic_forms_for_val_test_roots_analyze.jsonl"
GOLD_EVAL = {"val": "merho_ges_gold_val.jsonl", "test": "merho_ges_gold_test.jsonl"}
ETHIOPIC_ROOT = re.compile(r"[\u1200-\u137F](-[\u1200-\u137F]){1,4}")


def gold_eval_roots(dataset_dir):
    """{1st-order Ethiopic root: {"val", "test"}} from the current gold val/test files."""
    roots = collections.defaultdict(set)
    for split, name in GOLD_EVAL.items():
        with open(os.path.join(dataset_dir, "data", "paradigms", name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                values = [rec.get("root") or ""] + [row.get("root") or "" for row in rec.get("table_data", [])]
                for v in values:
                    for alt in v.split("/"):
                        alt = alt.strip()
                        if not ETHIOPIC_ROOT.fullmatch(alt):
                            continue
                        roots[geez_config.first_order_root(alt)].add(split)
    return dict(roots)


def line_root(d):
    """The root of a synthetic line in 1st-order Ethiopic letters (q-t-l -> ቀ-ተ-ለ)."""
    feats = d.get("features") or d.get("analysis") or {}
    return geez_config.first_order_root(geez_config.to_ethiopic_root(feats.get("root") or ""))


def split(full_dir, dataset_dir):
    """-> (train lines, analyze lines, held-out train lines, held-out analyze lines, Counter((split, root)))."""
    roots = gold_eval_roots(dataset_dir)
    tr, an, htr, han = [], [], [], []
    moved = collections.Counter()
    with open(os.path.join(full_dir, TRAIN), encoding="utf-8") as ft, open(os.path.join(full_dir, ANALYZE), encoding="utf-8") as fa:
        for n, (lt, la) in enumerate(zip(ft, fa), 1):
            d, a = json.loads(lt), json.loads(la)
            if d.get("surface_form") != a.get("surface_form") or d.get("features") != a.get("analysis"):
                sys.exit(f"[!] {TRAIN} and {ANALYZE} differ at line {n}: not a mirror pair")
            r = line_root(d)
            if r in roots:
                htr.append(lt); han.append(la)
                moved[("+".join(sorted(roots[r])), r)] += 1
            else:
                tr.append(lt); an.append(la)
    return tr, an, htr, han, moved


def targets(dataset_dir):
    return (os.path.join(dataset_dir, "data", "paradigms", TRAIN), os.path.join(dataset_dir, "data", "paradigms", ANALYZE),
            os.path.join(dataset_dir, HELDOUT_DIR, HELDOUT_TRAIN), os.path.join(dataset_dir, HELDOUT_DIR, HELDOUT_ANALYZE))


def split_into_dataset(full_dir, dataset_dir, write=True):
    """Split the generator output in full_dir; write (or compare with) the four dataset files.
    -> (Counter((split, root)), [files that differ] when write=False)."""
    tr, an, htr, han, moved = split(full_dir, dataset_dir)
    paths = targets(dataset_dir)
    texts = ["".join(x) for x in (tr, an, htr, han)]
    if write:
        os.makedirs(os.path.dirname(paths[2]), exist_ok=True)
        for p, t in zip(paths, texts):
            with open(p, "w", encoding="utf-8") as f:
                f.write(t)
        return moved, []
    differ = [p for p, t in zip(paths, texts) if not os.path.exists(p) or open(p, encoding="utf-8").read() != t]
    return moved, differ


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full_dir", required=True, help="generator output with the full synthetic pair of files")
    ap.add_argument("--dataset_dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--report", default=None, help="write moved lines per split and root to this TSV")
    args = ap.parse_args()
    moved, differ = split_into_dataset(args.full_dir, args.dataset_dir, write=args.write)
    per_split = collections.Counter()
    for (s, r), n in moved.items():
        per_split[s] += n
    print(f"held out: {sum(moved.values())} lines, {len(moved)} roots; per split {dict(sorted(per_split.items()))}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("split\troot\tlines\n")
            for (s, r), n in sorted(moved.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
                f.write(f"{s}\t{r}\t{n}\n")
    if args.check:
        print("split == dataset files" if not differ else f"[!] differ: {differ}")
        sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
