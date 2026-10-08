#!/usr/bin/env python3
"""Hold out the synthetic lines whose root is a gold val or test root.

The generator (generate_synthetic_data.py) writes the full synthetic pair of files. This one split step
moves every line whose root is a root of the current gold val or test file out of the training file into
a separate held-out file in data/synthetic/; nothing is deleted, the lines move, both files keep the
generator's order. The paired analyze file is split on the same lines.

  data/paradigms/synthetic_morphology_train.jsonl        synthetic TRAIN (the trainer reads only this)
  data/paradigms/synthetic_morphology_analyze.jsonl      its analyze mirror
  data/synthetic/synthetic_forms_for_val_test_roots.jsonl            held out: lines with a val/test root
  data/synthetic/synthetic_forms_for_val_test_roots_analyze.jsonl    their analyze mirror

Gold val/test roots: for every row of merho_ges_gold_val.jsonl and merho_ges_gold_test.jsonl, the row's
root and the record's root, each "/" alternative, in the 1st order (geez_script.first_order_root); only
roots of Ethiopic letters. A synthetic line is held out when its root, written in Ethiopic letters of
the 1st order, is one of them.

Near-duplicates of gold val/test forms: a line that would stay in the training file is dropped (with its
analyze line) when its surface form, folded with geez_script.fold_spelling (labiovelar letters made plain,
twin letters merged, vowel orders kept), equals a folded gold val/test form. The fold only decides which
lines to drop; roots are not merged or renamed, the held-out files and the gold files are not filtered.
The other lines keep their bytes and order. The dropped lines are counted, and written on request to a
report (--near_duplicate_report).

Usage:
  python split_heldout_synthetic.py --full_dir <generator output dir> --dataset_dir <dataset repo> [--write | --check]
    --write   write the four files into the dataset repo
    --check   compare what the split gives with the four files in the dataset repo, count the lines of the
              four files whose surface form is a gold val or test form, and count the lines of the training
              pair whose folded surface form is a folded gold val/test form (exit 1 if the files differ or
              either count is not 0)
    --near_duplicate_report <path>   write the dropped near-duplicate lines to this JSONL file

Gold val/test forms: every form of every row of merho_ges_gold_val.jsonl and merho_ges_gold_test.jsonl, rows
with no root included; a cell gives its "/"-spellings with "(...)" notes dropped, as scripts/check_dataset.py
reads it. The generator leaves these forms out (data/heldout_gold_forms.tsv), so the count must be 0.
  The generator calls split_into_dataset() itself when it mirrors into the dataset (generate_synthetic_data.py).
"""
import argparse
import collections
import json
import os
import re
import sys

import geez_script

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
                        roots[geez_script.first_order_root(alt)].add(split)
    return dict(roots)


def form_variants(cell):
    """The "/"-spellings of a geez_form cell: "(...)" notes dropped, edge punctuation stripped
    (the same rule as the dataset's scripts/check_dataset.py)."""
    out = []
    for v in re.sub(r"\([^)]*\)", " ", cell or "").split("/"):
        v = v.strip(" :፡።()[]፣,")
        if v:
            out.append(v)
    return out


def gold_eval_forms(dataset_dir):
    """Every form of every row of the current gold val/test files (rows with no root included)."""
    forms = set()
    for name in GOLD_EVAL.values():
        with open(os.path.join(dataset_dir, "data", "paradigms", name), encoding="utf-8") as f:
            for line in f:
                for row in json.loads(line).get("table_data", []):
                    forms.update(form_variants(row.get("geez_form") or ""))
    return forms


def gold_eval_forms_folded(dataset_dir):
    """{folded form: [(gold form, split, record_id)]} over every form of the current gold val/test files."""
    folded = collections.defaultdict(list)
    for split_name, name in GOLD_EVAL.items():
        with open(os.path.join(dataset_dir, "data", "paradigms", name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for row in rec.get("table_data", []):
                    for form in form_variants(row.get("geez_form") or ""):
                        entry = (form, split_name, rec.get("record_id"))
                        key = geez_script.fold_spelling(form)
                        if entry not in folded[key]:
                            folded[key].append(entry)
    return dict(folded)


def drop_near_duplicates(lines, folded_gold):
    """Split JSON lines into (kept lines, dropped lines): a line is dropped when its folded surface form is a key
    of folded_gold. Kept lines keep their order and bytes."""
    kept, dropped = [], []
    for line in lines:
        form = json.loads(line).get("surface_form") or ""
        (dropped if geez_script.fold_spelling(form) in folded_gold else kept).append(line)
    return kept, dropped


def near_duplicate_lines(dataset_dir):
    """{dataset training file: [(line number, root, surface form)]} for lines of the training pair whose folded
    surface form is a folded gold val/test form."""
    folded_gold = gold_eval_forms_folded(dataset_dir)
    found = {}
    for p in targets(dataset_dir)[:2]:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                d = json.loads(line)
                if geez_script.fold_spelling(d.get("surface_form") or "") in folded_gold:
                    feats = d.get("features") or d.get("analysis") or {}
                    found.setdefault(os.path.relpath(p, dataset_dir), []).append((n, feats.get("root"), d["surface_form"]))
    return found


def write_near_duplicate_report(path, dropped, folded_gold):
    """One JSON object per dropped training line: root, surface form, folded form, the gold val/test forms it
    matches, and the line itself."""
    with open(path, "w", encoding="utf-8") as f:
        for line in dropped:
            d = json.loads(line)
            folded = geez_script.fold_spelling(d.get("surface_form") or "")
            f.write(json.dumps({
                "root": (d.get("features") or {}).get("root"),
                "surface_form": d.get("surface_form"),
                "folded_form": folded,
                "gold_val_test_forms": [{"form": g, "split": s, "record_id": r} for g, s, r in folded_gold[folded]],
                "line": d,
            }, ensure_ascii=False) + "\n")


def gold_form_lines(dataset_dir):
    """{dataset synthetic file: [(line number, root, surface form)]} for lines whose surface form is a gold val/test form."""
    forms = gold_eval_forms(dataset_dir)
    found = {}
    for p in targets(dataset_dir):
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                d = json.loads(line)
                if d.get("surface_form") in forms:
                    feats = d.get("features") or d.get("analysis") or {}
                    found.setdefault(os.path.relpath(p, dataset_dir), []).append((n, feats.get("root"), d["surface_form"]))
    return found


def line_root(d):
    """The root of a synthetic line in 1st-order Ethiopic letters (q-t-l -> ቀ-ተ-ለ)."""
    feats = d.get("features") or d.get("analysis") or {}
    return geez_script.first_order_root(geez_script.to_ethiopic_root(feats.get("root") or ""))


def split(full_dir, dataset_dir):
    """-> (train lines, analyze lines, held-out train lines, held-out analyze lines, Counter((split, root)),
    dropped near-duplicate train lines). Near-duplicates are dropped from the training pair only."""
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
    folded_gold = gold_eval_forms_folded(dataset_dir)
    kept_tr, dropped = drop_near_duplicates(tr, folded_gold)
    if dropped:
        kept_an, _ = drop_near_duplicates(an, folded_gold)
        tr, an = kept_tr, kept_an
    return tr, an, htr, han, moved, dropped


def targets(dataset_dir):
    return (os.path.join(dataset_dir, "data", "paradigms", TRAIN), os.path.join(dataset_dir, "data", "paradigms", ANALYZE),
            os.path.join(dataset_dir, HELDOUT_DIR, HELDOUT_TRAIN), os.path.join(dataset_dir, HELDOUT_DIR, HELDOUT_ANALYZE))


def split_into_dataset(full_dir, dataset_dir, write=True, near_duplicate_report=None):
    """Split the generator output in full_dir; write (or compare with) the four dataset files.
    -> (Counter((split, root)), [files that differ] when write=False, dropped near-duplicate train lines).
    near_duplicate_report: path of a JSONL report of the dropped lines (write_near_duplicate_report)."""
    tr, an, htr, han, moved, dropped = split(full_dir, dataset_dir)
    if near_duplicate_report:
        write_near_duplicate_report(near_duplicate_report, dropped, gold_eval_forms_folded(dataset_dir))
    paths = targets(dataset_dir)
    texts = ["".join(x) for x in (tr, an, htr, han)]
    if write:
        os.makedirs(os.path.dirname(paths[2]), exist_ok=True)
        for p, t in zip(paths, texts):
            with open(p, "w", encoding="utf-8") as f:
                f.write(t)
        return moved, [], dropped
    differ = [p for p, t in zip(paths, texts) if not os.path.exists(p) or open(p, encoding="utf-8").read() != t]
    return moved, differ, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full_dir", required=True, help="generator output with the full synthetic pair of files")
    ap.add_argument("--dataset_dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--report", default=None, help="write moved lines per split and root to this TSV")
    ap.add_argument("--near_duplicate_report", default=None,
                    help="write the training lines dropped as near-duplicates of gold val/test forms to this JSONL file")
    args = ap.parse_args()
    moved, differ, dropped = split_into_dataset(args.full_dir, args.dataset_dir, write=args.write,
                                                near_duplicate_report=args.near_duplicate_report)
    per_split = collections.Counter()
    for (s, r), n in moved.items():
        per_split[s] += n
    print(f"held out: {sum(moved.values())} lines, {len(moved)} roots; per split {dict(sorted(per_split.items()))}")
    print(f"near-duplicates of gold val/test forms dropped from the training pair: {len(dropped)} lines, "
          f"{len({line_root(json.loads(x)) for x in dropped})} roots")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("split\troot\tlines\n")
            for (s, r), n in sorted(moved.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
                f.write(f"{s}\t{r}\t{n}\n")
    found = gold_form_lines(args.dataset_dir)
    n_found = sum(len(v) for v in found.values())
    print(f"lines whose surface form is a gold val/test form: {n_found}"
          + "".join(f"\n[!] {p}: {len(v)} lines, e.g. line {v[0][0]} root {v[0][1]}" for p, v in sorted(found.items())))
    near = near_duplicate_lines(args.dataset_dir)
    n_near = sum(len(v) for v in near.values())
    print(f"training lines whose folded surface form is a folded gold val/test form: {n_near}"
          + "".join(f"\n[!] {p}: {len(v)} lines, e.g. line {v[0][0]} root {v[0][1]}" for p, v in sorted(near.items())))
    if args.check:
        print("split == dataset files" if not differ else f"[!] differ: {differ}")
        sys.exit(1 if differ or n_found or n_near else 0)


if __name__ == "__main__":
    main()
