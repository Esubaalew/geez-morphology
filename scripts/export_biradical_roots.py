#!/usr/bin/env python3
"""Write the dictionary entries whose root has two radicals, with a proposed class and 3-radical root, for a check
by a Ge'ez reader. These entries are not trained on unless the trainer gets --biradical_dictionary.

    python3 scripts/export_biradical_roots.py --dataset_dir D --out biradical_roots.tsv

Proposed class, from the headword's letters (a guess to be checked):
  hollow-ወ        radical 1 in the 2nd or 7th order (ቆመ, ቁም): ቀ-መ -> ቀ-ወ-መ
  hollow-የ        radical 1 in the 3rd or 5th order (ሤመ, ሢም): ሠ-መ -> ሠ-የ-መ
  final-weak     the headword ends in ይ/ዩ/ያ/ዪ (-> final የ) or ው/ዉ/ዋ/ዊ, or radical 2 in the 2nd order at the end (-> final ወ)
  initial-weak   the headword starts with አ/እ or ወ that is not radical 1: አ-በ-የ for እበይ
  true-biradical none of these (no mapping proposed)
"""
import argparse
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dataset_loader  # noqa: E402
import geez_script  # noqa: E402
import verb_classes  # noqa: E402
from verb_class_paradigms import syl  # noqa: E402


def order(ch):
    return (ord(ch) - 0x1200) % 8 + 1


def first(ch):
    return syl(ch, 1)


def propose(headword, root):
    r = root.split("-")
    letters = [c for c in headword if 0x1200 <= ord(c) <= 0x137F]
    pos = next((i for i, c in enumerate(letters) if first(c) == first(r[0])), None)
    o1 = order(letters[pos]) if pos is not None else 0
    tail = letters[-1] if letters else ""
    if first(tail) == "የ" and first(r[-1]) != "የ":
        return "final-weak", f"{r[0]}-{r[1]}-የ"
    if first(tail) == "ወ" and first(r[-1]) != "ወ":
        return "final-weak", f"{r[0]}-{r[1]}-ወ"
    if letters and first(letters[0]) in ("አ", "ወ") and first(letters[0]) != first(r[0]):
        return "initial-weak", f"{first(letters[0])}-{r[0]}-{r[1]}"
    if o1 in (2, 7):
        return "hollow-ወ", f"{r[0]}-ወ-{r[1]}"
    if o1 in (3, 5):
        return "hollow-የ", f"{r[0]}-የ-{r[1]}"
    return "true-biradical", ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", default=dataset_loader.default_dataset_dir())
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fixes = verb_classes.root_fixes(args.dataset_dir)
    counts = collections.Counter()
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "headword", "root_as_printed", "pos", "proposed_class", "proposed_root", "verdict"])
        for e in dataset_loader.read_dictionary_entries(args.dataset_dir):
            value = dataset_loader.dictionary_entry_root(e, fixes)
            if not value:
                continue
            root = geez_script.first_order_root(geez_script.to_ethiopic_root(value))
            if len(root.split("-")) != 2:
                continue
            head = e.get("headword_geez") or ""
            cls, mapped = propose(head, root)
            counts[cls] += 1
            w.writerow([e["id"], head, e.get("root") or value, e.get("pos") or "", cls, mapped, ""])
    print(dict(counts), "->", args.out)


if __name__ == "__main__":
    main()
