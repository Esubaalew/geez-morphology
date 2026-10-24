#!/usr/bin/env python3
"""Build the text of the span-corruption stage (train_tpu.py --stage dapt) from every corpus of the dataset.

    python3 build_dapt_corpus.py --dataset_dir D --out dapt_corpus.txt [--report dapt_report.json]

Sources: data/corpora/*.txt (books as text), the Ge'ez fields (headword_geez,
geez_explanation, geez_phrase, citations, forms) of data/corpora/*.jsonl, data/parallel/*.jsonl,
and the two dictionaries. Amharic glosses and definitions are not read.
Lines are split on newlines and on the sentence mark ።, whitespace-normalized, and a line is kept once however many
files hold it (the literature corpus repeats the book texts, the two true_parallel_pairs files overlap).
A line is dropped when one of its words (split on spaces and ፡) is a held-out form (training_mixture.HeldOut:
gold val/test forms and the unified book val/test forms), so the stage never reads the context of a held-out word.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training_mixture  # noqa: E402

GEEZ_FIELDS = ("headword_geez", "geez_explanation", "geez_phrase", "citations", "forms")
ETHIOPIC = re.compile(r"[\u1200-\u137F]")
MIN_ETHIOPIC = 8


def text_files(d):
    return sorted(glob.glob(os.path.join(d, "data", "corpora", "*.txt")))


def jsonl_files(d):
    return (sorted(glob.glob(os.path.join(d, "data", "corpora", "*.jsonl"))) +
            sorted(glob.glob(os.path.join(d, "data", "parallel", "*.jsonl"))) +
            [os.path.join(d, "data", "lexicon", "classical_dictionary_kidane_wold.jsonl"),
             os.path.join(d, "data", "lexicon", "geez_amharic_dictionary.jsonl")])


def pieces(text):
    for line in text.split("\n"):
        for part in line.split("።"):
            part = " ".join(part.split())
            if len(ETHIOPIC.findall(part)) >= MIN_ETHIOPIC:
                yield part


def words(line):
    return {w.strip("()[]፣፤፥፦,.!?\"'") for w in re.split(r"[\s፡]+", line) if w}


def build(dataset_dir, out_path):
    held = training_mixture.HeldOut(dataset_dir)
    seen, kept = set(), []
    stats = collections.Counter()

    def add(text, src):
        for p in pieces(text):
            stats[f"{src}: pieces"] += 1
            if p in seen:
                stats[f"{src}: duplicate"] += 1
                continue
            seen.add(p)
            if words(p) & held.forms:
                stats[f"{src}: dropped, holds a held-out form"] += 1
                continue
            kept.append(p)
            stats[f"{src}: kept"] += 1
            stats[f"{src}: kept bytes"] += len(p.encode())

    for path in text_files(dataset_dir):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                add(f.read(), os.path.relpath(path, dataset_dir))
    for path in jsonl_files(dataset_dir):
        if not os.path.exists(path):
            continue
        src = os.path.relpath(path, dataset_dir)
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                for k in GEEZ_FIELDS:
                    v = d.get(k)
                    if isinstance(v, list):
                        v = " ፡ ".join(x for x in v if isinstance(x, str))
                    if isinstance(v, str):
                        add(v, src)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    report = {"lines": len(kept), "bytes": sum(len(k.encode()) for k in kept), "by_source": dict(sorted(stats.items()))}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    r = build(a.dataset_dir, a.out)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
    print(f"{r['lines']} lines, {r['bytes']} bytes -> {a.out}")


if __name__ == "__main__":
    main()
