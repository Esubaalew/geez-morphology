#!/usr/bin/env python3
"""Summarize trained runs: accuracy per held-out set over seeds, and a paired comparison of two arms.

    python3 scripts/summarize_runs.py RUN_DIR [RUN_DIR ...] [--compare ARM_A_DIRS -- ARM_B_DIRS]

A RUN_DIR holds what the Kaggle run writes: scores_gold_details.json and scores_unified_details.json (evaluate_gold.py
--out_json) and predictions_gold.jsonl / predictions_unified.jsonl (--predictions_out).

Per set (val, test, unified_val, unified_test) it prints, over the runs given: root accuracy and exact accuracy of every
run, their mean and sample standard deviation, and the 95% Wilson interval of the pooled correct/total (an interval
over pairs, not over seeds; with 3 seeds the standard deviation over runs is the figure to quote for seed noise).

--compare takes two lists of run folders, separated by "--" (arm A first). For each set it pairs the predictions of A
and B pair by pair (same seed order is not needed: a pair counts as right for an arm when the arm's runs are right on
it more often than not), and reports the exact two-sided McNemar test on the pairs where exactly one arm is right.
"""
import argparse
import json
import math
import os
import statistics
import sys

SETS = (("gold", "val"), ("gold", "test"), ("unified", "unified_val"), ("unified", "unified_test"))


def wilson(c, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def load_scores(run_dir):
    out = {}
    for group in ("gold", "unified"):
        path = os.path.join(run_dir, f"scores_{group}_details.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        for split, t in r["by_group"].items():
            a = t["ALL"]
            out[split] = {"root": tuple(a["root"]), "exact": tuple(a.get("exact", (0, 0)))}
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: b pairs only A right, c pairs only B right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pair_key(g):
    return (g["split"], g["surface"], g.get("record_id"), g.get("row"), g["root"], g["type"], g["stem"], g["tense"],
            g["person"], g["gender"], g["number"], g["object"])


def load_correct(run_dirs, split):
    """{pair key: share of the runs that are exact-right on it} for one split, None where exact is not scored."""
    votes = {}
    for d in run_dirs:
        for group in ("gold", "unified"):
            path = os.path.join(d, f"predictions_{group}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["gold"]["split"] != split:
                        continue
                    ex = r["correct"].get("exact")
                    if ex is None:
                        continue
                    votes.setdefault(pair_key(r["gold"]), []).append(bool(ex))
    return {k: sum(v) / len(v) for k, v in votes.items()}


def main():
    argv = sys.argv[1:]
    compare = None
    if "--compare" in argv:
        i = argv.index("--compare")
        rest = argv[i + 1:]
        argv = argv[:i]
        if "--" not in rest:
            sys.exit("--compare needs ARM_A_DIRS -- ARM_B_DIRS")
        j = rest.index("--")
        compare = (rest[:j], rest[j + 1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    args = ap.parse_args(argv)
    def summarize(label, runs):
        scores = {d: load_scores(d) for d in runs}
        print(f"{label}: {len(runs)} run(s)")
        for _, split in SETS:
            have = [(d, scores[d][split]) for d in runs if split in scores[d]]
            if not have:
                continue
            print(f"\n{split}: {have[0][1]['root'][1]} pairs")
            for key in ("root", "exact"):
                vals = [c / n for _, s in have for c, n in [s[key]] if n]
                tc = sum(s[key][0] for _, s in have)
                tn = sum(s[key][1] for _, s in have)
                lo, hi = wilson(tc, tn)
                sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
                print(f"  {key:5s} per run: " + ", ".join(f"{100 * v:.1f}" for v in vals) +
                      f"   mean {100 * statistics.mean(vals):.1f}  sd {100 * sd:.2f}  pooled 95% CI {100 * lo:.1f}-{100 * hi:.1f}")
        print()

    if compare:
        summarize("arm A", compare[0])
        summarize("arm B", compare[1])
    elif args.runs:
        summarize("runs", args.runs)
    else:
        sys.exit("no run folders given")
    if compare:
        a_dirs, b_dirs = compare
        print("\npaired comparison, exact accuracy (A = first list, B = second list)")
        for _, split in SETS:
            a, b = load_correct(a_dirs, split), load_correct(b_dirs, split)
            keys = sorted(set(a) & set(b), key=str)
            if not keys:
                continue
            only_a = sum(1 for k in keys if a[k] > 0.5 >= b[k])
            only_b = sum(1 for k in keys if b[k] > 0.5 >= a[k])
            acc_a = sum(a[k] > 0.5 for k in keys) / len(keys)
            acc_b = sum(b[k] > 0.5 for k in keys) / len(keys)
            print(f"  {split:13s} {len(keys)} pairs  A {100 * acc_a:.1f}%  B {100 * acc_b:.1f}%  only A right {only_a}  "
                  f"only B right {only_b}  McNemar p = {mcnemar_exact(only_a, only_b):.4f}")


if __name__ == "__main__":
    main()
