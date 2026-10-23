#!/usr/bin/env python3
"""
Score a system on the Merho Ges gold val and test splits of a geez-dataset checkout.

Systems:
  --system byt5    the trained ByT5 model alone (default; the final model in --model_dir)
  --system hybrid  HybridAnalyzer: fast path -> pattern matcher -> ByT5 fallback
                   (add --no_neural to switch the ByT5 fallback off)

What is scored:
  - Every pair dataset_loader.load_gold_split gives for the split: the gold file as written, without
    the rows in merho_ges_gold_flagged_rows.tsv and rows with no root (the same reader the trainer
    uses for the train split). On geez-dataset 0d0ceaa: val 1,173 pairs, test 1,558.
  - Root accuracy: the predicted root must equal the gold root exactly. The prediction is first
    written in Ethiopic letters (q-t-l -> ቀ-ተ-ለ, for the rule-based analyzer); letters are not
    otherwise changed, so ቅ-ት-ል for ቀ-ተ-ለ is wrong. This is the number to read for root detection.
  - The other fields count only where the gold pair has a value:
      verbs     type, stem, tense, person, gender, number (+ object where gold has one)
      nominals  type, number (+ gender where gold has one)
    "exact" = root and every field the gold pair labels. A system that only answers verbs (the
    rule-based analyzer) predicts type "verb", so its nominal "type" and "exact" are wrong by design.
  - A predicted "jussive" is read as "subjunctive" (the gold files use subjunctive only).
  - Results are broken down by type, verb class, tense, and (hybrid) by which tier answered.
  - --split unified_val unified_test score the book rows of data/unified/geez_morphology_rows.jsonl (status empty,
    split val or test; roots and forms the model was not trained on). Their word classes include the rootless
    ones (pronoun, preposition, particle, ...: the reply must say 'root: none'), nominal, infinitive, gerund and
    verbal_noun; results are also broken down by book. possessor and negated count where the pair has them or
    the reply gives them.
  - 95% Wilson intervals are written to --out_json for the root and exact accuracy of every group.
"""

import os
import sys
import json
import argparse
import math
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import geez_script
import dataset_loader  # reads the gold files; shared with train_byt5_morphology.py
from train_byt5_morphology import DEFAULT_OUTPUT_DIR as DEFAULT_MODEL_DIR, model_to_load

FIELDS = ["root", "type", "stem", "tense", "person", "gender", "number", "object", "possessor", "negated"]
UNIFIED_SPLITS = {"unified_val": "val", "unified_test": "test"}
TENSE_ALIASES = {"jussive": "subjunctive"}
CLASS_ORDER = ["Type A", "Type B", "Type C", "hollow", "initial-weak",
               "final-weak", "quadriliteral", "unknown"]


def parse_prediction(pred_str):
    """A reply in the verb or nominal shape, or a root-only reply ('root: <root>') -> dict. A root that is a
    no-value word ('root: none') becomes None."""
    out = {}
    for p in pred_str.split("|"):
        if ":" in p:
            k, v = p.split(":", 1)
            out[k.strip()] = v.strip()
    if "root" in out:
        out["root"] = clean_label(out["root"])
    return out


def root_matches(predicted, gold):
    """Exact root match. The prediction is written in Ethiopic letters first (q-t-l -> ቀ-ተ-ለ);
    the letters themselves are compared as they are. A gold root of None (a word without root) is matched by
    a reply without root ('root: none')."""
    if gold is None:
        return not predicted
    return bool(predicted) and geez_script.to_ethiopic_root(predicted.strip()) == gold


def clean_label(v):
    if v is None:
        return None
    v = str(v).strip()
    if v.lower() in ("", "none", "null", "n/a"):
        return None
    return TENSE_ALIASES.get(v, v)


def normalize_row(d):
    """Shared label normalization for gold and predictions. type defaults to "verb" (a prediction
    or gold row without a type is a verb analysis); object and possessor are None when there is none;
    negated is "yes" or None."""
    out = {k: clean_label(d.get(k)) for k in ("type", "stem", "tense", "person", "gender", "number", "object", "possessor")}
    out["negated"] = "yes" if str(d.get("negated")).strip().lower() in ("yes", "true", "1") else None
    out["type"] = out["type"] or "verb"
    if out["person"] == "1" and out["gender"] is None:
        out["gender"] = "c"
    for k in ("object", "possessor"):
        if out[k] in ("1s", "1p"):
            out[k] = "1c" + out[k][1:]
    return out


def verb_class(rec):
    """Verb class from the gold paradigm class name, then the root shape.

    Order: Type C (ባረከ), Type B (ቀደሰ), quadriliteral (4+ radicals or a
    quadriliteral class name), then initial-weak, hollow, final-weak by
    where w/y sits in the root, else Type A.
    """
    pc = rec.get("paradigm_class") or ""
    root = geez_script.root_symbols(rec.get("root") or "")
    weak = ("w", "y")
    if "ባረከ" in pc:
        return "Type C"
    if "ቀደሰ" in pc:
        return "Type B"
    quad_names = ("ደንገፀ", "ደንገጸ", "ተመንደበ", "ጠብለለ", "ጠንበለለ", "አንቀልቀለ", "ሰንደወ")
    if len(root) >= 4 or "quadri" in pc.lower() or any(n in pc for n in quad_names):
        return "quadriliteral"
    if len(root) == 3:
        if root[0] in weak:
            return "initial-weak"
        if root[1] in weak:
            return "hollow"
        if root[2] in weak:
            return "final-weak"
        return "Type A"
    if any(n in pc for n in ("ቆመ", "ጾመ", "ሤመ")):
        return "hollow"
    if "ወገረ" in pc:
        return "initial-weak"
    if any(n in pc for n in ("ሴሰየ", "ሰተየ", "ሠረየ", "tertiae", "Tertiae")):
        return "final-weak"
    if "ቀተለ" in pc:
        return "Type A"
    return "unknown"


def book_short(name):
    """Short book name for the per-book groups."""
    name = name or ""
    for key, short in (("Mets", "Kidane Wold"), ("ርባሐ", "Rebaha"), ("ፍሬ", "Fre Geez"), ("መማሪያ", "Memariya"),
                       ("ግዕዝ ቋንቋ", "Quanqua 2"), ("Mesehaf", "Merho Ges")):
        if name.startswith(key) or key in name[:30]:
            return short
    return name[:20] or "unknown"


def wilson(c, n, z=1.96):
    """95% Wilson interval of c correct out of n, as (low, high) fractions."""
    if not n:
        return (None, None)
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def duplicate_surplus(pairs):
    """Rows beyond the first for a spelling listed with more than one analysis: a single-answer system cannot be
    exact on all of them, so exact accuracy cannot exceed 1 - surplus / pairs."""
    counts = defaultdict(int)
    for p in pairs:
        counts[p["surface"]] += 1
    return sum(c - 1 for c in counts.values())


def folded_train_roots(dataset_dir):
    """Spelling-folded roots (geez_script.fold_spelling) of the gold train pairs and of the unified book rows in
    split train. The splits are disjoint by exact root; a held-out root that folds onto one of these is a spelling
    variant of a training root (ሠ/ሰ, ጸ/ፀ, ሀ/ሐ/ኀ, labiovelar and plain, ...), and the fold is wide, so the group is an
    upper bound on real variants."""
    roots = {p["root"] for p in dataset_loader.load_gold_split(dataset_dir, "train") if p["root"]}
    roots |= {p["root"] for p in dataset_loader.load_unified(dataset_dir, "train", ("book",)) if p["root"]}
    return {geez_script.fold_spelling(r) for r in roots}


def tag_root_variants(pairs, train_folded):
    """Add 'rootvar' to each pair: 'variant_in_train' when its folded root is a folded training root."""
    for p in pairs:
        p["rootvar"] = ("variant_in_train" if p["root"] and geez_script.fold_spelling(p["root"]) in train_folded
                        else "disjoint")


def load_gold_pairs(dataset_dir, split, stats=None):
    """The scored pairs of one split (dataset_loader.load_gold_split), with normalized labels."""
    pairs = []
    if split in UNIFIED_SPLITS:
        for p in dataset_loader.load_unified(dataset_dir, UNIFIED_SPLITS[split], ("book",), stats):
            pairs.append({"split": split, "surface": p["form"], "root": p["root"], "record_id": p["id"],
                          "row": p["page"], "verb_class": None, "book": book_short(p["book"]),
                          **normalize_row({**p, "type": p["type"]})})
        return pairs
    if split == "books":
        for p in dataset_loader.load_book_forms(dataset_dir, "heldout", stats):
            pairs.append({"split": split, "surface": p["form"], "root": p["root"], "record_id": None,
                          "row": p["line"], "verb_class": None, **normalize_row(p)})
        return pairs
    for p in dataset_loader.load_gold_split(dataset_dir, split, stats):
        pairs.append({
            "split": split,
            "surface": p["form"],
            "root": p["root"],
            "record_id": p["record_id"],
            "row": p["row"],
            "verb_class": verb_class(p["record"]) if p["type"] == "verb" else None,
            **normalize_row(p),
        })
    return pairs


class ByT5System:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        chkpt = model_to_load(model_dir)
        print(f"Loading model from: {chkpt}")
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(chkpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(chkpt).to(self.device)
        self.model.eval()

    def predict(self, surface):
        inp = self.tok(dataset_loader.input_text(surface), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_length=128, num_beams=2)
        text = self.tok.decode(out[0], skip_special_tokens=True)
        pred = parse_prediction(text)
        return {"root": pred.get("root"), **normalize_row(pred), "tier": "byt5", "raw": text}   # no "type:" = verb


class HybridSystem:
    def __init__(self, dataset_dir, model_dir=None, no_neural=False):
        from analyzer import HybridAnalyzer
        # dictionary from the same dataset checkout, ByT5 fallback from the same --model_dir
        self.a = HybridAnalyzer(dataset_dir=dataset_dir, model_dir=model_dir, use_neural=not no_neural)

    def predict(self, surface):
        res = self.a.analyze(surface)
        if not res:
            return {"root": None, **normalize_row({}), "type": None, "tier": "no_answer", "raw": ""}
        best = res[0]
        spec = best.pattern_spec or ""
        tier = ("fast_path" if spec.startswith("fast_path") else
                "neural" if spec.startswith("neural") else "pattern")
        pred = {"type": best.pos, "stem": best.stem, "tense": best.tense, "person": best.person,
                "gender": best.gender, "number": best.number, "object": best.object,
                "possessor": getattr(best, "possessor", None),
                "negated": "yes" if getattr(best, "negative", False) else None}
        return {"root": best.root_string or None, **normalize_row(pred), "tier": tier,
                "raw": f"{best.root_string} {best.stem} {best.tense} {best.person}{best.gender or ''}{best.number or ''}"}


def readings_key(item):
    """The labels that make one reading of a spelling: root, class and every scored field."""
    return (item["root"], item["type"]) + tuple(item.get(f) for f in FIELDS[1:])


def homograph_groups(pairs):
    """[(surface, [pairs])] in first-seen order: the pairs of one split that share a spelling. A group whose pairs
    differ in root, class or labels is a homograph (two or more readings)."""
    groups = {}
    for p in pairs:
        groups.setdefault((p.get("split"), p["surface"]), []).append(p)
    return [(k[1], v) for k, v in groups.items()]


def is_homograph(group):
    return len({readings_key(p) for p in group}) > 1


def score_reading(item, pred):
    """{field: True/False/None} of one reply against one reading; "exact" as in score."""
    ok = {"root": root_matches(pred["root"], item["root"])}
    for f in FIELDS[1:]:
        # possessor and negated count where the pair has them or the reply gives them (a wrong extra one is a miss)
        if f in ("possessor", "negated"):
            ok[f] = (pred[f] == item[f]) if (item[f] is not None or pred[f] is not None) else None
        elif f == "gender" and item["type"] == "noun" and pred[f] is None:
            ok[f] = None    # the noun target has no gender; a reply without one is not scored on it
        else:
            ok[f] = (pred[f] == item[f]) if item[f] is not None else None
    # exact: every field this pair should carry is labeled and right (verbs need root, stem,
    # tense, person, gender, number; nominals need root and number; type, gender, object as labeled)
    if item["type"] == "verb":
        needed = ["root", "number", "stem", "tense", "person", "gender"]
    elif item["root"] is None:
        needed = ["root"]                      # a word without root: the reply must say 'root: none'
    else:
        needed = ["root"] + (["number"] if item["number"] is not None else [])
    labeled_all = all(ok[f] is not None for f in needed)
    ok["exact"] = all(v for v in ok.values() if v is not None) if labeled_all else None
    return ok


def best_reading(group, pred):
    """(item, ok) of the reading the reply matches best: exact first, then the root, then the most right fields;
    ties go to the first reading."""
    scored = [(p, score_reading(p, pred)) for p in group]
    return max(scored, key=lambda x: (bool(x[1]["exact"]), bool(x[1]["root"]),
                                      sum(1 for v in x[1].values() if v)))


def score(pairs, system, predictions=None):
    """-> (totals, examples). totals: group -> field -> [correct, scored]. The pairs of one split that share a
    spelling are one item: the system is asked once and the reply counts as right if it matches any reading
    (the best-matching reading is scored). Spellings with two or more readings also count in the "homograph" group.
    `predictions` (a list) gets one row per spelling: the matched reading, all readings, the prediction and which
    fields were right."""
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    examples = []
    for surface, group in homograph_groups(pairs):
        pred = system.predict(surface)
        item, ok = best_reading(group, pred)
        groups = ["ALL", "type:" + item["type"], "tier:" + pred["tier"]]
        groups.append("source:" + ("book" if item.get("book") else "gold"))
        if item.get("book"):
            groups.append("book:" + item["book"])
        if item.get("rootvar"):
            groups.append("rootvar:" + item["rootvar"])
        if item["type"] == "verb":
            groups += ["class:" + str(item["verb_class"] or "unknown"), "tense:" + str(item["tense"])]
        if is_homograph(group):
            groups.append("homograph")
        for g in groups:
            totals[g]["rows"][1] += 1
            for f, v in ok.items():
                if v is not None:
                    totals[g][f][1] += 1
                    totals[g][f][0] += int(v)
        if len(examples) < 15 and not ok.get("exact"):
            examples.append((item, pred))
        if predictions is not None:
            predictions.append({"gold": item, "readings": group, "predicted": pred, "correct": ok})
    return totals, examples


def no_root_report(scored):
    """Answers of 'root: none', apart from the root accuracy. `scored` is the list `score` fills
    (gold and predicted per pair). -> dict with
      gold_none: pairs whose gold root is none, and how many the reply called none (recall, Wilson 95%);
      false_none: pairs with a real gold root that the reply called none (count and rate);
      precision: of all 'none' replies, how many were right."""
    gold_none = [r for r in scored if r["gold"]["root"] is None]
    gold_real = [r for r in scored if r["gold"]["root"] is not None]
    said_none = lambda r: not r["predicted"].get("root")
    hit = sum(said_none(r) for r in gold_none)
    false_none = sum(said_none(r) for r in gold_real)
    replies = hit + false_none
    return {"gold_none": len(gold_none), "predicted_none": hit,
            "recall": hit / len(gold_none) if gold_none else None, "recall_wilson95": wilson(hit, len(gold_none)),
            "gold_real": len(gold_real), "false_none": false_none,
            "false_none_rate": false_none / len(gold_real) if gold_real else None,
            "none_replies": replies, "precision": hit / replies if replies else None,
            "precision_wilson95": wilson(hit, replies)}


def print_no_root_report(rep):
    def ci(lo_hi):
        lo, hi = lo_hi
        return f"  95% CI {100 * lo:.1f}-{100 * hi:.1f}" if lo is not None else ""
    print("\nNo-root answers ('root: none'):")
    print(f"   recall    {pct(rep['predicted_none'], rep['gold_none'])}{ci(rep['recall_wilson95'])}"
          "   gold root none, reply none")
    print(f"   false     {pct(rep['false_none'], rep['gold_real'])}   gold root real, reply none")
    print(f"   precision {pct(rep['predicted_none'], rep['none_replies'])}{ci(rep['precision_wilson95'])}"
          "   'none' replies that were right")


def pct(c, n):
    return f"{100 * c / n:5.1f}% ({c}/{n})" if n else "   n/a"


def main():
    ap = argparse.ArgumentParser(description="Score a system on the Merho Ges gold val and test splits")
    ap.add_argument("--system", choices=["byt5", "hybrid"], default="byt5")
    ap.add_argument("--no_neural", action="store_true", help="hybrid only: no ByT5 fallback")
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                    help="The trainer's --output_dir (default: byt5_morphology_model/), or one checkpoint-<step> folder")
    ap.add_argument("--dataset_dir", default=dataset_loader.default_dataset_dir(),
                    help="Checkout of the geez-dataset repo the model was trained on "
                         "(default: $GEEZ_DATASET_DIR, else ../geez-dataset)")
    ap.add_argument("--split", nargs="+", choices=["val", "test", "books", "unified_val", "unified_test"], default=["val", "test"],
                    help="Gold splits to score (default: val test); books = data/lexicon/book_forms.jsonl rows of the "
                         "other books whose root or form is a gold val/test one (never trained on); "
                         "unified_val / unified_test = the book rows of data/unified/geez_morphology_rows.jsonl in "
                         "that split (status empty)")
    ap.add_argument("--dev_unified", action="store_true",
                    help="With --split val: score the dev set, the gold val pairs plus the unified book val rows, as one "
                         "split named val (the Kaggle launcher selects checkpoints on it); the groups source:gold and "
                         "source:book keep them apart")
    ap.add_argument("--limit", type=int, default=0, help="Random sample of N pairs per split (0 = all, the default)")
    ap.add_argument("--out_json", default=None,
                    help="Write the results to this JSON file, e.g. gold_eval_byt5.json")
    ap.add_argument("--predictions_out", default=None,
                    help="Write one line per scored pair (gold, prediction, correct fields) to this JSONL file, "
                         "e.g. gold_eval_byt5_predictions.jsonl")
    args = ap.parse_args()
    args.dataset_dir = os.path.abspath(args.dataset_dir)

    # The system first: the hybrid analyzer stops with a clear error when the dataset has no dictionary.
    system = ByT5System(args.model_dir) if args.system == "byt5" else HybridSystem(args.dataset_dir, args.model_dir, args.no_neural)
    for split in args.split:
        path = (os.path.join(args.dataset_dir, dataset_loader.BOOK_FORMS_FILE) if split == "books"
                else os.path.join(args.dataset_dir, dataset_loader.UNIFIED_FILE) if split in UNIFIED_SPLITS
                else dataset_loader.gold_file_path(args.dataset_dir, split))
        if not os.path.exists(path):
            sys.exit(f"[!] Gold file not found: {path}")
    name = args.system + (" (no neural)" if args.no_neural else "")
    cols = ["root", "type", "stem", "tense", "person", "gender", "number", "object", "possessor", "negated", "exact"]
    results, predictions = {}, ([] if args.predictions_out else None)
    no_root = {}

    train_folded = folded_train_roots(args.dataset_dir) if os.path.exists(dataset_loader.gold_file_path(args.dataset_dir, "train")) else None
    for split in args.split:
        stats = defaultdict(int)
        pairs = load_gold_pairs(args.dataset_dir, split, stats)
        if args.dev_unified and split == "val":
            extra = load_gold_pairs(args.dataset_dir, "unified_val", stats)
            for p in extra:
                p["split"] = "val"
            pairs += extra
            print(f"--dev_unified: {len(extra)} unified book val rows added to the {len(pairs) - len(extra)} gold val pairs")
        where = (dataset_loader.BOOK_FORMS_FILE if split == "books" else dataset_loader.UNIFIED_FILE if split in UNIFIED_SPLITS
                 else dataset_loader.gold_file_path(args.dataset_dir, split))
        print(f"\n{split}: {where}: {len(pairs)} pairs "
              f"({', '.join(f'{k} {v}' for k, v in sorted(stats.items()))})")
        if args.limit and args.limit < len(pairs):
            import random
            pairs = random.Random(42).sample(pairs, args.limit)
            print(f"[!] --limit {args.limit}: scoring a random sample, not the full split")
        if train_folded is not None:
            tag_root_variants(pairs, train_folded)
        surplus = duplicate_surplus(pairs)
        if surplus:
            homs = sum(is_homograph(g) for _, g in homograph_groups(pairs))
            print(f"{len(pairs)} pairs, {len(pairs) - surplus} spellings: a spelling is scored once and a reply that "
                  f"matches any of its readings is right ({homs} spelling(s) with two or more readings)")
        scored = []
        totals, examples = score(pairs, system, scored)
        if predictions is not None:
            predictions += scored
        no_root[split] = no_root_report(scored)
        totals["ALL"]["duplicate_surplus"] = [surplus, len(pairs)]
        results[split] = {g: dict(t) for g, t in totals.items()}

        print("=" * 72)
        print(f"GOLD {split.upper()} RESULTS: {name}   (accuracy over pairs where gold has the field)")
        print("=" * 72)

        def show(group_names):
            for g in group_names:
                if g not in totals:
                    continue
                t = totals[g]
                print(f"\n[{g}]  pairs={t['rows'][1]}")
                for c in cols:
                    if c in t:
                        print(f"   {c:8s} {pct(*t[c])}")

        show(["ALL"])
        show(["type:" + t for t in dataset_loader.TYPES + dataset_loader.UNIFIED_TYPES])
        show(sorted(g for g in totals if g.startswith("source:")))
        show(["homograph"])
        show(sorted(g for g in totals if g.startswith("book:")))
        show(sorted(g for g in totals if g.startswith("rootvar:")))
        show(["class:" + c for c in CLASS_ORDER])
        show(sorted(g for g in totals if g.startswith("tense:")))
        show(sorted(g for g in totals if g.startswith("tier:")))

        print_no_root_report(no_root[split])

        print("\nSample misses:")
        for item, pred in examples:
            print(f"  {item['surface']}  gold: {item['root']} {item['type']} {item['stem']} {item['tense']} "
                  f"{item['person'] or ''}{item['gender'] or ''}{item['number'] or ''} "
                  f"[{item['verb_class'] or item['type']}]  -> {pred['raw']}")

    print("\nRoot accuracy (exact match):")
    for split, totals in results.items():
        lo, hi = wilson(*totals['ALL']['root'])
        print(f"  {split:12s} {pct(*totals['ALL']['root'])}" + (f"  95% CI {100 * lo:.1f}-{100 * hi:.1f}" if lo is not None else ""))

    if args.out_json:
        summary = {split: {"pairs": t["ALL"]["rows"][1], "root_correct": t["ALL"]["root"][0],
                           "root_accuracy": t["ALL"]["root"][0] / max(1, t["ALL"]["root"][1]),
                           "root_wilson95": wilson(*t["ALL"]["root"]),
                           "exact_correct": t["ALL"].get("exact", [0, 0])[0],
                           "exact_scored": t["ALL"].get("exact", [0, 0])[1],
                           "exact_accuracy": t["ALL"].get("exact", [0, 0])[0] / max(1, t["ALL"].get("exact", [0, 0])[1]),
                           "exact_wilson95": wilson(*t["ALL"].get("exact", [0, 0]))}
                   for split, t in results.items()}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"system": name, "model_dir": args.model_dir if args.system == "byt5" else None,
                       "dataset_dir": os.path.abspath(args.dataset_dir), "root_exact_match": summary,
                       "no_root": no_root,
                       "by_group": results}, f, ensure_ascii=False, indent=1)
    if args.predictions_out:
        with open(args.predictions_out, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
