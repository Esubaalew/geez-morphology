#!/usr/bin/env python3
"""
evaluate_gold.py
Score a system on the Merho Ges gold val and test splits of a geez-dataset checkout.

Systems:
  --system byt5    the trained ByT5 model alone (default; the final model in --model_dir)
  --system hybrid  HybridAnalyzer: fast path -> pattern matcher -> ByT5 fallback
                   (add --no_neural to switch the ByT5 fallback off)

What is scored:
  - Every pair dataset_loader.load_gold_split gives for the split: the gold file as written, without
    the rows in merho_ges_gold_flagged_rows.tsv and rows with no root (the same reader the trainer
    uses for the train split). On geez-dataset 6d856e6: val 1165 pairs, test 1553.
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
"""

import os
import re
import sys
import json
import argparse
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import geez_config
import dataset_loader  # reads the gold files; shared with train_byt5_morphology.py
from train_byt5_morphology import DEFAULT_OUTPUT_DIR as DEFAULT_MODEL_DIR

FIELDS = ["root", "type", "stem", "tense", "person", "gender", "number", "object"]
TENSE_ALIASES = {"jussive": "subjunctive"}
CLASS_ORDER = ["Type A", "Type B", "Type C", "hollow", "initial-weak",
               "final-weak", "quadriliteral", "unknown"]


def parse_prediction(pred_str):
    """'root: ቀ-ተ-ለ | tense: past | ...' -> dict."""
    out = {}
    for p in pred_str.split("|"):
        if ":" in p:
            k, v = p.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def root_matches(predicted, gold):
    """Exact root match. The prediction is written in Ethiopic letters first (q-t-l -> ቀ-ተ-ለ);
    the letters themselves are compared as they are."""
    return bool(predicted) and geez_config.to_ethiopic_root(predicted.strip()) == gold


def clean_label(v):
    if v is None:
        return None
    v = str(v).strip()
    if v.lower() in ("", "none", "null", "n/a"):
        return None
    return TENSE_ALIASES.get(v, v)


def normalize_row(d):
    """Shared label normalization for gold and predictions. type defaults to "verb" (a prediction
    or gold row without a type is a verb analysis); object is None when there is none."""
    d = {k: clean_label(d.get(k)) for k in ("type", "stem", "tense", "person", "gender", "number", "object")}
    d["type"] = d["type"] or "verb"
    if d["person"] == "1" and d["gender"] is None:
        d["gender"] = "c"
    return d


def verb_class(rec):
    """Verb class from the gold paradigm class name, then the root shape.

    Order: Type C (ባረከ), Type B (ቀደሰ), quadriliteral (4+ radicals or a
    quadriliteral class name), then initial-weak, hollow, final-weak by
    where w/y sits in the root, else Type A.
    """
    pc = rec.get("paradigm_class") or ""
    root = geez_config.root_symbols(rec.get("root") or "")
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


def load_gold_pairs(dataset_dir, split, stats=None):
    """The scored pairs of one split (dataset_loader.load_gold_split), with normalized labels."""
    pairs = []
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


def model_to_load(model_dir):
    """The final model at the top of model_dir (written when training finished), else its latest
    checkpoint-<step> folder (a run that stopped early), else model_dir itself."""
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return model_dir
    subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir)
               if re.fullmatch(r"checkpoint-\d+", d)] if os.path.isdir(model_dir) else []
    if subdirs:
        return sorted(subdirs, key=lambda x: int(x.rsplit("-", 1)[-1]))[-1]
    return model_dir


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
        inp = self.tok(f"analyze: {surface}", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_length=128, num_beams=2)
        text = self.tok.decode(out[0], skip_special_tokens=True)
        pred = parse_prediction(text)
        return {"root": pred.get("root"), **normalize_row(pred), "tier": "byt5", "raw": text}   # no "type:" = verb


class HybridSystem:
    def __init__(self, dataset_dir, no_neural=False):
        from analyzer import HybridAnalyzer
        self.a = HybridAnalyzer(dataset_dir=dataset_dir)   # its dictionary comes from the same checkout
        if no_neural:
            self.a.checkpoint_dir = None

    def predict(self, surface):
        res = self.a.analyze(surface)
        if not res:
            return {"root": None, **normalize_row({}), "type": None, "tier": "no_answer", "raw": ""}
        best = res[0]
        spec = best.pattern_spec or ""
        tier = ("fast_path" if spec.startswith("fast_path") else
                "neural" if spec.startswith("neural") else "pattern")
        pred = {"type": best.pos, "stem": best.stem, "tense": best.tense, "person": best.person,
                "gender": best.gender, "number": best.number, "object": best.object}
        return {"root": best.root_string, **normalize_row(pred), "tier": tier,
                "raw": f"{best.root_string} {best.stem} {best.tense} {best.person}{best.gender or ''}{best.number or ''}"}


def score(pairs, system, predictions=None):
    """-> (totals, examples). totals: group -> field -> [correct, scored]. `predictions` (a list)
    gets one row per pair: the gold labels, the prediction and which fields were right."""
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    examples = []
    for item in pairs:
        pred = system.predict(item["surface"])
        ok = {"root": root_matches(pred["root"], item["root"])}
        for f in FIELDS[1:]:
            ok[f] = (pred[f] == item[f]) if item[f] is not None else None
        # exact: every field this pair should carry is labeled and right (verbs need root, stem,
        # tense, person, gender, number; nominals need root and number; type, gender, object as labeled)
        needed = ["root", "number"] + (["stem", "tense", "person", "gender"] if item["type"] == "verb" else [])
        labeled_all = all(ok[f] is not None for f in needed)
        ok["exact"] = all(v for v in ok.values() if v is not None) if labeled_all else None
        groups = ["ALL", "type:" + item["type"], "tier:" + pred["tier"]]
        if item["type"] == "verb":
            groups += ["class:" + item["verb_class"], "tense:" + str(item["tense"])]
        for g in groups:
            totals[g]["rows"][1] += 1
            for f, v in ok.items():
                if v is not None:
                    totals[g][f][1] += 1
                    totals[g][f][0] += int(v)
        if len(examples) < 15 and not ok.get("exact"):
            examples.append((item, pred))
        if predictions is not None:
            predictions.append({"gold": item, "predicted": pred, "correct": ok})
    return totals, examples


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
    ap.add_argument("--split", nargs="+", choices=["val", "test"], default=["val", "test"],
                    help="Gold splits to score (default: val test)")
    ap.add_argument("--limit", type=int, default=0, help="Random sample of N pairs per split (0 = all, the default)")
    ap.add_argument("--out_json", default=None,
                    help="Write the results to this JSON file, e.g. gold_eval_byt5.json")
    ap.add_argument("--predictions_out", default=None,
                    help="Write one line per scored pair (gold, prediction, correct fields) to this JSONL file, "
                         "e.g. gold_eval_byt5_predictions.jsonl")
    args = ap.parse_args()
    args.dataset_dir = os.path.abspath(args.dataset_dir)

    # The system first: the hybrid analyzer stops with a clear error when the dataset has no dictionary.
    system = ByT5System(args.model_dir) if args.system == "byt5" else HybridSystem(args.dataset_dir, args.no_neural)
    for split in args.split:
        path = dataset_loader.gold_file_path(args.dataset_dir, split)
        if not os.path.exists(path):
            sys.exit(f"[!] Gold file not found: {path}")
    name = args.system + (" (no neural)" if args.no_neural else "")
    cols = ["root", "type", "stem", "tense", "person", "gender", "number", "object", "exact"]
    results, predictions = {}, ([] if args.predictions_out else None)

    for split in args.split:
        stats = defaultdict(int)
        pairs = load_gold_pairs(args.dataset_dir, split, stats)
        print(f"\n{split}: {dataset_loader.gold_file_path(args.dataset_dir, split)}: {len(pairs)} pairs "
              f"({', '.join(f'{k} {v}' for k, v in sorted(stats.items()))})")
        if args.limit and args.limit < len(pairs):
            import random
            pairs = random.Random(42).sample(pairs, args.limit)
            print(f"[!] --limit {args.limit}: scoring a random sample, not the full split")
        totals, examples = score(pairs, system, predictions)
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
                    print(f"   {c:7s} {pct(*t[c])}")

        show(["ALL"])
        show(["type:" + t for t in dataset_loader.TYPES])
        show(["class:" + c for c in CLASS_ORDER])
        show(sorted(g for g in totals if g.startswith("tense:")))
        show(sorted(g for g in totals if g.startswith("tier:")))

        print("\nSample misses:")
        for item, pred in examples:
            print(f"  {item['surface']}  gold: {item['root']} {item['type']} {item['stem']} {item['tense']} "
                  f"{item['person'] or ''}{item['gender'] or ''}{item['number'] or ''} "
                  f"[{item['verb_class'] or item['type']}]  -> {pred['raw']}")

    print("\nRoot accuracy (exact match):")
    for split, totals in results.items():
        print(f"  {split:5s} {pct(*totals['ALL']['root'])}")

    if args.out_json:
        summary = {split: {"pairs": t["ALL"]["rows"][1], "root_correct": t["ALL"]["root"][0],
                           "root_accuracy": t["ALL"]["root"][0] / max(1, t["ALL"]["root"][1])}
                   for split, t in results.items()}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"system": name, "model_dir": args.model_dir if args.system == "byt5" else None,
                       "dataset_dir": os.path.abspath(args.dataset_dir), "root_exact_match": summary,
                       "by_group": results}, f, ensure_ascii=False, indent=1)
    if args.predictions_out:
        with open(args.predictions_out, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
