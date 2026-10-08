#!/usr/bin/env python3
"""
evaluate_gold.py
Score a system on the held-out Merho Ges gold test set.

Systems:
  --system byt5    the ByT5 checkpoint alone (default; latest checkpoint-N in --model_dir)
  --system hybrid  HybridAnalyzer: fast path -> pattern matcher -> ByT5 fallback
                   (add --no_neural to switch the ByT5 fallback off)

What is scored (no hidden misses):
  - Every verb_conjugation row in the test file, every "/" variant. Variants
    that are not a single Ethiopic word (notes, "(?)", Latin, spaces) are
    skipped and counted.
  - A field counts only where the gold row has a value; empty gold tense or
    person is reported as "unlabeled", not scored as wrong.
  - Pairs and labels come from gold_filters.file_pairs: the gold file as written, without the
    rows listed in merho_ges_gold_flagged_rows.tsv (the same reader the trainer uses).
      verbs     root, type, stem, tense, person, gender, number (+ object where gold has one)
      nominals  root, type, number (+ gender where gold has one); no stem/tense/person
    "exact" = every field the gold pair labels. A system that only answers verbs (the hybrid
    rule engine) predicts type "verb", so its nominal "type" and "exact" are wrong by design.
  - A predicted "jussive" is read as "subjunctive" (the gold files use subjunctive only).
  - Roots are compared as consonant sequences, whatever script the gold uses
    (Ethiopic, q-t-l symbols, or ṭ-y-b transliteration).
  - Results are broken down by verb class, tense, and (hybrid) by which tier answered.
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
import gold_filters  # reads the gold files; shared with train_byt5_morphology.py
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


def root_key(root_str):
    """Root in any script -> tuple of consonant symbols (or raw characters), after
    writing every letter in the 1st order of its series (geez_config.first_order_root),
    for gold and predicted roots alike."""
    if not root_str:
        return None
    return geez_config.root_symbols(geez_config.first_order_root(root_str)) or None


def gold_root_keys(root_str):
    """Gold roots may list alternatives ("ጤበ / ጠየበ"); any of them counts."""
    if not root_str:
        return set()
    keys = {root_key(r) for r in root_str.split("/")}
    keys.discard(None)
    return keys


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
    keys = gold_root_keys(rec.get("root"))
    root = max(keys, key=len) if keys else ()
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


def load_gold_test_pairs(test_file, stats=None):
    """Scored pairs of a gold file's verb_conjugation records, read as written (gold_filters.file_pairs),
    without the rows in merho_ges_gold_flagged_rows.tsv next to the file. Returns (pairs, records)."""
    stats = stats if stats is not None else defaultdict(int)
    name = os.path.basename(test_file)
    split = next((s for s in gold_filters.SPLITS if name == gold_filters.gold_file_name(s)), None)
    dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(test_file))))
    recs = [r for r in gold_filters.read_records(test_file) if r.get("category") == "verb_conjugation"]
    gold = gold_filters.file_pairs(recs, split, gold_filters.read_flagged_rows(dataset_dir), stats)
    pairs = []
    for p in gold:
        pairs.append({
            "surface": p["form"],
            "root_keys": gold_root_keys(p["root"]),
            "root_raw": p["root"],
            "verb_class": verb_class(p["record"]) if p["type"] == "verb" else None,
            **normalize_row(p),
        })
    return pairs, len({p["record_id"] for p in gold})


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
    def __init__(self, no_neural=False):
        from analyzer import HybridAnalyzer
        self.a = HybridAnalyzer()
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


def score(pairs, system):
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # group -> field -> [correct, labeled]
    examples = []
    for item in pairs:
        pred = system.predict(item["surface"])
        ok = {}
        pk = root_key(pred["root"]) if pred["root"] else None
        ok["root"] = (pk in item["root_keys"]) if item["root_keys"] else None
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
    return totals, examples


def pct(c, n):
    return f"{100 * c / n:5.1f}% ({c}/{n})" if n else "   n/a"


def main():
    ap = argparse.ArgumentParser(description="Evaluate on the Merho Ges gold test set")
    ap.add_argument("--system", choices=["byt5", "hybrid"], default="byt5")
    ap.add_argument("--no_neural", action="store_true", help="hybrid only: no ByT5 fallback")
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                    help="The trainer's --output_dir (default: byt5_morphology_model/), or one checkpoint-<step> folder")
    ap.add_argument("--dataset_dir", default=gold_filters.default_dataset_dir(),
                    help="Checkout of the geez-dataset repo the model was trained on "
                         "(default: $GEEZ_DATASET_DIR, else ../geez-dataset)")
    ap.add_argument("--limit", type=int, default=0, help="Random sample of N pairs (0 = all, the default)")
    ap.add_argument("--out_json", default=None, help="Write the metrics table to this JSON file")
    ap.add_argument("--export_pairs", default=None, help="Write the scored pairs to this JSONL file")
    args = ap.parse_args()

    args.test_file = gold_filters.gold_file_path(args.dataset_dir, "test")
    if not os.path.exists(args.test_file):
        sys.exit(f"[!] Test file not found: {args.test_file}")

    stats = defaultdict(int)
    pairs, records = load_gold_test_pairs(args.test_file, stats)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    if args.export_pairs:
        with open(args.export_pairs, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps({k: (sorted("-".join(x) for x in v) if k == "root_keys" else v)
                                    for k, v in p.items()}, ensure_ascii=False) + "\n")
    print(f"Gold file: {args.test_file}")
    print(f"verb_conjugation records: {records}; scored pairs: {len(pairs)}")
    if args.limit and args.limit < len(pairs):
        import random
        pairs = random.Random(42).sample(pairs, args.limit)
        print(f"[!] --limit {args.limit}: scoring a random sample, not the full set")

    system = ByT5System(args.model_dir) if args.system == "byt5" else HybridSystem(args.no_neural)
    totals, examples = score(pairs, system)

    name = args.system + (" (no neural)" if args.no_neural else "")
    print("\n" + "=" * 72)
    print(f"GOLD TEST RESULTS: {name}   (accuracy over rows where gold has the field)")
    print("=" * 72)
    cols = ["root", "type", "stem", "tense", "person", "gender", "number", "object", "exact"]

    def show(group_names):
        for g in group_names:
            if g not in totals:
                continue
            t = totals[g]
            print(f"\n[{g}]  rows={t['rows'][1]}")
            for c in cols:
                print(f"   {c:7s} {pct(*t[c])}")

    show(["ALL"])
    show(["type:" + t for t in gold_filters.TYPES])
    show(["class:" + c for c in CLASS_ORDER])
    show(sorted(g for g in totals if g.startswith("tense:")))
    show(sorted(g for g in totals if g.startswith("tier:")))

    print("\nSample misses:")
    for item, pred in examples:
        print(f"  {item['surface']}  gold: {item['root_raw']} {item['type']} {item['stem']} {item['tense']} "
              f"{item['person'] or ''}{item['gender'] or ''}{item['number'] or ''} "
              f"[{item['verb_class'] or item['type']}]  -> {pred['raw']}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({g: {k: v for k, v in t.items()} for g, t in totals.items()}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
