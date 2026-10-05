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
  - jussive and subjunctive are one label (subjunctive) on both sides.
  - Roots are compared as consonant sequences, whatever script the gold uses
    (Ethiopic, q-t-l symbols, or ṭ-y-b transliteration).
  - Results are broken down by verb class, tense, and (hybrid) by which tier answered.
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import geez_config
import gold_filters  # load-time root corrections, shared with train_byt5_morphology.py

FIELDS = ["root", "stem", "tense", "person", "gender", "number"]
TENSE_ALIASES = {"jussive": "subjunctive"}
# Merho Ges "causative_intensive" = አስተ- forms (synthetic stem "aste") plus
# reduplicated causatives (አንጸብረቀ). Only the አስተ- forms are scored as "aste".
ASTE_FORM = re.compile(r"^(ኢ)?[አያታናእይትን]ስተ")
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
    """Shared label normalization for gold and predictions."""
    d = {k: clean_label(d.get(k)) for k in ("stem", "tense", "person", "gender", "number")}
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
    keys = gold_root_keys(rec.get("canonical_root") or rec.get("root"))
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


def is_single_ethiopic_word(s):
    return bool(s) and all(0x1200 <= ord(c) <= 0x137F for c in s)


def load_exclusions(path):
    """(form, gold root) pairs that a relabel-proposals file marks as not verbs
    or unusable (exclude_from_verb_scoring); test split only."""
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test" and r.get("exclude_from_verb_scoring"):
                out.add((geez_config.normalize_guttural_4th(r["form"]), r.get("root")))
    return out


def load_gold_test_pairs(test_file, exclude=None, corrections=None, stats=None):
    """Scored pairs of a gold file. With `corrections` (gold_filters.Corrections) the records go
    through the same load-time root corrections as training, and excluded forms are left out."""
    pairs, skipped, records = [], [], 0
    excluded = 0
    stats = stats if stats is not None else defaultdict(int)
    with open(test_file, "r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    if corrections is not None:
        recs = corrections.gold_records(recs, stats)
    for rec in recs:
        if rec.get("category") != "verb_conjugation":
            continue
        records += 1
        vclass = verb_class(rec)
        for row in rec.get("table_data", []):
            form_raw = row.get("geez_form") or ""
            form_main = re.sub(r"\([^)]*\)", " ", form_raw)  # drop "(...)" notes
            for v in form_main.split("/"):
                v = v.strip(" :፡።()[]፣,")
                if not v:
                    continue
                if not is_single_ethiopic_word(v):
                    skipped.append(v)
                    continue
                if corrections is not None and gold_filters.is_excluded(row, v):
                    stats["excluded_variants"] += 1
                    continue
                # proposals name the root as extracted; fixed records keep it in root_as_extracted
                if exclude and (geez_config.normalize_guttural_4th(v),
                                rec.get("root_as_extracted") or rec.get("root")) in exclude:
                    excluded += 1
                    continue
                gold = normalize_row({"stem": rec.get("stem"), **row})
                if gold["stem"] == "causative_intensive" and ASTE_FORM.match(v):
                    gold["stem"] = "aste"
                pairs.append({
                    "surface": v,
                    "root_keys": gold_root_keys(rec.get("canonical_root") or rec.get("root")),
                    "root_raw": rec.get("root"),
                    "root_as_extracted": rec.get("root_as_extracted", rec.get("root")),
                    "verb_class": vclass,
                    **gold,
                })
    if exclude is not None:
        print(f"Excluded by --exclude_proposed: {excluded} scored variants")
    return pairs, skipped, records


class ByT5System:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        chkpt = model_dir
        if os.path.isdir(model_dir):
            subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
            if subdirs:
                chkpt = sorted(subdirs, key=lambda x: int(x.split("-")[-1]))[-1]
        print(f"Loading model checkpoint from: {chkpt}")
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(chkpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(chkpt).to(self.device)
        self.model.eval()

    def predict(self, surface):
        inp = self.tok(f"analyze: {surface}", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_length=128, num_beams=2)
        text = self.tok.decode(out[0], skip_special_tokens=True)
        pred = parse_prediction(text)
        return {"root": pred.get("root"), **normalize_row(pred), "tier": "byt5", "raw": text}


class HybridSystem:
    def __init__(self, no_neural=False):
        from analyzer import HybridAnalyzer
        self.a = HybridAnalyzer()
        if no_neural:
            self.a.checkpoint_dir = None

    def predict(self, surface):
        res = self.a.analyze(surface)
        if not res:
            return {"root": None, **normalize_row({}), "tier": "no_answer", "raw": ""}
        best = res[0]
        spec = best.pattern_spec or ""
        tier = ("fast_path" if spec.startswith("fast_path") else
                "neural" if spec.startswith("neural") else "pattern")
        pred = {"stem": best.stem, "tense": best.tense, "person": best.person,
                "gender": best.gender, "number": best.number}
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
        labeled_all = all(ok[f] is not None for f in FIELDS)
        ok["exact"] = all(ok[f] for f in FIELDS) if labeled_all else None
        groups = ["ALL", "class:" + item["verb_class"], "tense:" + str(item["tense"]), "tier:" + pred["tier"]]
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
    ap.add_argument("--model_dir", default=os.path.join(SCRIPT_DIR, "checkpoints"))
    default_test = os.path.join(SCRIPT_DIR, "..", "geez-dataset-v1", "data", "paradigms", "merho_ges_gold_test.jsonl")
    ap.add_argument("--test_file", default=os.path.normpath(default_test),
                    help="Must be the split whose *_train.jsonl the model saw (train_byt5_morphology.py injects data/paradigms/merho_ges_gold_train.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="Random sample of N pairs (0 = all, the default)")
    ap.add_argument("--out_json", default=None, help="Write the metrics table to this JSON file")
    ap.add_argument("--exclude_proposed", nargs="?", default=None,
                    const=os.path.join(SCRIPT_DIR, "notes", "gold_proposals_2026-10-05.jsonl"),
                    help="Leave out gold rows the relabel proposals mark exclude_from_verb_scoring "
                         "(not verbs, or unusable until re-read from the scan). Default file: "
                         "notes/gold_proposals_2026-10-05.jsonl. The gold file is not changed.")
    ap.add_argument("--no_corrections", action="store_true",
                    help="Score the gold file as written (no data/corrections tables; the pre-2026-10-05 numbers)")
    ap.add_argument("--corrections_dir", default=None,
                    help="Default: <dataset>/data/corrections next to the test file")
    ap.add_argument("--export_pairs", default=None, help="Write the scored pairs to this JSONL file")
    args = ap.parse_args()

    if not os.path.exists(args.test_file):
        sys.exit(f"[!] Test file not found: {args.test_file}")

    exclude = load_exclusions(args.exclude_proposed) if args.exclude_proposed else None
    corrections, stats = None, defaultdict(int)
    if not args.no_corrections:
        corrections = gold_filters.Corrections(args.corrections_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(args.test_file)))),
            gold_filters.CORRECTIONS_DIR), required=True)
        print(f"Corrections: {corrections.folder}")
        for name, sha in sorted(corrections.sha256.items()):
            print(f"  {name}  sha256 {sha}")
    pairs, skipped, records = load_gold_test_pairs(args.test_file, exclude, corrections, stats)
    if corrections is not None:
        print("Load counts (records/pairs as the trainer counts variants):")
        for line in gold_filters.format_stats({k: v for k, v in stats.items() if k.split(" ")[0] in ("records", "pairs")}):
            print(line)
        print(f"Excluded scored variants (gold_excluded_records.tsv): {stats['excluded_variants']}")
    if args.export_pairs:
        with open(args.export_pairs, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps({k: (sorted("-".join(x) for x in v) if k == "root_keys" else v)
                                    for k, v in p.items()}, ensure_ascii=False) + "\n")
    print(f"Gold file: {args.test_file}")
    print(f"verb_conjugation records: {records}; scored variants: {len(pairs)}; skipped non-word variants: {len(skipped)} {skipped[:8]}")
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
    cols = ["root", "stem", "tense", "person", "gender", "number", "exact"]

    def show(group_names):
        for g in group_names:
            if g not in totals:
                continue
            t = totals[g]
            print(f"\n[{g}]  rows={t['rows'][1]}")
            for c in cols:
                print(f"   {c:7s} {pct(*t[c])}")

    show(["ALL"])
    show(["class:" + c for c in CLASS_ORDER])
    show(sorted(g for g in totals if g.startswith("tense:")))
    show(sorted(g for g in totals if g.startswith("tier:")))

    print("\nSample misses:")
    for item, pred in examples:
        print(f"  {item['surface']}  gold: {item['root_raw']} {item['stem']} {item['tense']} "
              f"{item['person']}{item['gender'] or ''}{item['number'] or ''} [{item['verb_class']}]  -> {pred['raw']}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({g: {k: v for k, v in t.items()} for g, t in totals.items()}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
