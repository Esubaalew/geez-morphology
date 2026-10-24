"""The training mixture of the full fine-tune: every source of the dataset, one target format, one leak gate.

Tasks (input prefix -> target), all trained on the same pairs:
  analyze:  <form>      -> the full analysis (dataset_loader.target_text), "root: none | type: ..." for a word
                           without root; the format the analyzer reads at inference
  generate: <analysis>  -> <form>    (the reverse direction; pairs with a root only)
  pos:      <form>      -> "type: <class>"

Sources (sample "source" field):
  synthetic          data/paradigms/synthetic_morphology_train.jsonl (a seeded 2% is kept aside as a loss-only slice)
  gold_verb, gold_nominal   gold train paradigms
  unified_book       book rows of the unified file, split train
  dictionary         Kidane Wold nouns and adjectives, every root length in --dictionary_radicals (2-radical roots
                     only with --biradical_dictionary; default 3,4), on or off the synthetic allow-list; a pair
                     whose root was checked and not passed (data/dictionary_root_check.tsv) is left out
  dictionary_verb    Kidane Wold verb headwords that are their root's 3ms perfect
  word_without_root  data/closed_class_words_without_root.tsv, full target with the word's class
  extra_synthetic    extra_synthetic.py families, only those named in --extra_synthetic (default checked);
                     "checked" gives only the checked cells of data/extra_synthetic_checked.tsv

Upsampling: gold verbs x--gold_upsample, gold nominals x--gold_nominal_upsample, unified rows x--unified_upsample,
and every row of a rare class (RARE_CLASSES) is repeated until its class has at least --rare_class_min rows
(at most --rare_class_max_factor times).

Leak gate (leak_gate): a training pair is dropped when its form, or its form up to spelling, is a held-out form
(gold val/test, unified book val/test, the generator's blocked forms), or its root, first-order and folded, is a
held-out root (gold val/test roots, unified val/test roots, the generator's held-out roots file, the roots of the
dataset's held-out synthetic lines). This runs on every source, the synthetic one included.

Dev set (dev_samples): gold val plus unified book val, a seeded sample when capped; never trained on.
"""
import collections
import csv
import json
import os
import random

import dataset_loader
import geez_script
from geez_script import to_ethiopic_root

SEED = 42
GENERATE_TASK = "generate"
POS_TASK = "pos"
PREFIXES = {dataset_loader.ANALYZE_TASK: "analyze: ", GENERATE_TASK: "generate: ", POS_TASK: "pos: "}
RARE_CLASSES = ("gerund", "verbal_noun", "infinitive", "demonstrative", "preposition", "pronoun", "numeral")
# kind column of the words-without-root file -> class written in the target
CLOSED_CLASS_TYPE = {"pronoun": "pronoun", "demonstrative": "demonstrative", "preposition": "preposition",
                     "interrogative": "pronoun", "conjunction": "particle", "relative": "particle",
                     "negation": "particle", "interjection": "particle", "name": "name", "loan": "loanword"}
SYNTHETIC_FILE = os.path.join("data", "paradigms", "synthetic_morphology_train.jsonl")
SYNTHETIC_LOSS_SLICE = 0.02


def form_of_input(text):
    return text.split(":", 1)[1].strip()


def sample(pair, source, task=dataset_loader.ANALYZE_TASK):
    pair = dict(pair, task=dataset_loader.ANALYZE_TASK)
    analysis = dataset_loader.target_text(pair)
    if task == GENERATE_TASK:
        inp, tgt = PREFIXES[GENERATE_TASK] + analysis, pair["form"]
    elif task == POS_TASK:
        inp, tgt = PREFIXES[POS_TASK] + pair["form"], f"type: {pair_class(pair)}"
    else:
        inp, tgt = PREFIXES[dataset_loader.ANALYZE_TASK] + pair["form"], analysis
    return {"input_text": inp, "target_text": tgt, "task": task, "source": source, "form": pair["form"],
            "root": pair.get("root"), "cls": pair_class(pair)}


def pair_class(pair):
    return pair.get("type") or "verb"


# ----------------------------------------------------------------------------- held-out words

class HeldOut:
    def __init__(self, dataset_dir):
        import generate_synthetic_data
        import split_heldout_synthetic as sh
        self.forms = set(sh.gold_eval_forms(dataset_dir)) | generate_synthetic_data.read_blocked_forms_file()
        roots = set(sh.gold_eval_roots(dataset_dir))
        for split in ("val", "test"):
            unified = dataset_loader.load_unified(dataset_dir, split, ("book",))
            unified += dataset_loader.load_unified(dataset_dir, split, ("clitic_rule",), path=CLITIC_FILE)
            for p in unified:
                self.forms.add(p["form"])
                if p["root"]:
                    roots.add(geez_script.first_order_root(p["root"]))
            for p in dataset_loader.load_gold_split(dataset_dir, split):
                self.forms.add(p["form"])
                if p["root"]:
                    roots.add(geez_script.first_order_root(p["root"]))
        roots |= {geez_script.first_order_root(to_ethiopic_root("-".join(s)))
                  for s in generate_synthetic_data.read_held_out_roots_file()}
        roots |= {r for r in dataset_loader.held_out_synthetic_roots(dataset_dir) if r}
        self.roots = roots
        self.folded_forms = {geez_script.fold_spelling(f) for f in self.forms}
        self.folded_roots = {sh.fold_root(r) for r in roots}

    def leaks(self, form, root):
        if form in self.forms or geez_script.fold_spelling(form) in self.folded_forms:
            return "form"
        if root and root != dataset_loader.NO_ROOT:
            r = geez_script.first_order_root(root)
            if r in self.roots or geez_script.fold_spelling(r) in self.folded_roots:
                return "root"
        return None


def leak_gate(pairs, held, stats, source):
    kept = []
    for p in pairs:
        why = held.leaks(p["form"], p.get("root"))
        if why:
            stats[f"{source}: dropped, held-out {why}"] += 1
        else:
            kept.append(p)
    return kept


# ----------------------------------------------------------------------------- sources

def synthetic_pairs(dataset_dir, limit=0):
    with open(os.path.join(dataset_dir, SYNTHETIC_FILE), encoding="utf-8") as f:
        lines = f.readlines()
    rng = random.Random(SEED)
    if limit and limit < len(lines):
        lines = rng.sample(lines, limit)
    out = []
    for line in lines:
        d = json.loads(line)
        feat = d.get("features", {})
        out.append({"form": d.get("surface_form", ""), "root": to_ethiopic_root(feat.get("root", "")), "type": "verb",
                    "tense": feat.get("tense", ""), "person": feat.get("person", ""), "gender": feat.get("gender", ""),
                    "number": feat.get("number", ""), "stem": feat.get("stem", ""), "object": feat.get("object") or ""})
    return out


def words_without_root_pairs(stats):
    import csv
    out = []
    with open(dataset_loader.WORDS_WITHOUT_ROOT_FILE, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            cls = CLOSED_CLASS_TYPE.get(r["kind"], "particle")
            stats[f"word_without_root kind {r['kind']} -> {cls}"] += 1
            out.append({"form": r["word"], "root": None, "type": cls})
    return out


def dictionary_pairs(dataset_dir, radicals, stats):
    pairs = dataset_loader.load_dictionary_pairs(dataset_dir, stats, include_forms=True, allowed_roots_only=False,
                                                 radicals=set(radicals))
    out = []
    for p in pairs:
        if p.get("from_forms"):
            out.append(dict(p, type="nominal", gender="", number="", task=dataset_loader.ANALYZE_TASK))
        else:
            out.append(dict(p, task=dataset_loader.ANALYZE_TASK))
    return out


DICTIONARY_ROOT_CHECK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dictionary_root_check.tsv")


def dictionary_root_check(pairs, stats):
    """Drop the dictionary pairs whose root a Ge'ez reader checked and did not pass (DICTIONARY_ROOT_CHECK_FILE:
    verdict FIX, CHECK, DROP or VERB). Those are roots off the generator's allow-list, all 5-radical roots, and
    words of an entry's forms list that are not nouns or adjectives (the related verb, for example); a
    pair is matched by record_id, from_forms and form. A headword pair not in the file (an allow-list root) is
    kept; a forms-list word is kept only when the file passes it as OK."""
    verdict = {}
    with open(DICTIONARY_ROOT_CHECK_FILE, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            verdict[(r["record_id"], r["from_forms"] == "True", r["form"])] = r["verdict"]
    out = []
    for p in pairs:
        key = (p["record_id"], bool(p.get("from_forms")), p["form"])
        v = verdict.get(key, "NOT CHECKED" if p.get("from_forms") else "OK")
        if v == "OK":
            out.append(p)
        else:
            stats[f"dictionary root check: not loaded, {v}"] += 1
    return out


def rare_class_upsample(samples, classes, minimum, max_factor, stats):
    by = collections.defaultdict(list)
    for s in samples:
        by[s["cls"]].append(s)
    extra = []
    for c in classes:
        rows = by.get(c, [])
        if not rows or len(rows) >= minimum:
            continue
        factor = min(max_factor, -(-minimum // len(rows)))
        extra += [s for s in rows for _ in range(factor - 1)]
        stats[f"rare class {c}: {len(rows)} rows x{factor}"] = len(rows) * factor
    return samples + extra


# ----------------------------------------------------------------------------- the mixture

CLITIC_FILE = os.path.join("data", "clitics", "clitic_forms.jsonl")


def extra_rootless_pairs(path, stats):
    """Names and primary nouns without a root from a review table (word, word_class, review_note): a proper name
    gets type "name"; a primary noun keeps type "noun" with the number the note gives (singular / plural), and is
    left out when the note gives none."""
    import re
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            note = (r.get("review_note") or "").lower()
            if "proper noun" in note or "(name)" in note or "name)" in note:
                out.append({"form": r["word"], "root": None, "type": "name"})
                stats["extra_rootless: name"] += 1
                continue
            num = "p" if re.search(r"\bplural\b|broken pl", note) else "s" if re.search(r"\bsingular\b|\bcollective\b", note) else ""
            if not num:
                stats["extra_rootless: noun without number, left out"] += 1
                continue
            out.append({"form": r["word"], "root": None, "type": "noun", "number": num, "gender": "", "tense": None,
                        "person": None, "stem": None, "object": ""})
            stats["extra_rootless: primary noun"] += 1
    return out


def build(dataset_dir, opts):
    """opts: a namespace with the trainer's data flags (see train_tpu.add_data_args). Returns
    (train samples, synthetic loss slice, report dict)."""
    import extra_synthetic

    stats = collections.Counter()
    held = HeldOut(dataset_dir)
    tasks = [dataset_loader.ANALYZE_TASK] + ([GENERATE_TASK] if opts.generate_task else []) + \
            ([POS_TASK] if opts.pos_task else [])
    sources = collections.OrderedDict()

    syn = synthetic_pairs(dataset_dir, opts.sample_limit)
    rng = random.Random(SEED)
    rng.shuffle(syn)
    cut = int(len(syn) * SYNTHETIC_LOSS_SLICE)
    syn_slice, syn = syn[:cut], syn[cut:]
    frac = getattr(opts, "synthetic_fraction", 1.0)
    if frac < 1.0:                       # fine-tune: a seeded share of the synthetic pairs
        syn = syn[:int(len(syn) * frac)]
        stats["synthetic: kept share"] = round(frac, 3)
    sources["synthetic"] = (leak_gate(syn, held, stats, "synthetic"), 1)

    gold = dataset_loader.load_gold_split(dataset_dir, "train", stats)
    gold = leak_gate(gold, held, stats, "gold")
    sources["gold_verb"] = ([p for p in gold if p["type"] == "verb"], opts.gold_upsample)
    sources["gold_nominal"] = ([p for p in gold if p["type"] != "verb"], opts.gold_nominal_upsample)
    gold_forms = {p["form"] for p in gold}

    unified = [] if opts.no_unified else dataset_loader.load_unified(dataset_dir, "train", ("book",), stats)
    sources["unified_book"] = (leak_gate(unified, held, stats, "unified_book"), opts.unified_upsample)
    taken = gold_forms | {p["form"] for p in unified}

    radicals = [int(x) for x in opts.dictionary_radicals.split(",")]
    if not opts.biradical_dictionary:
        radicals = [r for r in radicals if r != 2]
    dic = dictionary_pairs(dataset_dir, radicals, stats)
    dic = dictionary_root_check(dic, stats)
    dic = [p for p in leak_gate(dic, held, stats, "dictionary") if p["form"] not in taken]
    sources["dictionary"] = (dic, 1)
    dverbs = leak_gate(dataset_loader.load_dictionary_verbs(dataset_dir, stats), held, stats, "dictionary_verb")
    sources["dictionary_verb"] = ([p for p in dverbs if p["form"] not in taken], 1)
    taken |= {p["form"] for p in dic} | {p["form"] for p in dverbs}

    wwr = words_without_root_pairs(stats)
    if getattr(opts, "extra_rootless", None):
        known = {p["form"] for p in wwr}
        wwr += [p for p in extra_rootless_pairs(opts.extra_rootless, stats) if p["form"] not in known]
    wwr = leak_gate(wwr, held, stats, "word_without_root")
    sources["word_without_root"] = ([p for p in wwr if p["form"] not in taken], getattr(opts, "rootless_upsample", 1))

    if getattr(opts, "clitic_forms", False):
        cl = dataset_loader.load_unified(dataset_dir, "train", ("clitic_rule",), stats, path=CLITIC_FILE)
        sources["clitic"] = (leak_gate(cl, held, stats, "clitic"), getattr(opts, "clitic_upsample", 1))

    families = [f for f in (opts.extra_synthetic or "").split(",") if f]
    extra = extra_synthetic.checked_pairs() if "checked" in families else []
    families = [f for f in families if f != "checked"]
    extra += extra_synthetic.generate_pairs(families, dataset_dir) if families else []
    stats["extra_synthetic pairs"] = len(extra)
    sources["extra_synthetic"] = (leak_gate(extra, held, stats, "extra_synthetic"), 1)

    train, report = [], {"pairs_by_source": {}, "samples_by_source_task": {}}
    for name, (pairs, times) in sources.items():
        report["pairs_by_source"][name] = len(pairs)
        for p in pairs:
            for task in tasks:
                if task == GENERATE_TASK and (not p.get("root") or p.get("from_forms") or name == "clitic"):
                    continue          # an analysis without a root, from unlabelled sub-forms, or of a clitic form
                                      # (the analysis does not name the clitic) names no single form
                s = sample(p, name, task)
                train += [s] * times
    train = rare_class_upsample(train, RARE_CLASSES, opts.rare_class_min, opts.rare_class_max_factor, stats)
    random.Random(opts.seed).shuffle(train)
    c = collections.Counter((s["source"], s["task"]) for s in train)
    report["samples_by_source_task"] = {f"{a}/{b}": n for (a, b), n in sorted(c.items())}
    report["samples_by_class"] = dict(collections.Counter(s["cls"] for s in train).most_common())
    report["train_samples"] = len(train)
    report["load_stats"] = dict(sorted(stats.items()))
    report["tasks"] = tasks
    return train, [sample(p, "synthetic_slice") for p in syn_slice], report


def exclude_checkpoint_exposure(pairs, manifest_path, stats=None):
    """Conservatively exclude forms/roots eligible for training in a checkpoint's lineage.

    A manifest is not proof that every eligible example was visited, nor does this filter
    establish an independently collected benchmark. Dataset split assignments are unchanged.
    """
    stats = stats if stats is not None else {}
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("forms"), list) \
            or not isinstance(manifest.get("roots"), list):
        raise ValueError("Exposure manifest must have schema_version 1 and forms/roots lists")
    if not all(isinstance(s, str) and s for s in manifest["forms"] + manifest["roots"]):
        raise ValueError("Exposure manifest forms and roots must be nonempty strings")
    forms = {geez_script.fold_spelling(form) for form in manifest["forms"]}
    roots = {geez_script.fold_spelling(geez_script.first_order_root(root))
             for root in manifest["roots"] if root != dataset_loader.NO_ROOT}
    stats.update(exposure_manifest=os.path.abspath(manifest_path),
                 exposure_coverage=manifest.get("coverage", "unspecified"),
                 pairs_before_exposure_filter=len(pairs), pairs_excluded_seen_form=0,
                 pairs_excluded_seen_root=0)
    kept = []
    for pair in pairs:
        if geez_script.fold_spelling(pair["form"]) in forms:
            stats["pairs_excluded_seen_form"] += 1
        elif pair.get("root") and pair["root"] != dataset_loader.NO_ROOT and \
                geez_script.fold_spelling(geez_script.first_order_root(pair["root"])) in roots:
            stats["pairs_excluded_seen_root"] += 1
        else:
            kept.append(pair)
    stats["pairs_after_exposure_filter"] = len(kept)
    return kept


def dev_samples(dataset_dir, cap=0, gold_only=False, pos_task=True, seen_manifest=None, stats=None):
    """Analyze (and pos) samples of the gold val pairs plus the unified book val rows, a seeded sample of `cap`
    pairs when cap > 0. Each keeps "dev" = gold_val or unified_val."""
    pairs = [dict(p, dev="gold_val") for p in dataset_loader.load_gold_split(dataset_dir, "val")]
    if not gold_only:
        pairs += [dict(p, dev="unified_val") for p in dataset_loader.load_unified(dataset_dir, "val", ("book",))]
    stats = stats if stats is not None else {}
    if seen_manifest:
        pairs = exclude_checkpoint_exposure(pairs, seen_manifest, stats)
    else:
        stats.update(pairs_before_exposure_filter=len(pairs), pairs_after_exposure_filter=len(pairs),
                     pairs_excluded_seen_form=0, pairs_excluded_seen_root=0)
    if cap and cap < len(pairs):
        keep = sorted(random.Random(SEED).sample(range(len(pairs)), cap))
        pairs = [pairs[i] for i in keep]
    stats["pairs_after_cap"] = len(pairs)
    out = []
    for p in pairs:
        for task in [dataset_loader.ANALYZE_TASK] + ([POS_TASK] if pos_task else []):
            out.append(dict(sample(p, p["dev"], task), dev=p["dev"]))
    stats["dev_samples"] = len(out)
    return out


def write_jsonl(path, rows, keys=("input_text", "target_text", "task", "source")):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n")
