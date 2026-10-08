#!/usr/bin/env python3
"""
Synthetic morphology data generator for Classical Ge'ez.

Writes two files with the same pairs: synthetic_morphology_train.jsonl (features -> surface form) and
synthetic_morphology_analyze.jsonl (the same lines keyed by surface form). They cover:
1. Base triliteral paradigms from the patterns_triliteral.json templates.
2. Object pronoun suffixes (-ኒ, -ከ, -ኪ, -ሁ/-ዎ, -ሃ, -ነ, -ክሙ, -ክን, -ሙ/-ሆሙ, -ን/-ሆን);
   after a vowel the 3rd-person suffixes are -ዎ, -ሃ, -ሆሙ, -ሆን.
3. Negative verbal prefix (ኢ-).
4. Quadriliteral (four-letter) roots: ደንገፀ, ሰንበተ.
5. Checked class paradigms from verb_class_paradigms.py (via verb_classes.py):
   Type B (ቀደሰ), Type C (ባረከ), final-w (ተለወ), final-y (ሰተየ type), and the
   ወለደ/ወረደ subjunctive. New checked classes go in verb_class_paradigms.py.
   Classes come from the Kidane Wold verb triples (kidane_wold_triples.py), Merho
   Ges (the corrected gold train file, verb_classes.GOLD_TRAIN_FILE) and the class lemmas; roots come from the dictionary with the
   kidane_wold_root_fixes.jsonl corrections applied.

Roots of the gold test split are left out (see --keep_test_roots), so the
gold score measures unseen roots. By default these are the 185 roots in
data/heldout_test_roots.txt (see below); --test_file takes them from a
gold test file instead. By default the generator writes into the dataset folder: output goes to
<dataset>/output and is copied to <dataset>/data/paradigms (what train_byt5_morphology.py reads).
After the copy, split_heldout_synthetic.py moves every line whose root is a root of the
current gold val or test file into <dataset>/data/synthetic/ (held out, not read by
the trainer); --no_heldout_split skips that step. --no_mirror skips the copy and the split, and
--output_dir sends the output somewhere other than <dataset>/output.
Random draws are per root: which base pairs of a root get object and
negative forms, which object suffixes, which augmentation forms get the negative, and whether a
quadriliteral root is used all come from that root's own random.Random, seeded with
sha256("<SEED>:<purpose>:<root>") (root_rng). A change to one root (its class tag, its templates)
changes only that root's lines; every other root's lines stay identical and in the same order.
The synthetic files in the dataset were generated with the gold test file as extracted, dataset
commit c7bc587. Its test roots differ from the
current test file's. They are frozen in data/heldout_test_roots.txt (in this repo), so
reproducing the files needs no git history of the dataset:
  python generate_synthetic_data.py --dataset_dir <dataset>

Two more committed files in data/ limit the output; the generator stops if either is missing:
  data/synthetic_roots_allowed_v5.txt  the only roots it conjugates: roots with a dictionary verb
      entry, a gold root or a Kidane Wold verb triple, plus a few roots checked by hand (the file
      header records how the list was made).
  data/heldout_gold_forms.tsv       every gold val/test form of dataset commit 6183c24; a line
      whose surface form is one of them is not written. The rest of that root's lines are made as if
      the line were there, so no other line changes.
"""

import os
import sys
import json
import argparse
import hashlib
import random
import shutil
import unicodedata
from typing import List, Dict, Any, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import geez_script
import dataset_loader  # default_dataset_dir()
import verb_class_paradigms as vcp
import verb_classes
from engine import InterdigitationEngine, slots_from_spec
import engine as engine_module   # NON_HOLLOW is extended from the dictionary triples

OBJECT_SUFFIXES = [
    {"key": "1s", "person": "1", "number": "s", "gender": None, "suffix": "ኒ"},
    {"key": "2ms", "person": "2", "number": "s", "gender": "m", "suffix": "ከ"},
    {"key": "2fs", "person": "2", "number": "s", "gender": "f", "suffix": "ኪ"},
    {"key": "3ms", "person": "3", "number": "s", "gender": "m", "suffix": "ዎ"},
    {"key": "3fs", "person": "3", "number": "s", "gender": "f", "suffix": "ሃ"},
    {"key": "1p", "person": "1", "number": "p", "gender": None, "suffix": "ነ"},
    {"key": "2mp", "person": "2", "number": "p", "gender": "m", "suffix": "ክሙ"},
    {"key": "2fp", "person": "2", "number": "p", "gender": "f", "suffix": "ክን"},
    {"key": "3mp", "person": "3", "number": "p", "gender": "m", "suffix": "ሆሙ"},
    {"key": "3fp", "person": "3", "number": "p", "gender": "f", "suffix": "ሆን"},
]


# Per-root random streams: every draw for a root comes from root_rng(root, purpose), seeded from
# SEED, the purpose and the root string through sha256 (Python's hash() is salted per process).
# Rates are fixed numbers, not "N of all roots", so no root's draws depend on another root, and a
# change to one root (its class tag, its templates) changes only that root's lines.
SEED = 42
AUGMENT_EVERY_BASE = 8      # one in 8 Type A template pairs of a root -> object/negative forms
AUGMENT_EVERY_QUAD = 6      # one in 6 quadriliteral pairs of a root
QUAD_SAMPLE_RATE = 0.28     # share of quadriliteral roots used
NEGATIVE_RATE = 0.22        # share of augmentation forms given the ኢ- form


def root_rng(root: str, purpose: str, seed: int = SEED) -> random.Random:
    """random.Random for one root ("q-t-l") and one purpose ("augment", "objects", "negative",
    "quad_sample"), seeded with the first 8 bytes of sha256(f"{seed}:{purpose}:{root}")."""
    digest = hashlib.sha256(f"{seed}:{purpose}:{root}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# Third-person object suffixes change shape with the vowel they attach to.
# Keys are the OBJECT_SUFFIXES "suffix" values (the after-consonant shape).
THIRD_PERSON_AFTER_U = {"ዎ": "ዎ", "ሃ": "ዋ", "ሆሙ": "ዎሙ", "ሆን": "ዎን"}   # ቀተልኩዎ
THIRD_PERSON_AFTER_I = {"ዎ": "ዮ", "ሃ": "ያ", "ሆሙ": "ዮሙ", "ሆን": "ዮን"}   # ቀተልኪዮ
THIRD_PERSON_AFTER_LONG_A = {"ዎ": "ሁ", "ሃ": "ሃ", "ሆሙ": "ሆሙ", "ሆን": "ሆን"}  # ቀተላሁ


def attach_object_pronoun(surface: str, obj_suffix: str) -> Optional[str]:
    """Attach an object suffix to a verb form.

    The caller skips perfect 1p/2mp/2fp subjects (ቀተልነ, ቀተልክሙ, ቀተልክን):
    their object forms are not settled yet, so no unchecked forms enter the data.
    """
    if not surface:
        return None

    units = geez_script.decompose_string(surface)
    if not units:
        return None
    last_cons, last_vowel = units[-1]
    third = obj_suffix in ("ዎ", "ሃ", "ሆሙ", "ሆን")

    # Final 1st order (-a) or 6th order (no vowel): ቀተለ, ቀተልከ, ይቀትል, ቀተለት.
    if last_vowel in ("a", "I"):
        if obj_suffix == "ዎ":
            return surface[:-1] + geez_script.render_unit(last_cons, "o")          # ቀተሎ
        if obj_suffix == "ሃ":
            return surface[:-1] + geez_script.render_unit(last_cons, "A")          # ቀተላ, ይቀትላ
        if obj_suffix in ("ሆሙ", "ሆን"):
            return surface[:-1] + geez_script.render_unit(last_cons, "o") + obj_suffix[1:]  # ቀተሎሙ
        if last_vowel == "I":
            return surface[:-1] + geez_script.render_unit(last_cons, "a") + obj_suffix  # ይቀትለኒ
        return surface + obj_suffix                                                # ቀተለኒ

    if third and last_vowel == "u":
        return surface + THIRD_PERSON_AFTER_U[obj_suffix]
    if third and last_vowel == "i":
        return surface + THIRD_PERSON_AFTER_I[obj_suffix]
    if third and last_vowel == "A":
        return surface + THIRD_PERSON_AFTER_LONG_A[obj_suffix]
    return surface + obj_suffix


def load_quadriliteral_templates() -> List[Dict[str, Any]]:
    """Quadriliteral templates (patterns_quadriliteral.json, shared with the analyzer)."""
    with open(os.path.join(BASE_DIR, "patterns_quadriliteral.json"), encoding="utf-8") as f:
        return [{"features": q["features"], "spec": q["pattern_spec"]} for q in json.load(f)]


QUADRILITERAL_TEMPLATES = load_quadriliteral_templates()


# Ge'ez has one non-indicative prefix mood, the subjunctive (Lambdin, Dillmann).
# patterns_triliteral.json still has separate "jussive" templates that build the same
# forms, so they are relabeled here; the duplicate rows are then dropped.
TENSE_ALIASES = {"jussive": "subjunctive"}


def normalize_features(feat: Dict[str, Any]) -> Dict[str, Any]:
    """One label per category: jussive -> subjunctive, 1st person gender -> "c"."""
    feat = dict(feat)
    feat["tense"] = TENSE_ALIASES.get(feat.get("tense"), feat.get("tense"))
    if feat.get("person") == "1" and not feat.get("gender"):
        feat["gender"] = "c"
    return feat


def has_unassigned_char(text: str) -> bool:
    """True if text has a code point Unicode leaves unassigned (category Cn).

    Labiovelar rows (ቈ, ኰ) have no u or o order, so rendering qh/kh2 with
    those vowels gives an empty code point instead of a letter.
    """
    return any(unicodedata.category(ch) == "Cn" for ch in text)


def template_is_checked(pattern: Dict[str, Any], cons: Tuple[str, ...]) -> bool:
    """Keep only weak-verb templates whose output has been checked.

    - Hollow roots: templates that write the glide as a fixed letter ([ው] or
      [ይ]) skip the engine's contraction rule and put ይ into w-roots (ይቀይም),
      so only pure root-slot templates are used.
    - Initial-w roots: only the perfect and imperfect templates are right. The
      subjunctive drops the first radical (ይለድ); for the ወለደ type it comes from
      the initial_w_drop class paradigm, not from these templates.
    """
    feat = pattern["features"]
    spec = pattern["pattern_spec"]
    vtype = feat.get("verb_type")
    if vtype == "weak_middle" and ("[ው]" in spec or "[ይ]-C2" in spec or "-[ይ]-" in spec):
        return False
    if vtype == "weak_initial" and feat.get("tense") not in ("past", "present"):
        return False
    if vtype == "weak_initial" and "[የ]" in spec:
        return False
    return True


# First radical አ ዐ ሀ ሐ ኀ: the imperfect person prefix takes the 1st order
# (የኀፍር, ተኀፍር, አኀፍር, ነኀፍር); the subjunctive keeps ይ (ይኀፍር). See docs/grammar_rules.md section 1.
GUTTURAL_FIRST = ("a", "a2", "h1", "h2", "kh")
GUTTURAL_PREFIX = {"[ይ]": "[የ]", "[ት]": "[ተ]", "[እ]": "[አ]", "[ን]": "[ነ]"}


def guttural_prefix(spec: str) -> str:
    """"[ይ]-C0a-C1I-C2I" -> "[የ]-C0a-C1I-C2I"; other specs unchanged."""
    for old, new in GUTTURAL_PREFIX.items():
        if spec.startswith(old):
            return new + spec[len(old):]
    return spec


# The gold test roots of dataset commit c7bc587, one root per line (made with load_held_out_roots
# from that commit's merho_ges_gold_test.jsonl). The sha256 is checked in tests/test_heldout_roots_file.py.
HELD_OUT_ROOTS_FILE = os.path.join(BASE_DIR, "data", "heldout_test_roots.txt")


def read_held_out_roots_file(path: str = HELD_OUT_ROOTS_FILE) -> set:
    """Roots listed in a held-out roots file ("ቀ-ተ-ለ" per line, "#" lines are notes), as
    consonant-symbol tuples."""
    roots = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                roots.add(geez_script.root_symbols(line))
    return roots


# Roots the generator may conjugate, one per line as the "root" feature writes them ("h2-s-t"); "#" lines are
# notes; the file's header records how the list was made. Checked in
# tests/test_synthetic_allowed_roots_and_blocked_forms.py.
ALLOWED_ROOTS_FILE = os.path.join(BASE_DIR, "data", "synthetic_roots_allowed_v5.txt")
# Every gold val/test form of dataset commit 6183c24 (column geez_form; "#" lines are notes).
BLOCKED_FORMS_FILE = os.path.join(BASE_DIR, "data", "heldout_gold_forms.tsv")


def read_allowed_roots_file(path: str = ALLOWED_ROOTS_FILE) -> set:
    """Roots of the allow-list file as consonant-symbol tuples. Raises FileNotFoundError when the file is
    missing and ValueError on a line that is not a root of known consonant symbols."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"allow-list of synthetic roots not found: {path} "
                                "(the generator conjugates only the roots listed there)")
    roots = set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = tuple(line.split("-"))
            if len(parts) not in (2, 3, 4, 5) or not all(p in geez_script.CONSONANTS for p in parts):
                raise ValueError(f"{path} line {n}: {line!r} is not a root of consonant symbols")
            roots.add(parts)
    if not roots:
        raise ValueError(f"{path}: no roots")
    return roots


def read_blocked_forms_file(path: str = BLOCKED_FORMS_FILE) -> set:
    """Forms (column geez_form) of the blocked gold forms file. Raises FileNotFoundError when it is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"gold val/test forms file not found: {path} "
                                "(the generator leaves out every synthetic line with one of these forms)")
    forms = set()
    header = None
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cells = line.split("\t")
            if header is None:
                header = cells
                if "geez_form" not in header:
                    raise ValueError(f"{path} line {n}: no geez_form column in the header")
                col = header.index("geez_form")
                continue
            if not cells[col]:
                raise ValueError(f"{path} line {n}: empty geez_form")
            forms.add(cells[col])
    if header is None:
        raise ValueError(f"{path}: no header line")
    return forms


def load_held_out_roots(test_file: Optional[str]) -> set:
    """Roots of a gold test file (each record's root, every "/" alternative), as consonant-symbol tuples."""
    roots = set()
    if not test_file or not os.path.exists(test_file):
        return roots
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for alt in (rec.get("canonical_root") or rec.get("root") or "").split("/"):
                cons = geez_script.root_symbols(alt)
                if cons:
                    roots.add(cons)
    return roots


def load_dictionary_and_patterns(dataset_dir: str):
    dict_candidates = [
        os.path.join(dataset_dir, "dictionary.jsonl"),
        os.path.join(dataset_dir, "data", "geez_amharic_dictionary.jsonl"),
        os.path.join(dataset_dir, "output", "dictionary_with_roots.jsonl"),
    ]
    dict_path = None
    for c in dict_candidates:
        if os.path.exists(c):
            dict_path = c
            break

    patterns_path = os.path.join(BASE_DIR, "patterns_triliteral.json")
    if not os.path.exists(patterns_path):
        patterns_path = os.path.join(dataset_dir, "..", "geez-morphology-model", "patterns.json")

    dictionary = []
    if dict_path and os.path.exists(dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                dictionary.append(json.loads(line))

    patterns_data = []
    if patterns_path and os.path.exists(patterns_path):
        with open(patterns_path, "r", encoding="utf-8") as f:
            patterns_data = json.load(f)

    return dictionary, patterns_data


def generate_extended_dataset(
    dataset_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    include_objects: bool = True,
    include_negative: bool = True,
    include_quads: bool = True,
    held_out: Optional[set] = None,
    mirror_dir: Optional[str] = None,
    split_heldout: bool = False,
    allowed_roots_file: str = ALLOWED_ROOTS_FILE,
    blocked_forms_file: str = BLOCKED_FORMS_FILE,
):
    # Both files are read first: a missing file stops the run before anything is written.
    allowed_roots = read_allowed_roots_file(allowed_roots_file)
    blocked_forms = read_blocked_forms_file(blocked_forms_file)
    if dataset_dir is None:
        dataset_dir = dataset_loader.default_dataset_dir()
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    out_train = os.path.join(output_dir, "synthetic_morphology_train.jsonl")
    out_analyze = os.path.join(output_dir, "synthetic_morphology_analyze.jsonl")

    dictionary, patterns_data = load_dictionary_and_patterns(dataset_dir)
    engine = InterdigitationEngine(renderer=geez_script.render_unit)

    print(f"Loaded {len(dictionary)} dictionary entries and {len(patterns_data)} triliteral patterns.")

    triliteral_roots = set()
    quadriliteral_roots = set()

    skipped_roots = set()
    fixes = verb_classes.root_fixes(dataset_dir)
    print(f"Kidane Wold root fixes loaded: {len(fixes)} entries")
    for entry in dictionary:
        r = verb_classes.entry_root(entry, fixes)
        if r:
            clean = r.split(" (")[0].strip()
            parts = [p.strip() for p in clean.split("-") if p.strip()]
            # Every radical must be a consonant symbol the renderer knows.
            # Otherwise every template fails for the root (n-'-s, Ethiopic
            # "roots" from the raw root field): skip it and report it.
            if len(parts) in (3, 4) and not all(p in geez_script.CONSONANTS for p in parts):
                skipped_roots.add("-".join(parts))
                continue
            if len(parts) == 3:
                triliteral_roots.add(tuple(parts))
            elif len(parts) == 4:
                quadriliteral_roots.add(tuple(parts))

    held_out = held_out or set()   # root tuples left out of the data
    class_of, tag_counts = verb_classes.class_tags(dataset_dir, exclude=held_out)
    # Roots named by the checked lists are attested verbs even where the
    # dictionary's root inference missed them (ወለደ has no inferred root).
    added_roots = {r for r in ({tuple(r.split("-")) for r in vcp.W_DROP} | set(class_of))
                   if len(r) == 3 and all(c in geez_script.CONSONANTS for c in r)}
    added_roots -= triliteral_roots
    triliteral_roots |= added_roots

    # No root with an Amharic-only letter (ሸ ቸ ኘ ዠ;
    # ኸ ጀ ጨ never decompose and are already in skipped_roots); no doubly weak
    # triliteral root (two or more w/y: ሐይወ, የ-ወ-ሰ), which went through the hollow
    # or strong templates (ተሀይዉ, ሎየት, ይየብው, እወልይ).
    amharic_dropped = {"triliteral": sorted("-".join(r) for r in triliteral_roots
                                            if verb_classes.has_amharic_letter(r)),
                       "quadriliteral": [], "class_tags": sorted(
                           "-".join(r) for r in class_of if verb_classes.has_amharic_letter(r))}
    amharic_raw = sorted(r for r in skipped_roots
                         if any(ch in verb_classes.AMHARIC_ONLY_LETTERS for ch in r))
    doubly_weak_dropped = {"triliteral": sorted("-".join(r) for r in triliteral_roots
                                                if not verb_classes.has_amharic_letter(r)
                                                and verb_classes.weak_count(r) >= 2),
                           "quadriliteral": []}
    triliteral_roots = {r for r in triliteral_roots
                        if not verb_classes.has_amharic_letter(r) and verb_classes.weak_count(r) < 2}
    class_of = {r: c for r, c in class_of.items() if not verb_classes.has_amharic_letter(r)}
    quad_final_weak_dropped = []
    # Roots dropped by name.
    DROP_ROOTS = {("w", "a2", "a"): "ወ-ዐ-አ: no clean Kidane Wold perfect"}
    named_dropped = sorted("-".join(r) for r in triliteral_roots if r in DROP_ROOTS)
    triliteral_roots = {r for r in triliteral_roots if r not in DROP_ROOTS}
    class_of = {r: c for r, c in class_of.items() if r not in DROP_ROOTS}

    n_tri, n_quad = len(triliteral_roots), len(quadriliteral_roots)
    triliteral_roots = sorted(r for r in triliteral_roots if r not in held_out)
    quadriliteral_roots = sorted(r for r in quadriliteral_roots if r not in held_out)
    # Allow-list: only roots with verb evidence. Filtered after the held-out step; every draw
    # is per root (root_rng), so the kept roots' lines are the same as without the filter.
    n_tri_before_allow, n_quad_before_allow = len(triliteral_roots), len(quadriliteral_roots)
    allowed_not_found = allowed_roots - set(triliteral_roots) - set(quadriliteral_roots)
    triliteral_roots = [r for r in triliteral_roots if r in allowed_roots]
    quadriliteral_roots = [r for r in quadriliteral_roots if r in allowed_roots]
    print(f"Allow-list {os.path.basename(allowed_roots_file)}: {len(allowed_roots)} roots; left out "
          f"{n_tri_before_allow - len(triliteral_roots)} triliteral, {n_quad_before_allow - len(quadriliteral_roots)} "
          f"quadriliteral roots not on it; {len(allowed_not_found)} listed roots not among the generator's roots"
          + (f" {sorted('-'.join(r) for r in allowed_not_found)}" if allowed_not_found else ""))

    print(f"Roots added from checked lists / class tags: {len(added_roots)}")
    print(f"Held-out gold test roots removed: {n_tri - len(triliteral_roots)} triliteral, "
          f"{n_quad - len(quadriliteral_roots)} quadriliteral (from {len(held_out)} test roots)")
    print(f"Class tags: {tag_counts}")
    # Type A subjunctive/imperative shape from the dictionary triple: ይቅትል or ይግበር.
    subj_shape, subj_conflicts = verb_classes.type_a_subjunctive_shapes(dataset_dir)
    SHAPE_NAME = {("I", "I"): "ይቅትል", ("I", "a"): "ይግበር", ("a", "I"): "ይአድም (guttural C1)",
                  ("a", "a"): "ይሐበር-shape"}
    subj_used = {"no triple: current ይቅትል shape kept": 0}
    d_orders = verb_classes.type_d_orders(dataset_dir)
    # Exceptions from the Kidane Wold triple check (data/triple_check_exceptions.tsv)
    b_subj = verb_classes.type_b_subjunctive_orders(dataset_dir)          # S-a2-l: guttural middle, subjunctive from the triple
    kw_final_gutt_A = verb_classes.kw_final_guttural_subjunctive(dataset_dir)
    kw_w_drop_A = verb_classes.kw_w_drop_final_guttural(dataset_dir)
    kw_w_keep = verb_classes.kw_initial_w_keep_doubled(dataset_dir)
    engine_module.HOLLOW_SUBJ_4TH.update(verb_classes.kw_hollow_final_guttural_subjunctive(dataset_dir))
    engine_module.NON_HOLLOW.update(verb_classes.kw_quad_uncontracted(dataset_dir))
    rule_roots = {"type_b_subjunctive_from_triple": set(b_subj), "final_guttural_subjunctive": set(), "w_drop_final_guttural": set(),
                  "w_keep_subjunctive": set(), "hollow_subjunctive_4th": set(engine_module.HOLLOW_SUBJ_4TH),
                  "quad_uncontracted": {r for r in engine_module.NON_HOLLOW if len(r) == 4}}
    print(f"type_d roots with orders from their own triple: {sorted('-'.join(r) for r in d_orders)}")
    gutt_present = 0
    # Middle-w/y roots whose triple says type_a keep the glide (ኀየረ ፡ የኀይር ፡ ይኀይር):
    # no hollow contraction, ordinary Type A templates.
    keep_glide = {r for r, c in class_of.items() if c == "type_a" and len(r) == 3 and r[1] in ("w", "y")}
    engine_module.NON_HOLLOW.update(keep_glide)
    # ፍሬ ግእዝ middle-guttural rule (ይምሕል ፡ ይምሐል ፡ ምሐል) for default Type A roots.
    kw_triple_roots = verb_classes.triple_roots(dataset_dir)
    mid_gutt_changed = set()
    mid_gutt_excluded = {}
    mid_gutt_forms = {"present": 0, "subjunctive": 0, "imperative": 0}
    # Guttural-final rule (Merho Ges ሰምዐ, ቃርሐ):
    # last radical አ ዐ ሀ ሐ ኀ, past tense ->
    #   3rd person, basic/causative/aste: middle radical 6th order (ሰምዐ ሰምዐት ሰምዑ ሰምዓ, አስምዐ, አስተስምዐ);
    #     passive 3rd person is already short in the templates (ተሰምዐ) and is left as it is;
    #   1st/2nd person, basic/causative/aste/passive: middle radical 4th order (ā), the guttural
    #     6th as before (ሰማዕኩ, አስማዕኩ, አስተስማዕኩ, ተሰማዕኩ).
    # Quadriliterals and hollow roots (ሞአ, ኖኀ: the engine contracts radicals 1+2) unchanged.
    final_gutt_roots = set()
    final_gutt_forms = {}
    # Kidane Wold subjunctive written like the imperfect, with a 6th-order guttural middle (OCR collapse).
    kw_collapsed = verb_classes.kw_collapsed_subjunctive(dataset_dir)
    # Kidane Wold triple that writes the middle-guttural forms (ሰሐለ ፡ ይስሕል ፡ ይስሐል, class unclear): middle-guttural rule.
    kw_frye = verb_classes.kw_middle_guttural_pattern(dataset_dir)
    kw_frye_changed = set()
    kw_collapse_changed = set()
    print(f"Type A subjunctive shapes from triples: {len(subj_shape)} roots, {subj_conflicts} conflicting")
    print(f"Middle-w/y roots kept uncontracted (triple says type_a): {sorted('-'.join(r) for r in keep_glide)}")
    print(f"Attested roots: {len(triliteral_roots)} triliteral, {len(quadriliteral_roots)} quadriliteral.")
    if skipped_roots:
        print(f"Skipped {len(skipped_roots)} roots with unknown consonant symbols: {sorted(skipped_roots)}")

    def get_root_type(cons):
        if cons[0] in ("w", "y"):
            return "weak_initial"
        if cons[1] in ("w", "y"):
            return "weak_middle"
        if cons[2] in ("w", "y"):
            return "weak_final"
        return "strong_A"

    total_base = 0
    total_quads = 0
    total_objects = 0
    total_negatives = 0
    failed_forms = 0
    skipped_templates = 0
    skipped_objects = 0
    duplicates_dropped = 0
    unassigned_forms = 0
    blocked_lines = {}          # root -> {form: lines} left out because the form is a gold val/test form
    final_roots_skipped = {}
    class_counts = {}

    with open(out_train, "w", encoding="utf-8") as f_train, \
         open(out_analyze, "w", encoding="utf-8") as f_analyze:

        seen = set()

        def write_pair(feat: Dict[str, Any], form: str) -> bool:
            """Write one pair; skip it if the same (surface, features) was written."""
            nonlocal duplicates_dropped
            key = (form, json.dumps(feat, sort_keys=True, ensure_ascii=False))
            if key in seen:
                duplicates_dropped += 1
                return False
            seen.add(key)
            if form in blocked_forms:
                # A gold val/test form: not written. Returns True so the root's counts and augmentation
                # picks stay as if it were written; only this line is missing from the files.
                per_root = blocked_lines.setdefault(feat["root"], {})
                per_root[form] = per_root.get(form, 0) + 1
                return True
            f_train.write(json.dumps({"features": feat, "surface_form": form}, ensure_ascii=False) + "\n")
            f_analyze.write(json.dumps({"surface_form": form, "analysis": feat}, ensure_ascii=False) + "\n")
            return True

        print("Generating Phase 1: Base Triliteral Paradigms...")
        base_forms_for_augmentation = []
        obj_base = {}   # (root, form) -> form the object suffix is attached to, where it differs (final guttural rule)

        def write_class_rows(cons, cls, orders=None):
            """Checked paradigm of one class. Not used for object/negative
            augmentation: suffixes on these classes are not checked yet."""
            nonlocal total_base, unassigned_forms
            if orders is None:
                orders = b_subj.get(cons) if cls == "type_b" else d_orders.get(cons)
            for feat, form in verb_classes.paradigm_rows(cons, cls, orders):
                if has_unassigned_char(form):
                    unassigned_forms += 1
                    continue
                if write_pair(feat, form):
                    total_base += 1
                    class_counts[cls] = class_counts.get(cls, 0) + 1

        for cons in triliteral_roots:
            rtype = get_root_type(cons)
            cls = class_of.get(cons)
            if cons in keep_glide:
                rtype = "strong_A"
            if cls == "type_d" and cons not in d_orders:
                cls = None   # no usable triple vowels: ordinary templates
            if cls in ("type_b", "type_c", "type_c_o", "type_c_e", "type_d"):
                # Basic stem only; derived stems of Type B/C are not checked yet.
                write_class_rows(cons, cls)
                continue
            if rtype == "weak_final":
                # Only with a tag: ተለወ-type needs a ይተሉ/ይትሉ imperfect (ሀለወ, ይሄሉ,
                # is Type B with final w); the ሰተየ type needs a ይሰቲ/ይስተይ one
                # (በከየ-type has subjunctive ይብኪ).
                if cls == "final_w" and verb_classes.class_fits_root("final_w", cons):
                    write_class_rows(cons, "final_w")
                elif cls == "final_y" and verb_classes.class_fits_root("final_y", cons):
                    write_class_rows(cons, "final_y")
                else:
                    why = cls or ("untagged final-w" if cons[2] == "w" else "untagged final-y")
                    final_roots_skipped[why] = final_roots_skipped.get(why, 0) + 1
                continue
            if rtype == "weak_initial" and verb_classes.takes_w_drop(cons, class_of):
                write_class_rows(cons, "initial_w_drop")   # ይለድ, ለድ
            elif rtype == "weak_initial" and cons in kw_w_drop_A:
                write_class_rows(cons, "initial_w_drop", 4)   # final guttural: radical 2 in the 4th order (w-S2-a, w-k-a2, w-q-a2)
                rule_roots["w_drop_final_guttural"].add(cons)
            if rtype == "weak_initial" and cons in kw_w_keep:
                rule_roots["w_keep_subjunctive"].add(cons)   # subjunctive/imperative keep the w (w-k-k), Type A templates
            mid_gutt = False
            if rtype == "strong_A" and cons[1] in GUTTURAL_FIRST:
                why = ("own Kidane Wold triple" if cons in kw_triple_roots or cons in subj_shape else
                       f"class tag {cls}" if cls is not None else
                       "first radical also guttural" if cons[0] in GUTTURAL_FIRST else None)
                if (why == "own Kidane Wold triple" and cons in kw_collapsed and cls is None
                        and cons[0] not in GUTTURAL_FIRST and cons not in subj_shape
                        and kw_collapsed[cons]["imperfect"] == ("I", "I")):
                    mid_gutt = True                  # present = the triple's ይጽሕፍ; ይጽሐፍ, ጽሐፍ
                    kw_collapse_changed.add(cons)
                elif (why == "own Kidane Wold triple" and cons in kw_frye and cls is None
                        and cons[0] not in GUTTURAL_FIRST and cons not in subj_shape):
                    mid_gutt = True                  # the triple writes ይስሕል ፡ ይስሐል
                    kw_frye_changed.add(cons)
                elif why:
                    mid_gutt_excluded.setdefault(why, []).append("-".join(cons))
                else:
                    mid_gutt = True
                    mid_gutt_changed.add(cons)
            # Final guttural (triple_followed in the triple check): basic subjunctive/jussive/imperative put radical 2 in the
            # 4th order (ይስማዕ pattern). Default Type A roots; radicals 1 and 2 not gutturals; a root's own triple wins.
            final_gutt_subj = (rtype == "strong_A" and cons[2] in GUTTURAL_FIRST and cons[0] not in GUTTURAL_FIRST
                               and cons[1] not in GUTTURAL_FIRST and cons not in subj_shape
                               and (cons not in kw_triple_roots or cons in kw_final_gutt_A))
            if final_gutt_subj:
                rule_roots["final_guttural_subjunctive"].add(cons)
            if rtype == "strong_A":
                shape = subj_shape.get(cons)
                key = (SHAPE_NAME.get(shape, str(shape)) + " (from triple)") if shape else \
                    "no triple: current ይቅትል shape kept"
                subj_used[key] = subj_used.get(key, 0) + 1
            # Augmentation picks: every AUGMENT_EVERY_BASE-th template pair of THIS root, from an
            # offset drawn from the root's own stream (class rows above are not counted or used).
            aug_offset = root_rng("-".join(cons), "augment").randrange(AUGMENT_EVERY_BASE)
            root_pairs = 0
            for p in patterns_data:
                ptype = p["features"].get("verb_type", "strong_A")
                if rtype in ("weak_middle", "weak_initial"):
                    # Hollow roots go through the Type A basic-stem templates; the
                    # engine contracts slots 0+1 (ቆመ, ይቀውም, ይቁም). The hand-written
                    # weak_middle templates hardcode one glide and are not used.
                    # Initial-weak roots also use the Type A basic perfect and
                    # imperfect, which are right as written (ወለደ, ወለደት, ይወልድ,
                    # ትወልዲ). The hand-written weak_initial templates put 3fp
                    # shapes under 3fs labels (ወለዳ, ይወልዳ) and are not used.
                    if ptype != "strong_A" or p["features"].get("stem") != "basic":
                        continue
                    if (rtype == "weak_initial" and p["features"].get("tense") not in ("past", "present")
                            and cons not in kw_w_keep):
                        # Subjunctive drops the first radical (ይለድ): not modeled yet.
                        skipped_templates += 1
                        continue
                elif ptype != rtype:
                    continue
                if not template_is_checked(p, cons):
                    skipped_templates += 1
                    continue
                try:
                    spec = p["pattern_spec"]
                    aug_spec = None
                    basic = p["features"].get("stem") == "basic"
                    shape = subj_shape.get(cons) if rtype == "strong_A" else None
                    if shape and basic and p["features"].get("tense") in ("subjunctive", "jussive", "imperative"):
                        # vowels of radicals 1 and 2 as in the triple: ይቅትል, ይግበር, ይአድም
                        spec = spec.replace("C0I", "C0" + shape[0]).replace("C1I", "C1" + shape[1])
                    if mid_gutt and basic:
                        tns = p["features"].get("tense")
                        if tns == "present":
                            spec = spec.replace("-C0a-", "-C0I-", 1)        # ይምሕል, not ይመሕል
                            mid_gutt_forms["present"] += 1
                        elif tns in ("subjunctive", "jussive", "imperative"):
                            spec = spec.replace("C1I", "C1a", 1)            # ይምሐል, ምሐል
                            mid_gutt_forms["imperative" if tns == "imperative" else "subjunctive"] += 1
                    if (final_gutt_subj and basic and p["features"].get("tense") in ("subjunctive", "jussive", "imperative")
                            and spec.endswith("C2I")):
                        # only where the guttural ends the word (data/triple_check_exceptions_extra.tsv): before any vowel
                        # suffix, -ā included, radical 2 stays 6th (as gold train writes ይምስሑ, ይንቅሁ, ንሥኡ, ንሥኢ)
                        aug_spec = spec   # an object suffix gives radical 3 a vowel, so radical 2 stays 6th (gold ምርሐኒ)
                        spec = spec.replace("C1I", "C1A", 1)
                    if basic and p["features"].get("tense") == "present" and cons[0] in GUTTURAL_FIRST:
                        spec = guttural_prefix(spec)   # የኀፍር, not ይኀፍር
                        gutt_present += 1
                    pstem = p["features"].get("stem")
                    if (rtype in ("strong_A", "weak_initial") and cons[2] in GUTTURAL_FIRST
                            and p["features"].get("tense") == "past" and "-C1a-" in spec):
                        third = p["features"].get("person") == "3"
                        if third and pstem in ("basic", "causative", "aste"):
                            spec = spec.replace("-C1a-", "-C1I-", 1)      # ሰምዐ, አስምዐ, አስተስምዐ
                        elif not third and pstem in ("basic", "causative", "aste", "passive"):
                            spec = spec.replace("-C1a-", "-C1A-", 1)      # ሰማዕኩ, አስማዕኩ, ተሰማዕኩ
                        if "-C1a-" not in spec:
                            final_gutt_roots.add(cons)
                            k = (rtype, pstem, "3" if third else "1/2")
                            final_gutt_forms[k] = final_gutt_forms.get(k, 0) + 1
                    slots = slots_from_spec(spec)
                    form = engine.generate(cons, slots, stem=p["features"].get("stem", "basic"))
                    if has_unassigned_char(form):
                        unassigned_forms += 1
                        continue

                    feat = normalize_features(p["features"])
                    feat["root"] = "-".join(cons)
                    feat["verb_type"] = rtype

                    if not write_pair(feat, form):
                        continue
                    total_base += 1
                    root_pairs += 1

                    if (root_pairs + aug_offset) % AUGMENT_EVERY_BASE == 0:
                        if aug_spec:
                            obj_base[(cons, form)] = engine.generate(cons, slots_from_spec(aug_spec), stem=p["features"].get("stem", "basic"))
                        base_forms_for_augmentation.append((form, feat, cons))

                except Exception:
                    failed_forms += 1

        print(f"  Phase 1 base paradigms: {total_base:,} pairs")

        if include_quads:
            print("Generating Phase 2: Quadriliteral Verbs (ደንገፀ, ሰንበተ, etc.)...")
            # A quad root is used if its own stream draws it (QUAD_SAMPLE_RATE), in sorted root order.
            drawn_quads = {r for r in quadriliteral_roots
                           if root_rng("-".join(r), "quad_sample").random() < QUAD_SAMPLE_RATE}
            print(f"  Quad roots drawn (per-root, rate {QUAD_SAMPLE_RATE}): {len(drawn_quads)} of {len(quadriliteral_roots)}")

            # Every quad root with a class from its triple is used, drawn or not.
            confirmed_quads = [r for r in quadriliteral_roots
                               if class_of.get(r) in ("quad_c", "quadriliteral") and r not in drawn_quads]
            used_quads = drawn_quads | set(confirmed_quads)
            sampled_quads = [r for r in quadriliteral_roots if r in used_quads]
            print(f"  Confirmed quad roots added to the sample: {len(confirmed_quads)}")
            # Filtered after sampling so the rest of the sample is unchanged.
            # Amharic-only letter -> drop; two or more w/y -> doubly weak, drop; last
            # radical w/y -> no checked quad final-weak template (ፈቅድዉ, ንነዝፍው), drop.
            kept_quads = []
            for cons in sampled_quads:
                if verb_classes.has_amharic_letter(cons):
                    amharic_dropped["quadriliteral"].append("-".join(cons))
                elif verb_classes.weak_count(cons) >= 2:
                    doubly_weak_dropped["quadriliteral"].append("-".join(cons))
                elif cons[-1] in ("w", "y"):
                    quad_final_weak_dropped.append("-".join(cons))
                else:
                    kept_quads.append(cons)
            sampled_quads = kept_quads
            print(f"  Quad roots dropped from the sample: Amharic letter "
                  f"{len(amharic_dropped['quadriliteral'])}, doubly weak "
                  f"{len(doubly_weak_dropped['quadriliteral'])}, final w/y {len(quad_final_weak_dropped)}")
            for cons in sampled_quads:
                if class_of.get(cons) == "quad_c":
                    write_class_rows(cons, "quad_c")   # ማህረከ ፡ ይማህርክ
                    continue
                aug_offset = root_rng("-".join(cons), "augment").randrange(AUGMENT_EVERY_QUAD)
                root_pairs = 0
                for q_temp in QUADRILITERAL_TEMPLATES:
                    try:
                        spec = q_temp["spec"]
                        if q_temp["features"].get("tense") == "present" and cons[0] in GUTTURAL_FIRST:
                            spec = guttural_prefix(spec)   # የሐነፍጽ
                            gutt_present += 1
                        slots = slots_from_spec(spec)
                        form = engine.generate(cons, slots)
                        if has_unassigned_char(form):
                            unassigned_forms += 1
                            continue

                        feat = normalize_features(q_temp["features"])
                        feat["root"] = "-".join(cons)

                        if not write_pair(feat, form):
                            continue
                        total_quads += 1
                        root_pairs += 1

                        if (root_pairs + aug_offset) % AUGMENT_EVERY_QUAD == 0:
                            base_forms_for_augmentation.append((form, feat, cons))
                    except Exception:
                        failed_forms += 1

            print(f"  Phase 2 quadriliteral forms: {total_quads:,} pairs")

        if include_objects:
            print("Generating Phase 3: Object Pronominal Suffixes (-ኒ, -ከ, -ኪ, -ዎ, -ሃ, -ነ, -ክሙ, -ክን, -ሆሙ, -ሆን)...")
            print(f"  Augmentation forms (per-root picks): {len(base_forms_for_augmentation):,}")
            obj_rngs = {}

            for form, feat, cons in base_forms_for_augmentation:
                if cons not in obj_rngs:
                    obj_rngs[cons] = root_rng("-".join(cons), "objects")
                obj_sample = obj_rngs[cons].sample(OBJECT_SUFFIXES, 2)
                for obj in obj_sample:
                    if feat.get("person") == obj["person"] and obj["person"] in ("1", "2"):
                        continue

                    unsettled = feat.get("tense") == "past" and (feat.get("person"), feat.get("number")) in (("1", "p"), ("2", "p"))
                    obj_form = None if unsettled else attach_object_pronoun(obj_base.get((cons, form), form), obj["suffix"])
                    if obj_form is None:
                        skipped_objects += 1
                        continue
                    if has_unassigned_char(obj_form):
                        unassigned_forms += 1
                        continue
                    obj_feat = feat.copy()
                    obj_feat["object_suffix"] = obj["suffix"]
                    obj_feat["object_person"] = obj["person"]
                    obj_feat["object_number"] = obj["number"]
                    if obj["gender"]:
                        obj_feat["object_gender"] = obj["gender"]
                    obj_feat["object"] = obj["key"]

                    if write_pair(obj_feat, obj_form):
                        total_objects += 1

            print(f"  Phase 3 object suffix pairs: {total_objects:,} pairs")

        if include_negative:
            print("Generating Phase 4: Negative Prefix (ኢቀተለ, ኢይቀትል, ኢተቀተለ)...")
            # Each augmentation form gets the negative with NEGATIVE_RATE, from its root's own stream
            # (drawn for every form, imperative or not, so the draws do not depend on the skip below).
            neg_rngs = {}
            neg_sample = []
            for item in base_forms_for_augmentation:
                cons = item[2]
                if cons not in neg_rngs:
                    neg_rngs[cons] = root_rng("-".join(cons), "negative")
                if neg_rngs[cons].random() < NEGATIVE_RATE:
                    neg_sample.append(item)
            for form, feat, cons in neg_sample:
                if feat.get("tense") == "imperative":
                    continue
                neg_form = f"ኢ{form}"
                neg_feat = feat.copy()
                neg_feat["negative"] = True
                neg_feat["polarity"] = "negative"

                if write_pair(neg_feat, neg_form):
                    total_negatives += 1

            print(f"  Phase 4 negative prefix pairs: {total_negatives:,} pairs")

    total_pairs = total_base + total_quads + total_objects + total_negatives
    n_blocked = sum(sum(v.values()) for v in blocked_lines.values())
    print("\n============================================================")
    print("Synthetic data generation finished")
    print("============================================================")
    print(f"Base Triliterals:     {total_base:8,}")
    print(f"Quadriliteral Verbs:  {total_quads:8,}")
    print(f"Object Pronouns:      {total_objects:8,}")
    print(f"Negative Forms (ኢ-):  {total_negatives:8,}")
    print(f"Total pairs made:     {total_pairs:8,}")
    print(f"Left out (gold form): {n_blocked:8,}  (surface form is a gold val/test form, {os.path.basename(blocked_forms_file)})")
    print(f"Total Output Pairs:   {total_pairs - n_blocked:8,}")
    print(f"Engine failures:      {failed_forms:8,}")
    print(f"Duplicate pairs dropped (jussive = subjunctive): {duplicates_dropped:,}")
    print(f"Forms with unassigned code points skipped (labiovelar + u/o): {unassigned_forms:,}")
    print(f"Unchecked weak templates skipped: {skipped_templates:,}")
    print(f"Objects skipped:      {skipped_objects:8,}  (1p, 2mp, 2fp subjects: object forms not settled yet)")
    print(f"Checked class paradigm pairs: {class_counts}")
    print(f"Final-weak roots skipped (no ተለወ/ሰተየ tag): {final_roots_skipped}")
    print(f"Dropped roots: Amharic-only letter: triliteral {len(amharic_dropped['triliteral'])}, "
          f"quad (in sample) {len(amharic_dropped['quadriliteral'])}, class tags {len(amharic_dropped['class_tags'])}, "
          f"undecomposable roots with ኸ/ጀ/ጨ (already skipped) {len(amharic_raw)}")
    print(f"  Doubly weak: triliteral {len(doubly_weak_dropped['triliteral'])}, "
          f"quad (in sample) {len(doubly_weak_dropped['quadriliteral'])}")
    print(f"  Quads ending in w/y (in sample): {len(quad_final_weak_dropped)}")
    print(f"Roots dropped by name: {named_dropped}")
    print(f"Middle-guttural rule applied to {len(mid_gutt_changed)} roots; "
          f"templates rewritten (before duplicate removal): {mid_gutt_forms}")
    print(f"  Middle-guttural Type A roots left unchanged: "
          f"{ {k: len(v) for k, v in mid_gutt_excluded.items()} } {mid_gutt_excluded}")
    print(f"Kidane Wold subjunctive = imperfect (6th-order guttural middle): "
          f"{len(kw_collapsed)} roots {sorted('-'.join(r) for r in kw_collapsed)}; "
          f"fixed through the Type A templates: {sorted('-'.join(r) for r in kw_collapse_changed)}")
    print("KW_COLLAPSE_JSON " + json.dumps({
        "found": {"-".join(r): v for r, v in kw_collapsed.items()},
        "changed": sorted("-".join(r) for r in kw_collapse_changed)}, ensure_ascii=False))
    print("TRIPLE_CHECK_RULES_JSON " + json.dumps({k: sorted("-".join(r) for r in v) for k, v in rule_roots.items()}))
    print("KW_MID_GUTT_TRIPLE_JSON " + json.dumps({"found": sorted("-".join(r) for r in kw_frye), "changed": sorted("-".join(r) for r in kw_frye_changed)}))
    print("MID_GUTT_JSON " + json.dumps({"changed": sorted("-".join(r) for r in mid_gutt_changed),
                                                "excluded": mid_gutt_excluded}, ensure_ascii=False))
    print("DROPPED_ROOTS_JSON " + json.dumps({
        "amharic_only": amharic_dropped, "amharic_raw_letters": amharic_raw,
        "doubly_weak": doubly_weak_dropped, "quad_final_weak": quad_final_weak_dropped},
        ensure_ascii=False))
    print(f"Gold val/test forms left out: {n_blocked} lines, {sum(len(v) for v in blocked_lines.values())} forms, "
          f"{len(blocked_lines)} roots {sorted(blocked_lines)}")
    print("BLOCKED_GOLD_FORMS_JSON " + json.dumps(
        {r: {"lines": sum(v.values()), "forms": dict(sorted(v.items()))} for r, v in sorted(blocked_lines.items())},
        ensure_ascii=False))
    print(f"Type A roots by subjunctive/imperative shape: {subj_used}")
    print(f"Present forms given the guttural prefix (የ/ተ/አ/ነ): {gutt_present:,}")
    print(f"Guttural-final rule (past: middle radical 6th order in 3rd person, 4th (ā) in 1st/2nd): {len(final_gutt_roots)} roots; "
          f"templates rewritten by (type, stem, person): {dict(sorted(final_gutt_forms.items()))}")
    print(f"Saved to: {out_train}")
    if mirror_dir and os.path.abspath(mirror_dir) != os.path.abspath(output_dir):
        os.makedirs(mirror_dir, exist_ok=True)
        for path in (out_train, out_analyze):
            shutil.copyfile(path, os.path.join(mirror_dir, os.path.basename(path)))
        print(f"Copied to:  {mirror_dir} (train_byt5_morphology.py reads this copy)")
        if split_heldout:
            import split_heldout_synthetic
            moved, _ = split_heldout_synthetic.split_into_dataset(output_dir, dataset_dir)
            print(f"Held out (split_heldout_synthetic.py): {sum(moved.values()):,} lines whose root is a gold val/test root "
                  f"-> {os.path.join(dataset_dir, split_heldout_synthetic.HELDOUT_DIR)}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic morphology data for Classical Ge'ez. "
                                     "By default it writes into the dataset folder; see --no_mirror and --output_dir.")
    parser.add_argument("--dataset_dir", default=dataset_loader.default_dataset_dir(),
                        help="Checkout of the geez-dataset repo (default: $GEEZ_DATASET_DIR, else ../geez-dataset)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: <dataset>/output)")
    parser.add_argument("--no_objects", action="store_true", help="Exclude object pronominal suffixes")
    parser.add_argument("--no_negative", action="store_true", help="Exclude negative prefix")
    parser.add_argument("--no_quads", action="store_true", help="Exclude quadriliterals")
    parser.add_argument("--keep_test_roots", action="store_true",
                        help="Do not remove the gold test roots (makes the gold score optimistic)")
    parser.add_argument("--no_mirror", action="store_true",
                        help="Leave the dataset's data/ folder alone: do not copy the output to "
                             "<dataset>/data/paradigms and do not split off the held-out lines")
    parser.add_argument("--test_file", default=None,
                        help="Leave out the roots of this gold test file instead of the frozen list "
                             "data/heldout_test_roots.txt (the default)")
    parser.add_argument("--no_heldout_split", action="store_true",
                        help="Do not move the lines with a gold val/test root to <dataset>/data/synthetic/ after the copy")
    args = parser.parse_args()

    resolved_dataset_dir = os.path.abspath(args.dataset_dir)
    if args.keep_test_roots:
        held_out = set()
    elif args.test_file:
        held_out = load_held_out_roots(args.test_file)
    else:
        held_out = read_held_out_roots_file()

    generate_extended_dataset(
        dataset_dir=resolved_dataset_dir,
        output_dir=args.output_dir,
        include_objects=not args.no_objects,
        include_negative=not args.no_negative,
        include_quads=not args.no_quads,
        held_out=held_out,
        mirror_dir=None if args.no_mirror else os.path.join(resolved_dataset_dir, "data", "paradigms"),
        split_heldout=not args.no_mirror and not args.no_heldout_split,
    )
