"""Read the Merho Ges gold files of a geez-dataset checkout, as written.

The dataset carries its corrections in the files themselves (the change log lives in the dataset
repo), so nothing is corrected here. The trainer and evaluate_gold.py both read the gold files
through load_gold_split(), so they see the same pairs.

A pair is one form of one table row, with the labels that row gives it. A row's own root, stem,
type and object override the record's. A pair does not load when:

  - it is listed in data/paradigms/merho_ges_gold_flagged_rows.tsv (a file of the dataset);
  - its row has no root: the row has none of its own and the record root is empty, or the
    record is marked root_unsettled.

Anything else that is not a clean pair (a root that is not 2-5 Ge'ez letters, a form that is not
one Ge'ez word) raises ValueError: the dataset's scripts/check_dataset.py rejects such files, so
the loader does not skip them silently. The counts match the "pairs that load" line of that check.

load_dictionary_pairs() reads the nouns and adjectives of the Kidane Wold dictionary as headword-to-root
pairs; its docstring lists what it keeps and what it leaves out.
"""
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import geez_script  # noqa: E402

SPLITS = ("train", "val", "test")
DATASET_DIR_ENV = "GEEZ_DATASET_DIR"
PARADIGMS_DIR = os.path.join("data", "paradigms")
FLAGGED_ROWS_FILE = "merho_ges_gold_flagged_rows.tsv"
TYPES = ("verb", "noun", "adjective")
NOMINAL_TYPES = ("noun", "adjective")
# The other word classes of data/unified (the unified row table): they have their own target shape (target_text).
# Rooted: the root is read; rootless: the word has no root and the target says "root: none".
UNIFIED_ROOTED_TYPES = ("infinitive", "gerund", "verbal_noun", "nominal")
UNIFIED_ROOTLESS_TYPES = ("pronoun", "demonstrative", "preposition", "particle", "numeral", "adverb")
UNIFIED_TYPES = UNIFIED_ROOTED_TYPES + UNIFIED_ROOTLESS_TYPES
# Classes of the words-without-root file that the unified rows do not use; rootless, generic target shape.
CLOSED_CLASS_EXTRA_TYPES = ("name", "loanword")
ROOTLESS_TYPES = UNIFIED_ROOTLESS_TYPES + CLOSED_CLASS_EXTRA_TYPES
UNIFIED_FILE = os.path.join("data", "unified", "geez_morphology_rows.jsonl")
# Label fields of the generic target shape, in the order they are written.
GENERIC_FIELDS = ("tense", "person", "gender", "number", "stem", "object", "possessor")
# The "task" field of a pair says what the model is asked for, and so both its input prefix and its target.
# A pair without a "task" field has the full analysis.
ANALYZE_TASK = "analyze"          # target: the full analysis (dataset_loader.target_text)
ROOT_ONLY_TASK = "root_only"      # target: the root alone, "root: <root>"
TASK_PREFIXES = {ANALYZE_TASK: "analyze: ", ROOT_ONLY_TASK: "root: "}
# The root written for a word that has none (target "root: none"); the analyzer and evaluate_gold.py read
# "none" as no value.
NO_ROOT = "none"


def default_dataset_dir():
    """Where the geez-dataset checkout is when no --dataset_dir is given: $GEEZ_DATASET_DIR if set,
    else ../geez-dataset next to this repo (the folder `git clone .../geez-dataset` makes)."""
    return os.environ.get(DATASET_DIR_ENV) or os.path.normpath(os.path.join(SCRIPT_DIR, "..", "geez-dataset"))


def gold_file_name(split):
    """The gold file of a split: merho_ges_gold_<split>.jsonl in data/paradigms/."""
    return f"merho_ges_gold_{split}.jsonl"


def gold_file_path(dataset_dir, split):
    return os.path.join(dataset_dir, PARADIGMS_DIR, gold_file_name(split))


def record_id(rec):
    """The id stored in a gold record. Records without one get it computed the way the dataset
    computed it once: sha1("page|title_or_lemma|raw_transcription")[:12]."""
    if rec.get("record_id"):
        return rec["record_id"]
    key = f"{rec.get('page_number')}|{rec.get('title_or_lemma')}|{rec.get('raw_transcription')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def is_single_ethiopic_word(s):
    return bool(s) and all(0x1200 <= ord(c) <= 0x137F for c in s)


def clean_variants(form_raw):
    """The "/"-variants of a geez_form cell, with "(...)" notes dropped and edge punctuation
    stripped, as the dataset's check reads them. Returns (words, other): `words` are single Ge'ez
    words, `other` the remaining non-empty variants (open brackets, several words, Latin)."""
    words, other = [], []
    for v in re.sub(r"\([^)]*\)", " ", form_raw or "").split("/"):
        v = v.strip(" :፡።()[]፣,")
        if not v:
            continue
        (words if is_single_ethiopic_word(v) else other).append(v)
    return words, other


def root_problem(root):
    """Why a root is not a usable label (None when it is): 2-5 Ge'ez letters joined by "-"."""
    if not root:
        return "empty root"
    if "/" in root or "," in root:
        return "two roots joined"
    if any(not (0x1200 <= ord(c) <= 0x137F) and c != "-" for c in root):
        return "Latin or other non-Ethiopic text in root"
    parts = root.split("-")
    if any(len(p) != 1 or geez_script.decompose_char(p)[0] is None for p in parts):
        return "letter outside the Ge'ez table"
    if not 2 <= len(parts) <= 5:
        return "root is not 2-5 letters"
    return None


def read_flagged_rows(dataset_dir):
    """{(split, record_id, row, form)} from data/paradigms/merho_ges_gold_flagged_rows.tsv
    (empty when the file is missing)."""
    path = os.path.join(dataset_dir, PARADIGMS_DIR, FLAGGED_ROWS_FILE)
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return {(r["split"], r["record_id"], int(r["row"]), r["form"]) for r in rows}


def read_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_pairs(records, split, flagged=(), stats=None):
    """The pairs of one split's records, in file order. `flagged` holds (split, record_id, row, form)
    keys; `stats` (a Counter) gets the loaded / not-loaded counts.

    Each pair is a dict: form, root, type, tense, person, gender, number, stem, object, record,
    record_id, row. Nouns and adjectives have tense, person, stem and object set to None."""
    stats = stats if stats is not None else collections.Counter()
    out = []
    for rec in records:
        rid = record_id(rec)
        rec_root = "" if rec.get("root_unsettled") else (rec.get("root") or "")
        for j, row in enumerate(rec.get("table_data", [])):
            words, other = clean_variants(row.get("geez_form"))
            root = row.get("root") or rec_root
            if not root:
                stats["not loaded: row has no root"] += len(words) + len(other)
                continue
            problem = root_problem(root)
            if problem:
                raise ValueError(f"{split} record {rid} row {j}: root {root!r}: {problem} "
                                 "(run the dataset's scripts/check_dataset.py)")
            for v in words + other:
                if (split, rid, j, v) in flagged:
                    stats["not loaded: flagged (" + FLAGGED_ROWS_FILE + ")"] += 1
                    continue
                if v in other:
                    raise ValueError(f"{split} record {rid} row {j}: form {v!r} is not one Ge'ez word "
                                     "(run the dataset's scripts/check_dataset.py)")
                typ = row.get("type") or "verb"
                pair = {"form": v, "root": root, "type": typ, "gender": row.get("gender") or "",
                        "number": row.get("number") or "", "record": rec, "record_id": rid, "row": j}
                if typ in NOMINAL_TYPES:
                    pair.update(tense=None, person=None, stem=None, object=None)
                else:
                    pair.update(tense=row.get("tense") or "", person=row.get("person") or "",
                                stem=row.get("stem") or rec.get("stem") or "", object=row.get("object") or "")
                stats["loaded"] += 1
                out.append(pair)
    return out


def load_gold_split(dataset_dir, split, stats=None):
    """The pairs of data/paradigms/merho_ges_gold_<split>.jsonl, flagged rows left out."""
    records = read_records(gold_file_path(dataset_dir, split))
    return file_pairs(records, split, read_flagged_rows(dataset_dir), stats)


# ------------------------------------------------------------------ forms printed in the other books

BOOK_FORMS_FILE = os.path.join("data", "lexicon", "book_forms.jsonl")
BOOK_FORM_SPLITS = ("train", "heldout")


def load_book_forms(dataset_dir, split, stats=None):
    """Pairs of data/lexicon/book_forms.jsonl for one split ("train", or "heldout": a gold val/test root or form).
    Only rows with an empty status load (CHECK, CONFLICT and NO_ROOT rows stay in the file), and only verbs
    with tense, person, gender, number and stem, and nouns and adjectives with a number. [] when the file is missing."""
    if split not in BOOK_FORM_SPLITS:
        raise ValueError(f"book_forms split must be one of {BOOK_FORM_SPLITS}, not {split!r}")
    stats = stats if stats is not None else collections.Counter()
    path = os.path.join(dataset_dir, BOOK_FORMS_FILE)
    if not os.path.exists(path):
        return []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            r = json.loads(line)
            if r.get("split") != split:
                continue
            if r.get("status"):
                stats[f"not loaded: status {r['status']}"] += 1
                continue
            t = r.get("type")
            need = ("tense", "person", "gender", "number", "stem") if t == "verb" else ("number",)
            if t not in ("verb",) + NOMINAL_TYPES or not r.get("root") or any(not r.get(k) for k in need):
                stats["not loaded: type or labels missing"] += 1
                continue
            pairs.append({"form": r["form"], "root": r["root"], "type": t, "tense": r.get("tense"),
                          "person": r.get("person"), "gender": r.get("gender"), "number": r.get("number"),
                          "stem": r.get("stem"), "object": r.get("object") or "", "book": r.get("book"),
                          "page": r.get("page"), "line": n})
            stats["loaded"] += 1
    return pairs


# ------------------------------------------------------------------ the unified rows (all books and the gold, one schema)

def load_unified(dataset_dir, split, sources=("book",), stats=None):
    """Pairs of data/unified/geez_morphology_rows.jsonl for one split ("train", "val" or "test"). Only rows with an
    empty status load (CHECK, CONFLICT, NO_ROOT, FLAGGED and EXCLUDED rows stay in the file). `sources` picks
    "book" rows, "gold" rows or both (the gold rows are also read by load_gold_split; the trainer takes the
    book rows). A row of a rooted class needs its root; a rootless class (UNIFIED_ROOTLESS_TYPES) has root None
    and the target 'root: none'. The pair has the keys of load_gold_split's pairs plus possessor, negated, source,
    book, page and id. [] when the file is missing."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    stats = stats if stats is not None else collections.Counter()
    path = os.path.join(dataset_dir, UNIFIED_FILE)
    if not os.path.exists(path):
        return []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != split or r.get("source") not in sources:
                continue
            if r.get("status"):
                stats[f"not loaded: status {r['status']}"] += 1
                continue
            t = r["word_class"]
            if t not in TYPES + UNIFIED_TYPES:
                stats[f"not loaded: class {t}"] += 1
                continue
            if t in UNIFIED_ROOTLESS_TYPES:
                root = None
            elif not r.get("root"):
                stats["not loaded: row has no root"] += 1
                continue
            else:
                root = r["root"]
            pairs.append({"form": r["form"], "root": root, "type": t, "tense": r.get("tense"), "person": r.get("person"),
                          "gender": r.get("gender") or "", "number": r.get("number") or "", "stem": r.get("stem"),
                          "object": r.get("object") or "", "possessor": r.get("possessor") or "",
                          "negated": bool(r.get("negated")), "source": r["source"], "book": r.get("book"),
                          "page": r.get("page"), "id": r["id"]})
            stats["loaded"] += 1
    return pairs


# ------------------------------------------------------------------ inputs and targets

def task_of(pair):
    return pair.get("task") or ANALYZE_TASK


def input_text(form, task=ANALYZE_TASK, single_prefix=False):
    """Model input: the prefix of the task (TASK_PREFIXES) followed by the form. With single_prefix every task
    gets the full-analysis prefix "analyze: ", so the input does not tell the tasks apart."""
    return TASK_PREFIXES[ANALYZE_TASK if single_prefix else task] + form


def target_text(pair):
    """Training target. Verbs: 'root | tense | person | gender | number | stem' plus ' | object: X'
    when the form carries an object suffix. Nouns: 'root | type: noun | number'. Adjectives:
    'root | type: adjective | gender | number' (gender left out when the book gives none).
    Every other class of the unified rows (UNIFIED_TYPES: infinitive, gerund, verbal_noun, nominal and the
    rootless pronoun, demonstrative, preposition, particle, numeral, adverb) has the generic shape
    'root: <root or none> | type: <class>' followed by the labels it has, in GENERIC_FIELDS order.
    A pair with a possessor ends with ' | possessor: 3ms'; a negated pair ends with ' | negated: yes'.
    A pair whose task is ROOT_ONLY_TASK has the root alone: 'root: <root>' ('root: none' for a word without root)."""
    if task_of(pair) == ROOT_ONLY_TASK:
        return f"root: {pair['root']}"
    tail = (f" | possessor: {pair['possessor']}" if pair.get("possessor") and pair["type"] not in UNIFIED_TYPES else "") + \
           (" | negated: yes" if pair.get("negated") else "")
    if pair["type"] in UNIFIED_TYPES + CLOSED_CLASS_EXTRA_TYPES:
        parts = [f"root: {pair.get('root') or NO_ROOT}", f"type: {pair['type']}"]
        parts += [f"{k}: {pair[k]}" for k in GENERIC_FIELDS if pair.get(k)]
        return " | ".join(parts) + (" | negated: yes" if pair.get("negated") else "")
    if pair["type"] in NOMINAL_TYPES:
        parts = [f"root: {pair['root']}", f"type: {pair['type']}"]
        if pair["type"] == "adjective" and pair.get("gender"):
            parts.append(f"gender: {pair['gender']}")
        parts.append(f"number: {pair['number']}")
        return " | ".join(parts) + tail
    text = (f"root: {pair['root']} | tense: {pair['tense']} | person: {pair['person']} | gender: {pair['gender']} | "
            f"number: {pair['number']} | stem: {pair['stem']}")
    return text + (f" | object: {pair['object']}" if pair.get("object") else "") + tail


# ------------------------------------------------------------------ Kidane Wold nouns and adjectives

KIDANE_WOLD_FILE = os.path.join("data", "lexicon", "classical_dictionary_kidane_wold.jsonl")
# headword_type values that are not one dictionary word: several words, page markup, a single letter.
NOT_A_WORD_ENTRY = ("phrase", "artifact", "character")
# Text in root_notes that says the root still needs a look (the dataset writes "CHECK:" for that).
UNCLEAR_ROOT_MARKERS = ("CHECK", "unclear", "unsettled")
ROOT_RADICALS_KEPT = (3, 4)
# The dictionary gives no number for a headword. A headword is the citation form, and the citation form of a
# Ge'ez noun or adjective is the singular, so a pair gets "s" unless its entry has a "number" field.
CITATION_NUMBER = "s"
DICTIONARY_EVAL_FRACTION = 0.1
DICTIONARY_SPLIT_SEED = 42
# Stats keys of a pair or word left out because its form, folded with geez_script.fold_spelling, is a folded gold
# val/test form (split_heldout_synthetic.gold_eval_forms_folded).
GOLD_FORM_UP_TO_SPELLING = "not loaded: form is a gold val/test form up to spelling"
GOLD_FORM_UP_TO_SPELLING_EVAL = "not loaded (eval slice): form is a gold val/test form up to spelling"
# Words with no root (particles, pronouns, names, loanwords), one per line in column "word"; label "none".
# They are left out of the dictionary pairs and trained on as root-only pairs (load_words_without_root).
WORDS_WITHOUT_ROOT_FILE = os.path.join(SCRIPT_DIR, "data", "closed_class_words_without_root.tsv")
# The dictionary's abbreviation for a Greek source, followed by the word separator, as it is written inside the
# parenthesis that opens a definition. A word taken from Greek has no Ge'ez root. Other source abbreviations
# (e.g. the one for an Arabic cognate) do not mark a loanword and are not checked.
GREEK_SOURCE_MARKER = 'ጽር ፡'
# Entries whose definition text is too garbled to check the headword and the root against it.
UNVERIFIABLE_ENTRY_IDS = frozenset({3524})


def is_geez_word(s):
    """One word of Ethiopic syllables (U+1200-U+135A): no space, word separator, digit, dot, hyphen or bracket."""
    return bool(s) and all(0x1200 <= ord(c) <= 0x135A for c in s)


def dictionary_entry_root(entry, fixes):
    """The root of a dictionary entry in Latin notation ("q-t-l"), or None. An entry with a row in the dataset's
    an optional root-fixes file overrides inferred_root (null means no root), as the verb generator
    does (verb_classes.entry_root); otherwise its inferred_root. The Ge'ez `root` field is not read."""
    import verb_classes
    if entry.get("id") in fixes:
        return verb_classes.usable_root(fixes[entry["id"]])
    return verb_classes.usable_root(entry.get("inferred_root"))


def root_marked_unclear(entry, value):
    """True when the entry says its root is not settled: root_unsettled set, a "?" in the root, or a note
    in root_notes with one of UNCLEAR_ROOT_MARKERS."""
    notes = entry.get("root_notes") or ""
    return bool(entry.get("root_unsettled")) or "?" in (value or "") or \
        any(m in notes if m.isupper() else m in notes.lower() for m in UNCLEAR_ROOT_MARKERS)


WEAK_RADICALS = ("w", "y")


def carries_root(form, root):
    """True when the letters of `form` hold the radicals of `root` (1st-order Ethiopic letters joined by "-") in
    order, each as written. A w or y radical may be missing (a hollow q-w-m form without its w), and so may a
    radical that repeats the one before it, since a doubled letter is written once (n-d-d with one d)."""
    letters = [geez_script.decompose_char(c)[0] for c in form]
    at, previous = 0, None
    for r in geez_script.root_symbols(root):
        if r in letters[at:]:
            at = letters.index(r, at) + 1
        elif r not in WEAK_RADICALS and r != previous:
            return False
        previous = r
    return True


def read_words_without_root(path=WORDS_WITHOUT_ROOT_FILE):
    """{word: label} of the words-without-root file (tab-separated, header line: word, label, kind, note)."""
    with open(path, encoding="utf-8", newline="") as f:
        return {r["word"]: r["label"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}


def load_words_without_root(dataset_dir, stats=None, path=WORDS_WITHOUT_ROOT_FILE):
    """Root-only pairs of the words in the words-without-root file, in file order, each with root NO_ROOT
    (target "root: none"). A word is left out when it is a gold val/test form (or in the synthetic generator's
    blocked forms file), a gold val/test form up to spelling (folded with geez_script.fold_spelling and looked up
    in split_heldout_synthetic.gold_eval_forms_folded, the rule that drops near-duplicate synthetic lines), or a
    form of the gold train file whose row has a root. Rows of the gold train file that have no root are not read
    as pairs (load_gold_split), so they do not count here."""
    import generate_synthetic_data
    import split_heldout_synthetic

    stats = stats if stats is not None else collections.Counter()
    eval_forms = split_heldout_synthetic.gold_eval_forms(dataset_dir) | generate_synthetic_data.read_blocked_forms_file()
    folded_eval_forms = split_heldout_synthetic.gold_eval_forms_folded(dataset_dir)
    gold_train = gold_file_path(dataset_dir, "train")
    gold_train_forms = {p["form"] for p in load_gold_split(dataset_dir, "train")} if os.path.exists(gold_train) else set()
    with open(path, encoding="utf-8", newline="") as f:
        kinds = {r["word"]: r.get("kind") for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    out = []
    for word, label in read_words_without_root(path).items():
        if label != NO_ROOT:
            raise ValueError(f"{path}: word {word!r} has label {label!r}, expected {NO_ROOT!r}")
        if not is_geez_word(word):
            raise ValueError(f"{path}: {word!r} is not one Ge'ez word")
        if word in eval_forms:
            stats["not loaded: form is a gold val/test form"] += 1
            continue
        if geez_script.fold_spelling(word) in folded_eval_forms:
            stats[GOLD_FORM_UP_TO_SPELLING] += 1
            continue
        if word in gold_train_forms:
            stats["not loaded: form has a root in the gold train file"] += 1
            continue
        stats["loaded"] += 1
        out.append({"form": word, "root": NO_ROOT, "task": ROOT_ONLY_TASK, "kind": kinds.get(word)})
    return out


def names_greek_source(entry):
    """True when the parenthesis that opens the entry's definition holds GREEK_SOURCE_MARKER as a word."""
    m = re.search(r"\(([^()]*)\)", entry.get("amharic_definition") or "")
    if not m:
        return False
    group = " " + m.group(1)
    starts = (k.start() for k in re.finditer(re.escape(GREEK_SOURCE_MARKER), group))
    return any(group[i - 1] in " \u1361(" for i in starts)


def is_root_perfect(headword, root):
    """True when the headword is the root's letters, each in the 1st order (q-t-l written as qatala): the
    verb's 3ms perfect, not a noun or adjective of that root."""
    return headword == root.replace("-", "")


def held_out_synthetic_roots(dataset_dir):
    """The roots (1st-order Ethiopic) of the dataset's held-out synthetic lines, data/synthetic/ (the lines the split
    moved out of training for a gold val/test root, or the same root up to spelling); empty when the file is missing."""
    import split_heldout_synthetic

    path = split_heldout_synthetic.targets(dataset_dir)[2]
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {split_heldout_synthetic.line_root(json.loads(line)) for line in f if line.strip()}


def read_dictionary_entries(dataset_dir):
    path = os.path.join(dataset_dir, KIDANE_WOLD_FILE)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dictionary_pairs(dataset_dir, stats=None, include_forms=False, allowed_roots_only=True, radicals=None):
    """Headword-to-root pairs of the nouns and adjectives of data/lexicon/classical_dictionary_kidane_wold.jsonl,
    in file order. Each pair has task ROOT_ONLY_TASK, so its target_text is the root alone. It also keeps the
    entry's labels: type noun or adjective, number from the entry's "number" field or CITATION_NUMBER, gender
    only for an adjective whose entry has a "gender" field.

    An entry gives a pair when, in this order:
      1. its pos is noun or adjective (not a verb, name, particle, other word class or no pos);
      2. its headword is one dictionary word (headword_type not in NOT_A_WORD_ENTRY) of Ge'ez letters only, is
         not in the words-without-root file (WORDS_WITHOUT_ROOT_FILE), its id is not in UNVERIFIABLE_ENTRY_IDS,
         and its definition does not name a Greek source (names_greek_source);
      3. it has a root (dictionary_entry_root);
      4. the root is not marked unclear (root_marked_unclear);
      5. the root passes the gold check (root_problem) once written in Ge'ez letters, the headword holds
         its radicals in order (carries_root): a root whose letters are not in the headword was misread, and
         the headword is not the root's 3ms perfect (is_root_perfect);
      6. `radicals` (a set of radical counts, None for any) keeps only roots of those lengths; with allowed_roots_only: the root has 3 or 4 radicals and is on the synthetic generator's allow-list
         (generate_synthetic_data.ALLOWED_ROOTS_FILE), so a verb backs every root; a noun prefix (m-, t- or a-)
         read as a radical gives a root that fails here;
      7. the root is not a gold val/test root (split_heldout_synthetic.gold_eval_roots), not in the
         generator's held-out roots file (generate_synthetic_data.HELD_OUT_ROOTS_FILE), not a root of the
         dataset's held-out synthetic lines (held_out_synthetic_roots), nor any of these up to spelling (the
         roots folded with split_heldout_synthetic.fold_root: labiovelar letters plain, twin letters merged), and
         the form is not a gold val/test form (split_heldout_synthetic.gold_eval_forms, generate_synthetic_data.BLOCKED_FORMS_FILE);
      8. the form is not a form of the gold train file: the gold pair already trains it, with the book's labels;
      9. no other pair that passes steps 1-5 gives the same form another root;
     10. no earlier pair has the same form and target (the first entry's record_id is kept).
    With include_forms, the words of an entry's `forms` list are tried too, after its headword, when they are one
    Ge'ez word that holds the root's radicals in order (carries_root); the list mixes spellings, plurals and words
    of the definition, so it is off by default. Such pairs have from_forms True.

    `stats` (a Counter) gets "loaded" and one "not loaded: <reason>" count per pair left out, at the first step
    it fails; words of `forms` are counted under "forms: ...".

    A pair whose form is a gold val/test form only up to spelling is not left out here, so that the split by
    root (split_dictionary_pairs) does not depend on it: dictionary_train_eval drops it after the split.
    """
    import generate_synthetic_data
    import split_heldout_synthetic
    import verb_classes

    stats = stats if stats is not None else collections.Counter()
    fixes = verb_classes.root_fixes(dataset_dir)
    without_root = read_words_without_root()
    allowed = generate_synthetic_data.read_allowed_roots_file()
    held_out = generate_synthetic_data.read_held_out_roots_file()
    eval_roots = split_heldout_synthetic.gold_eval_roots(dataset_dir)
    eval_forms = split_heldout_synthetic.gold_eval_forms(dataset_dir) | generate_synthetic_data.read_blocked_forms_file()
    synthetic_held_out = held_out_synthetic_roots(dataset_dir)
    folded_held_out = {split_heldout_synthetic.fold_root(r) for r in
                       set(eval_roots) | synthetic_held_out | {geez_script.to_ethiopic_root("-".join(s)) for s in held_out}}
    gold_train = gold_file_path(dataset_dir, "train")
    gold_train_forms = {p["form"] for p in load_gold_split(dataset_dir, "train")} if os.path.exists(gold_train) else set()

    candidates = []          # (form, root, entry, from_forms) that pass steps 1-5
    for entry in read_dictionary_entries(dataset_dir):
        pos = entry.get("pos")
        if pos not in NOMINAL_TYPES:
            stats[f"not loaded: pos {pos or 'missing'}"] += 1
            continue
        headword = entry.get("headword_geez") or ""
        if entry.get("headword_type") in NOT_A_WORD_ENTRY:
            stats[f"not loaded: headword_type {entry['headword_type']}"] += 1
            continue
        if not is_geez_word(headword):
            stats["not loaded: headword is not one Ge'ez word"] += 1
            continue
        if headword in without_root:
            stats["not loaded: headword is a word without root"] += 1
            continue
        if entry.get("id") in UNVERIFIABLE_ENTRY_IDS:
            stats["not loaded: entry cannot be checked"] += 1
            continue
        if names_greek_source(entry):
            stats["not loaded: definition names a Greek source"] += 1
            continue
        value = dictionary_entry_root(entry, fixes)
        if not value:
            stats["not loaded: no root"] += 1
            continue
        if root_marked_unclear(entry, value):
            stats["not loaded: root marked unclear"] += 1
            continue
        root = geez_script.to_ethiopic_root(value)
        problem = root_problem(root)
        if problem:
            stats[f"not loaded: {problem}"] += 1
            continue
        if not carries_root(headword, root):
            stats["not loaded: headword does not hold the root's radicals"] += 1
            continue
        if is_root_perfect(headword, root):
            stats["not loaded: headword is the root's 3ms perfect"] += 1
            continue
        candidates.append((headword, root, entry, False))
        if include_forms:
            for w in entry.get("forms") or []:
                if w == headword:
                    continue
                if not is_geez_word(w):
                    stats["forms: not one Ge'ez word"] += 1
                elif not carries_root(w, root):
                    stats["forms: does not hold the root's radicals"] += 1
                else:
                    candidates.append((w, root, entry, True))

    roots_of_form = collections.defaultdict(set)
    for form, root, _, _ in candidates:
        roots_of_form[form].add(root)

    out, seen = [], {}
    for form, root, entry, from_forms in candidates:
        prefix = "forms: " if from_forms else "not loaded: "
        symbols = geez_script.root_symbols(root)
        if radicals is not None and len(symbols) not in radicals:
            stats[f"{prefix}root has {len(symbols)} radicals (not requested)"] += 1
            continue
        if allowed_roots_only and len(symbols) not in ROOT_RADICALS_KEPT:
            stats[f"{prefix}root has {len(symbols)} radicals"] += 1
            continue
        if allowed_roots_only and symbols not in allowed:
            stats[f"{prefix}root not on the synthetic allow-list"] += 1
            continue
        if geez_script.first_order_root(root) in eval_roots:
            stats[f"{prefix}root is a gold val/test root"] += 1
            continue
        if symbols in held_out:
            stats[f"{prefix}root is in the held-out roots file"] += 1
            continue
        if geez_script.first_order_root(root) in synthetic_held_out:
            stats[f"{prefix}root is a root of the held-out synthetic lines"] += 1
            continue
        if split_heldout_synthetic.fold_root(geez_script.first_order_root(root)) in folded_held_out:
            stats[f"{prefix}root is a gold val/test or held-out root up to spelling"] += 1
            continue
        if form in eval_forms:
            stats[f"{prefix}form is a gold val/test form"] += 1
            continue
        if form in gold_train_forms:
            stats[f"{prefix}form is in the gold train file"] += 1
            continue
        if len(roots_of_form[form]) > 1:
            stats[f"{prefix}headword has conflicting roots"] += 1
            continue
        pos = entry["pos"]
        pair = {"form": form, "root": root, "type": pos,
                "gender": (entry.get("gender") or "") if pos == "adjective" else "",
                "number": entry.get("number") or CITATION_NUMBER,
                "tense": None, "person": None, "stem": None, "object": None,
                "record_id": str(entry["id"]), "from_forms": from_forms, "task": ROOT_ONLY_TASK}
        key = (form, target_text(pair))
        if key in seen:
            seen[key]["record_ids"].append(pair["record_id"])
            stats[f"{prefix}duplicate (form, target)"] += 1
            continue
        pair["record_ids"] = [pair["record_id"]]
        seen[key] = pair
        stats["forms: loaded" if from_forms else "loaded"] += 1
        out.append(pair)
    stats["headwords with conflicting roots"] = sum(1 for r in roots_of_form.values() if len(r) > 1)
    return out


def drop_gold_forms_up_to_spelling(pairs, folded_eval_forms, stats=None, reason=None):
    """The pairs, in order, without those whose form folded with geez_script.fold_spelling is a key of
    folded_eval_forms (split_heldout_synthetic.gold_eval_forms_folded): the rule that drops near-duplicate
    synthetic lines. `stats` gets one `reason` count (default GOLD_FORM_UP_TO_SPELLING) per pair dropped."""
    stats = stats if stats is not None else collections.Counter()
    kept = []
    for p in pairs:
        if geez_script.fold_spelling(p["form"]) in folded_eval_forms:
            stats[reason or GOLD_FORM_UP_TO_SPELLING] += 1
        else:
            kept.append(p)
    return kept


def dictionary_train_eval(dataset_dir, pairs, stats=None):
    """(train, eval) of the dictionary pairs: split_dictionary_pairs on all the pairs, then a pair whose form is a
    gold val/test form up to spelling is dropped from either side (drop_gold_forms_up_to_spelling). The split
    comes first, so dropping a pair does not move other roots between train and eval. A pair dropped from eval
    is counted under GOLD_FORM_UP_TO_SPELLING_EVAL."""
    import split_heldout_synthetic

    train, evals = split_dictionary_pairs(pairs)
    folded_eval_forms = split_heldout_synthetic.gold_eval_forms_folded(dataset_dir)
    return (drop_gold_forms_up_to_spelling(train, folded_eval_forms, stats),
            drop_gold_forms_up_to_spelling(evals, folded_eval_forms, stats, GOLD_FORM_UP_TO_SPELLING_EVAL))


def split_dictionary_pairs(pairs, fraction=DICTIONARY_EVAL_FRACTION, seed=DICTIONARY_SPLIT_SEED):
    """(train, eval): whole roots go to eval, in a seeded random order of the roots, until eval holds `fraction`
    of the pairs, so no root is in both. Both keep the order of `pairs`."""
    by_root = collections.Counter(p["root"] for p in pairs)
    roots = sorted(by_root)
    random.Random(seed).shuffle(roots)
    eval_roots, n = set(), 0
    for r in roots:
        if n >= fraction * len(pairs):
            break
        eval_roots.add(r)
        n += by_root[r]
    return [p for p in pairs if p["root"] not in eval_roots], [p for p in pairs if p["root"] in eval_roots]


def load_dictionary_verbs(dataset_dir, stats=None):
    """Pairs of the dictionary's verb headwords that are their root's 3ms perfect (is_root_perfect), labelled
    'past | 3 | m | s | basic', in file order. Only the checks of steps 2-5 of load_dictionary_pairs run here; the
    held-out roots and forms are dropped later by the training mixture's leak gate."""
    import verb_classes

    stats = stats if stats is not None else collections.Counter()
    fixes = verb_classes.root_fixes(dataset_dir)
    without_root = read_words_without_root()
    out, seen = [], set()
    for entry in read_dictionary_entries(dataset_dir):
        if entry.get("pos") != "verb":
            continue
        headword = entry.get("headword_geez") or ""
        if entry.get("headword_type") in NOT_A_WORD_ENTRY or not is_geez_word(headword) or headword in without_root:
            stats["not loaded: headword is not one Ge'ez word"] += 1
            continue
        value = dictionary_entry_root(entry, fixes)
        if not value or root_marked_unclear(entry, value):
            stats["not loaded: no settled root"] += 1
            continue
        root = geez_script.to_ethiopic_root(value)
        if root_problem(root) or not is_root_perfect(headword, root):
            stats["not loaded: headword is not the root's 3ms perfect"] += 1
            continue
        if headword in seen:
            stats["not loaded: duplicate headword"] += 1
            continue
        seen.add(headword)
        stats["loaded"] += 1
        out.append({"form": headword, "root": root, "type": "verb", "tense": "past", "person": "3", "gender": "m",
                    "number": "s", "stem": "basic", "object": "", "record_id": str(entry["id"])})
    return out
