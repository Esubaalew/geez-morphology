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
"""
import collections
import csv
import hashlib
import json
import os
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


# ------------------------------------------------------------------ targets

def target_text(pair):
    """Training target. Verbs: 'root | tense | person | gender | number | stem' plus ' | object: X'
    when the form carries an object suffix. Nouns: 'root | type: noun | number'. Adjectives:
    'root | type: adjective | gender | number' (gender left out when the book gives none)."""
    if pair["type"] in NOMINAL_TYPES:
        parts = [f"root: {pair['root']}", f"type: {pair['type']}"]
        if pair["type"] == "adjective" and pair.get("gender"):
            parts.append(f"gender: {pair['gender']}")
        parts.append(f"number: {pair['number']}")
        return " | ".join(parts)
    text = (f"root: {pair['root']} | tense: {pair['tense']} | person: {pair['person']} | gender: {pair['gender']} | "
            f"number: {pair['number']} | stem: {pair['stem']}")
    return text + (f" | object: {pair['object']}" if pair.get("object") else "")
