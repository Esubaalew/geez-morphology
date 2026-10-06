"""Gold and synthetic data loading, with or without load-time corrections.

Gold files corrected in place (geez-dataset-v1 branch gold-clean-in-files and later): when
data/paradigms/merho_ges_gold_flagged_rows.tsv exists, the gold files already carry every fix
.
Then NO gold correction is applied here: Corrections.gold_pairs reads the files as written
(file_pairs) and only leaves out the pairs listed in data/paradigms/merho_ges_gold_flagged_rows.tsv
and forms/roots that cannot be a training pair (not one Ethiopic word, empty or joined root). The
synthetic table (synthetic_root_corrections.tsv) still applies until the synthetic file is rewritten.

The rest of this docstring is the load-time path, used for gold files as extracted (dataset main
before the rewrite): rule.

The data files stay byte-identical. train_byt5_morphology.py and evaluate_gold.py both read the
tables in <dataset>/data/corrections/ through this module:

  gold_root_corrections.tsv       per gold record: skip | fix (new_root) | split (forms) | keep
  gold_excluded_records.tsv       table rows/forms left out of this run
  synthetic_root_corrections.tsv  synthetic roots: fix (new_root) | keep (no skips: bad synthetic pairs are
                                  fixed in the synthetic files themselves)

and, on top of those, the page-checked corrections in this repo
(gold_corrections/sources/: her three files, verbatim; see PageCorrections and Corrections.gold_pairs):
per gold pair a new label (verb or noun/adjective), form_fix, root_fix; flagged rows stay out.

Order for each gold record:
  1. the table row for its record_id (sha1("page|headword|raw_transcription")[:12]), if any;
  2. every root letter -> 1st order of its own series (geez_config.first_order_root);
  3. guard: a root that is still empty, joined ("/" or ","), Latin, outside the Ge'ez letter table,
     or not 2-5 letters is skipped and logged.
Excluded forms are marked on the row ("_exclude"); callers test them with is_excluded().

Undo: run without the tables (train_byt5_morphology.py --allow_missing_corrections on a dataset
without data/corrections/, evaluate_gold.py --no_corrections). Steps 2-3 still apply in the trainer.
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

import geez_config  # noqa: E402

CORRECTIONS_DIR = os.path.join("data", "corrections")
TABLES = {"gold": "gold_root_corrections.tsv",
          "excluded": "gold_excluded_records.tsv",
          "synthetic": "synthetic_root_corrections.tsv"}
PE_TO_P2 = {0x1330 + i: 0x1350 + i for i in range(8)}   # ጰ-series -> ፐ-series
GOLD_SPLIT = re.compile(r"merho_ges_gold_(train|val|test)(_tense_filled)?\.jsonl$")
# Merho Ges "causative_intensive" mixes አስተ- forms (the synthetic "aste" stem) and reduplicated
# causatives (አንጸብረቀ): only አስተ- forms are labelled "aste". Used by the load-time path for the
# gold files as extracted; the files corrected in place carry stem "aste" themselves.
ASTE_FORM = re.compile(r"^(ኢ)?[አያታናእይትን]ስተ")

PAGE_DIR = os.path.join(SCRIPT_DIR, "gold_corrections")      # the page corrections (sources/, keys/)
PAGE_SOURCES = ("gold_page_table.tsv", "gold_extra_fixes.tsv", "proposal_page_outcomes.tsv")
# her list of gold pairs to keep out (split, gold_file, file_line, row, variant, form, page, reason):
# every listed pair stays out, whatever the other sources say
FLAGGED_SOURCE = "flagged_rows.tsv"
# gold files corrected in place: the list of pairs kept in the files but left out of training and
# scoring; the apply script always writes it, and its presence switches the loader to file_pairs
CLEAN_FLAGGED = "merho_ges_gold_flagged_rows.tsv"
TYPES = ("verb", "noun", "adjective")
NOMINAL_TYPES = ("noun", "adjective")
TENSES = ("past", "present", "subjunctive", "jussive", "imperative")
STEMS = ("basic", "passive", "causative", "reciprocal", "causative_intensive", "quadriliteral", "frequentative")
VERB_PERSONS = {("1", "c", "s"), ("1", "c", "p"), ("2", "m", "s"), ("2", "f", "s"), ("2", "m", "p"), ("2", "f", "p"),
                ("3", "m", "s"), ("3", "f", "s"), ("3", "m", "p"), ("3", "f", "p")}
OBJECTS = {"".join(p) for p in VERB_PERSONS}          # object suffix as person+gender+number: 3ms, 1cs, ...
LABEL_COLUMNS = ("type", "tense", "person", "gender", "number", "object", "stem")


def record_id(rec):
    """Stable id of a gold record. The corrected gold files store it ("record_id"): it was computed once
    from the values as extracted (OCR, dataset commit 544271a) and is never recomputed, so a corrected
    title_or_lemma or a merged record does not change it. Records without the
    field (the files as extracted) get it computed from those values: sha1(page|title_or_lemma|raw)[:12]."""
    if rec.get("record_id"):
        return rec["record_id"]
    key = f"{rec.get('page_number')}|{rec.get('title_or_lemma')}|{rec.get('raw_transcription')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def form_key(v):
    """A form without "(...)" notes, 4th-order gutturals as 1st (how the proposals name forms)."""
    return geez_config.normalize_guttural_4th(re.sub(r"\([^)]*\)?", " ", v or "").strip(" :፡።()[]፣,"))


def variants(form_raw):
    """Raw "/"-variants of a geez_form cell (used for the correction counts)."""
    return [v.strip(" :፡።()[]") for v in (form_raw or "").split("/") if v.strip(" :፡።()[]")]


def is_single_ethiopic_word(s):
    return bool(s) and all(0x1200 <= ord(c) <= 0x137F for c in s)


def clean_variants(form_raw):
    """Scored variants of a geez_form cell, as evaluate_gold.py has always read them: "(...)" notes
    dropped, split on "/", edge punctuation stripped, and only single Ethiopic words kept.
    Returns (kept, rejected); rejected are the non-empty variants that are not one Ethiopic word
    (bracketed notes left open, extra words, Latin)."""
    kept, rejected = [], []
    for v in re.sub(r"\([^)]*\)", " ", form_raw or "").split("/"):
        v = v.strip(" :፡።()[]፣,")
        if not v:
            continue
        (kept if is_single_ethiopic_word(v) else rejected).append(v)
    return kept, rejected


def split_of(path):
    m = GOLD_SPLIT.search(os.path.basename(path or ""))
    return m.group(1) if m else None


def root_problem(root):
    """Why a final root is not trainable, or None."""
    if not root:
        return "guard: empty root"
    if "/" in root or "," in root:
        return "guard: joined root"
    if any(not (0x1200 <= ord(c) <= 0x137F) and c != "-" for c in root):
        return "guard: Latin or other non-Ethiopic text in root"
    parts = root.split("-")
    if any(len(p) != 1 or geez_config.decompose_char(p)[0] is None for p in parts):
        return "guard: letter outside the Ge'ez table"
    if not 2 <= len(parts) <= 5:
        return "guard: root is not 2-5 letters"
    return None


def is_excluded(row, variant):
    return form_key(variant) in row.get("_exclude", ())


def _read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Corrections:
    """The three tables (any may be missing when required=False: then nothing is applied from it)."""

    def __init__(self, folder, required=False, page_dir=None, gold_dir=None, in_place=None):
        """gold_dir: where the gold files are (default <folder>/../paradigms). in_place: True/False
        forces the file mode / load-time mode; None = file mode when gold_dir has the change log."""
        self.folder = folder
        gold_dir = gold_dir or os.path.join(os.path.dirname(os.path.normpath(folder)), "paradigms")
        self.in_place = os.path.exists(os.path.join(gold_dir, CLEAN_FLAGGED)) if in_place is None else in_place
        self.flagged_file = {}
        if self.in_place:                  # the gold files carry the fixes: no gold table, no page corrections
            path = os.path.join(gold_dir, CLEAN_FLAGGED)
            if os.path.exists(path):
                self.flagged_file = {(r["split"], r["record_id"], int(r["row"]), r["form"]): r for r in _read_tsv(path)}
        paths = {k: os.path.join(folder, v) for k, v in TABLES.items()}
        if self.in_place:
            paths = {"synthetic": paths["synthetic"]}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing and required and not self.in_place:
            raise FileNotFoundError(f"correction tables missing: {missing}")
        self.paths = {k: p for k, p in paths.items() if os.path.exists(p)}
        self.sha256 = {TABLES[k]: _sha256(p) for k, p in self.paths.items()}
        self.gold, self.excluded, self.synthetic = {}, collections.defaultdict(set), {}
        if "gold" in self.paths:
            for r in _read_tsv(self.paths["gold"]):
                assert r["record_id"] not in self.gold, f"duplicate record_id {r['record_id']}"
                self.gold[r["record_id"]] = r
        if "excluded" in self.paths:
            for r in _read_tsv(self.paths["excluded"]):
                self.excluded[(r["record_id"], int(r["row"]))].add(form_key(r["form"]))
        if "synthetic" in self.paths:
            for r in _read_tsv(self.paths["synthetic"]):
                if r["action"] == "fix":
                    self.synthetic[r["root"]] = r["new_root"]
                elif r["action"] != "keep":
                    raise ValueError(f"synthetic_root_corrections.tsv: action {r['action']!r} for {r['root']} is not "
                                     "supported (fix or keep; skipped pairs are fixed in the synthetic file instead)")
        has_page = page_dir and not self.in_place and any(os.path.exists(os.path.join(page_dir, "sources", n))
                                                          for n in PAGE_SOURCES + (FLAGGED_SOURCE,))
        if self.in_place and os.path.exists(os.path.join(gold_dir, CLEAN_FLAGGED)):
            self.sha256[CLEAN_FLAGGED] = _sha256(os.path.join(gold_dir, CLEAN_FLAGGED))
        self.page = PageCorrections(page_dir, gold_dir) if has_page else None
        if self.page is not None:
            self.sha256.update({"sources/" + n: h for n, h in self.page.sha256.items()})

    @classmethod
    def from_dataset(cls, dataset_dir, required=False, page_dir=PAGE_DIR, in_place=None):
        return cls(os.path.join(dataset_dir, CORRECTIONS_DIR), required=required, page_dir=page_dir, in_place=in_place)

    @property
    def present(self):
        return self.in_place or len(self.paths) == len(TABLES)

    def describe(self):
        """One line for the logs: which gold loading is used."""
        if self.in_place:
            return (f"gold files corrected in place; no gold corrections at load, "
                    f"{len(self.flagged_file)} flagged pairs left out ({CLEAN_FLAGGED})")
        return "gold files as extracted; load-time corrections from " + self.folder + (
            f" and page corrections from {self.page.folder}/sources" if self.page else "")

    # ------------------------------------------------------------------ gold
    def gold_records(self, records, stats=None):
        """Corrected records (a split record may become several). stats: Counter of
        'records/pairs <kind>: <group>' counts."""
        stats = stats if stats is not None else collections.Counter()
        out = []
        for rec in records:
            out.extend(self._one(rec, stats))
        return out

    def _one(self, rec, stats, root_override=None):
        """root_override: a page-checked record root (PageCorrections.record_root); it replaces
        the root tables' decision for this record (skip, fix or split)."""
        rid = record_id(rec)
        orig = rec.get("root") or ""
        pairs = sum(len(variants(row.get("geez_form"))) for row in rec.get("table_data", []))
        stats["records loaded"] += 1
        stats["pairs loaded"] += pairs
        rows = []
        for j, row in enumerate(rec.get("table_data", [])):
            row = dict(row)
            ex = self.excluded.get((rid, j))
            if ex:
                row["_exclude"] = ex
            rows.append(row)
        groups = [(None, rows)]          # (root override, rows)
        t = self.gold.get(rid)
        if t and (t["root"] != orig or t["headword"] != (rec.get("title_or_lemma") or "")):
            raise ValueError(f"correction table does not match the data at record {rid} "
                             f"(page {t['page']}): rebuild data/corrections")
        if root_override is not None:
            groups = [(root_override, rows)]
            stats["records fixed: page_root_fix"] += 1
            stats["pairs fixed: page_root_fix"] += pairs
        elif t:
            action = t["action"]
            if action == "skip":
                stats["records skipped: " + t["group"]] += 1
                stats["pairs skipped: " + t["group"]] += pairs
                return []
            if action == "fix":
                groups = [(t["new_root"], rows)]
                stats["records fixed: " + t["group"]] += 1
                stats["pairs fixed: " + t["group"]] += pairs
            elif action == "split":
                by_form = {form_key(f): r for f, r in (x.split("=", 1) for x in t["forms"].split("|"))}
                parts = collections.OrderedDict()
                for row in rows:
                    roots = {by_form.get(form_key(v)) for v in variants(row.get("geez_form"))}
                    if len(roots) != 1 or None in roots:
                        stats["pairs skipped: split form not settled"] += len(variants(row.get("geez_form")))
                        continue
                    parts.setdefault(roots.pop(), []).append(row)
                groups = list(parts.items())
                stats["records split: split"] += 1
                stats["pairs split: split"] += sum(len(variants(r.get("geez_form"))) for g in parts.values() for r in g)
        out = []
        for new_root, part_rows in groups:
            source = new_root if new_root is not None else orig
            root = geez_config.first_order_root(source)
            # to_ethiopic_root would join "ከ-ሰ / ከ-ደ-ሰ" into one 5-letter root: check the source text too
            problem = root_problem(source if ("/" in source or "," in source) else root)
            if not problem and new_root is None and root != geez_config.to_ethiopic_root(orig):
                stats["records normalized: first_order"] += 1
                stats["pairs normalized: first_order"] += sum(len(variants(r.get("geez_form"))) for r in part_rows)
            if problem:
                n = sum(len(variants(r.get("geez_form"))) for r in part_rows)
                stats["records skipped: " + problem] += 1
                stats["pairs skipped: " + problem] += n
                print(f"[gold_filters] skipped page {rec.get('page_number')} {rec.get('title_or_lemma')!r} "
                      f"root {orig!r} -> {root!r}: {problem}")
                continue
            r = dict(rec, root=root, table_data=part_rows)
            if root != orig:
                r["root_as_extracted"] = orig
            out.append(r)
        return out

    # ------------------------------------------------------------ gold pairs
    def gold_pairs(self, records, split, stats=None):
        """The gold (form, labels) pairs the trainer and evaluate_gold use, in a fixed order.

        For each record: the root tables (gold_records) decide the root and which rows load; then the
        page corrections are applied per pair; see _pair for the order of rules.
        Each pair is a dict: form, source_form, root, root_from_page, root_as_extracted, type, stem, tense,
        person, gender, number, object, record (the corrected record), record_id, row, page (the table
        row or None).
        """
        stats = stats if stats is not None else collections.Counter()
        if self.in_place:
            return file_pairs(records, split, self.flagged_file, stats)
        out = []
        for rec in records:
            out.extend(self._pairs_of(rec, split, stats))
        return out

    def _pairs_of(self, rec, split, stats):
        rid = record_id(rec)
        raw_rows = rec.get("table_data", [])
        tagged = dict(rec, table_data=[dict(r, _row=j) for j, r in enumerate(raw_rows)])
        current, usable = self._current_root(rec, rid)
        root_override = self.page.record_root(split, rid, current, usable) if self.page else None
        # A record the root tables skip (empty, joined, Amharic-only letter...) comes back with a page
        # root: as a whole when she marked the root root_fix_scope=record, else pair by pair (checked pairs).
        unlocked = root_override is not None and not usable and not self.page.record_root_explicit(split, rid)
        if root_override is not None and not usable and not unlocked:
            stats["page records restored: record the root tables skip (root_fix_scope=record)"] += 1
        loaded = self._one(tagged, stats, root_override)
        by_row = {row["_row"]: (r, row) for r in loaded for row in r["table_data"]}
        # loaded rows in the order of the corrected records, then rows the root tables left out
        order = [row["_row"] for r in loaded for row in r["table_data"]]
        order += [j for j in range(len(raw_rows)) if j not in by_row]
        pairs = []
        for j in order:
            kept, rejected = clean_variants(raw_rows[j].get("geez_form"))
            for v, single in [(x, True) for x in kept] + [(x, False) for x in rejected]:
                entry = self.page.entry(split, rid, j, v) if self.page else None
                if unlocked and not (entry and entry["checked"]):
                    stats["page pairs still out: record root from the page, pair not page-checked"] += 1
                    continue
                p = self._pair(rec, rid, j, raw_rows[j], v, single, by_row.get(j), entry, stats)
                if p is not None and root_override is not None and not usable:
                    stats["page pairs restored: record the root tables skip (page record root)"] += 1
                if p is not None:
                    p["root_from_page"] = p["root_from_page"] or root_override is not None
                    pairs.append(p)
        return pairs

    def _current_root(self, rec, rid):
        """(root, usable): the record root the root tables give (1st order) and whether it loads."""
        t = self.gold.get(rid)
        if t and t["action"] == "split":
            return "", True
        source = t["new_root"] if t and t["action"] == "fix" else (rec.get("root") or "")
        joined = "/" in source or "," in source
        root = geez_config.first_order_root(source)
        return root, not (t and t["action"] == "skip") and root_problem(source if joined else root) is None

    def _pair(self, rec, rid, j, raw_row, v, single, loaded, entry, stats):
        """One variant -> pair dict or None. entry: the merged page entry for this pair, or None.
        Rules, in order:
        1. a flagged pair, or one whose sources disagree, stays out;
        2. root: a row the root tables left out comes back only with a page-checked root_fix;
           a page-checked root_fix replaces the root for this pair (the record-wide page root is
           already in `loaded`, see _pairs_of);
        3. form: the page form_fix (page-checked) or the variant; a variant that is not one Ethiopic
           word stays out unless a form_fix replaces it;
        4. a relabel-proposal exclusion is lifted only by a page-checked entry;
        5. labels: the gold row's, overridden by every label the page gives (None = keep)."""
        checked = entry is not None and entry["checked"]
        if entry is not None and entry["status"] in ("flag", "conflict"):
            stats["page pairs left out: " + ("flagged" if entry["status"] == "flag" else "sources disagree")] += 1
            return None
        restored = []
        root_fix = geez_config.first_order_root(entry["root_fix"]) if checked and entry["root_fix"] else None
        if loaded is None:
            if root_fix is None:
                return None                     # skipped by the root tables (counted there)
            base, row, root = rec, raw_row, root_fix
            restored.append("row the root tables left out")
        else:
            base, row = loaded
            root = root_fix or base["root"]
        form = entry["form_fix"] if checked and entry["form_fix"] else v
        if not single:
            if form == v:
                stats["gold variants dropped: notes or extra words (not one Ethiopic word)"] += 1
                return None
            restored.append("note-damaged form fixed")
        if form_key(v) in self.excluded.get((rid, j), ()):
            if not checked:
                stats["pairs excluded: pending (relabel proposals)"] += 1
                return None
            restored.append("relabel proposal (was excluded)")
        stem = base.get("stem", "basic")
        labels = {"type": "verb", "tense": row.get("tense", ""), "person": row.get("person", ""),
                  "gender": row.get("gender", ""), "number": row.get("number", ""), "object": "", "stem": stem}
        if entry is not None:                    # page labels; None = keep the gold label
            given = {k: x for k, x in entry["labels"].items() if x is not None}
            labels.update(given)
            stats["page pairs: labels from the page sources"] += bool(given)
        if labels["type"] in NOMINAL_TYPES:
            labels = dict(_nominal(labels), stem=None)
        elif not labels["stem"]:
            labels["stem"] = stem
        if labels["type"] == "verb" and labels["stem"] == "causative_intensive" and ASTE_FORM.match(form):
            labels["stem"] = "aste"
        tense_as_extracted = None
        if labels.get("tense") == "jussive":
            # load-time label convention of the extracted files (544271a): one subjunctive label, as in
            # the synthetic data. The clean files already carry "subjunctive" (file_pairs reads as written).
            labels["tense"], tense_as_extracted = "subjunctive", "jussive"
            stats["pairs: tense jussive read as subjunctive (load time)"] += 1
        if form != v:
            stats["page pairs: form_fix"] += 1
        if loaded is not None and root_fix is not None and root_fix != base["root"]:
            stats["page pairs: root_fix (row scope)"] += 1
        for why in restored:
            stats["page pairs restored: " + why] += 1
        return {"form": form, "source_form": v, "root": root, "root_from_page": root_fix is not None,
                "root_as_extracted": rec.get("root") or "", "record": base, "record_id": rid, "row": j,
                "page": entry, "tense_as_extracted": tense_as_extracted, **labels}

    # ------------------------------------------------------------- synthetic
    def synthetic_record(self, d):
        """-> (record, changed). Fixes features/analysis root and the surface letters."""
        feats = d.get("features") if "features" in d else d.get("analysis")
        if not feats or feats.get("root") not in self.synthetic:
            return d, False
        d = dict(d)
        new = dict(feats, root=self.synthetic[feats["root"]])
        d["features" if "features" in d else "analysis"] = new
        if "p2" in new["root"].split("-"):
            d["surface_form"] = d.get("surface_form", "").translate(PE_TO_P2)
        return d, True


def format_stats(stats):
    """Lines 'records/pairs' per kind, sorted."""
    keys = sorted({k.split(" ", 1)[1] for k in stats})
    return [f"  {k:55s} {stats.get('records ' + k, 0):6d} records {stats.get('pairs ' + k, 0):7d} pairs"
            for k in keys]


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
    # Tense as loaded: the clean files carry "subjunctive" (no jussive); the extracted-file path maps
    # jussive -> subjunctive at load time (Corrections._pair, plain_pairs).
    text = (f"root: {pair['root']} | tense: {pair['tense']} | person: {pair['person']} | gender: {pair['gender']} | "
            f"number: {pair['number']} | stem: {pair['stem']}")
    return text + (f" | object: {pair['object']}" if pair.get("object") else "")


def file_pairs(records, split, flagged=(), stats=None):
    """Pairs of gold files corrected in place, read as written (no tables, no page corrections).

    Record fields: root (1st order), stem; "root_unsettled": the record root is not usable (rows with
    their own root still load). Row fields: geez_form ("/"-variants), tense, person, gender, number,
    and where they differ from the record: root, stem, type (noun/adjective), object.
    Left out, and counted: pairs in `flagged` ((split, record_id, row, form) keys), variants that are
    not one Ethiopic word, rows whose root is empty, joined, Latin or not 2-5 letters.
    Tense is read as written: the files carry "subjunctive" where the book's extraction said "jussive"
    (no load-time mapping on this path; a "jussive" left in a file is counted and kept as is).
    Stems are read as written (the files carry "aste" for አስተ- forms; no derivation here)."""
    stats = stats if stats is not None else collections.Counter()
    out = []
    for rec in records:
        rid = record_id(rec)
        rec_root = "" if rec.get("root_unsettled") else (rec.get("root") or "")
        for j, row in enumerate(rec.get("table_data", [])):
            root = row.get("root") or rec_root
            kept, rejected = clean_variants(row.get("geez_form"))
            stats["file pairs not loaded: not one Ethiopic word (notes, phrase)"] += len(rejected)
            problem = root_problem(root) if not rec.get("root_unsettled") or row.get("root") else "root not settled"
            if problem:
                stats["file pairs not loaded: " + ("record root not settled" if problem == "root not settled"
                                                   else "root " + problem.replace("guard: ", ""))] += len(kept)
                continue
            for v in kept:
                if (split, rid, j, v) in flagged:
                    stats["file pairs not loaded: flagged (merho_ges_gold_flagged_rows.tsv)"] += 1
                    continue
                labels = {"type": row.get("type") or "verb", "tense": row.get("tense", ""), "person": row.get("person", ""),
                          "gender": row.get("gender", ""), "number": row.get("number", ""),
                          "object": row.get("object", ""), "stem": row.get("stem") or rec.get("stem", "basic")}
                if labels["type"] in NOMINAL_TYPES:
                    labels = dict(_nominal(labels), stem=None)
                stats["file pairs loaded"] += 1
                if labels.get("tense") == "jussive":
                    stats["file pairs with tense jussive (read as written)"] += 1
                out.append({"form": v, "source_form": v, "root": root, "root_from_page": bool(row.get("root")),
                            "root_as_extracted": root, "record": rec, "record_id": rid, "row": j, "page": None,
                            **labels})
    return out


def plain_pairs(records):
    """Pairs of the gold file as written (no tables): evaluate_gold.py --no_corrections."""
    out = []
    for rec in records:
        for j, row in enumerate(rec.get("table_data", [])):
            for v in clean_variants(row.get("geez_form"))[0]:
                stem = rec.get("stem", "basic")
                stem = "aste" if stem == "causative_intensive" and ASTE_FORM.match(v) else stem
                tense = row.get("tense", "")
                tense = "subjunctive" if tense == "jussive" else tense   # extracted files (544271a): one subjunctive label
                out.append({"form": v, "source_form": v, "root": rec.get("root") or "",
                            "root_as_extracted": rec.get("root") or "", "record": rec, "record_id": record_id(rec),
                            "row": j, "page": None, "root_from_page": False, "type": "verb", "stem": stem, "tense": tense,
                            "person": row.get("person", ""), "gender": row.get("gender", ""),
                            "number": row.get("number", ""), "object": ""})
    return out


# ------------------------------------------------------------ page corrections
#
# the page-checked corrections, read verbatim from gold_corrections/sources/:
#
#   gold_page_table.tsv       labels for loaded pairs (full label set: type, tense, person, gender,
#                                number, object [, stem]); form_fix, root_fix, page_checked. Keyed by
#                                split + line of the 7a4aca2 export, or by record_id + row (+ form).
#   gold_extra_fixes.tsv   other pairs: form_fix, root_fix (page in "page"). Keyed by record_id +
#                                row + form (gold_file, file_line, variant are cross-checked by the validator).
#   proposal_page_outcomes.tsv the 108 relabel proposals: outcome, type, gender, number, form_fix,
#                                root_fix and stem_fix / tense_fix / person_fix (blank = keep the gold
#                                label). Keyed by proposal_no (gold_corrections/keys/proposal_keys.tsv).
#
# Every source row becomes one normalized entry for a gold pair (split, record_id, row, form); the
# entries of one pair are merged; a pair whose sources disagree stays out (see PageCorrections).

EXPORT_KEYS = os.path.join("keys", "gold_pair_keys_7a4aca2.tsv")
PROPOSAL_KEYS = os.path.join("keys", "proposal_keys.tsv")
FIX_COLUMNS = ("form_fix", "root_fix", "root_fix_scope")


def _clean(e):
    return {k: (v or "").strip() for k, v in e.items() if k is not None}


def _labels(e):
    """Full label set of a relabel-table-style row (relabel table, extra table), or all None (keep the
    gold labels) when the row gives no type. The stem comes from stem_fix (or stem); blank = keep."""
    if not e.get("type"):
        return dict.fromkeys(LABEL_COLUMNS)
    labels = {c: e.get(c, "") for c in LABEL_COLUMNS if c != "stem"}
    labels["stem"] = e.get("stem_fix") or e.get("stem") or None
    return labels


def _with_stem(e):
    """The row as page_entry_problems reads it (stem_fix -> stem)."""
    return dict(e, stem=e.get("stem_fix") or e.get("stem", ""))


def _norm_table(e, keys):
    """gold_page_table.tsv row -> (key, entry, problems)."""
    key, probs = None, []
    if e.get("record_id") and e.get("row", "").isdigit():
        key = (e.get("split"), e["record_id"], int(e["row"]), e.get("form"))
    else:
        k = keys.get((e.get("split"), e.get("line")))
        if k is None:
            probs.append(f"{e.get('split')} line {e.get('line')!r} is not a pair of the 7a4aca2 export: give record_id + row")
        elif k[3] != e.get("form"):
            probs.append(f"{e.get('split')} line {e.get('line')} is {k[3]!r} in the 7a4aca2 export, not {e.get('form')!r}")
        else:
            key = k
    entry = {"status": e.get("status"), "checked": e.get("page_checked") == "yes", "labels": _labels(e),
             **{c: e.get(c, "") for c in FIX_COLUMNS}}
    return key, entry, probs + page_entry_problems(_with_stem(e))


def _nominal(labels):
    """Nouns and adjectives carry no tense, person, object or stem; nouns no gender either."""
    if labels["type"] in NOMINAL_TYPES:
        labels.update(tense="", person="", object="", stem="")
        if labels["type"] == "noun" or labels["gender"] is None:   # the verb's subject gender is no adjective gender
            labels["gender"] = ""
    return labels


def _norm_extra(e, keys):
    """gold_extra_fixes.tsv row: form_fix, root_fix, root_fix_scope and labels in the relabel
    table's columns (type, stem_fix, tense, person, gender, number, object), each one a change:
    blank = keep the gold label. A noun/adjective type clears the verb labels (_nominal).
    "page" (the scan page she read) marks the row page-checked."""
    probs = []
    key = (e.get("split"), e.get("record_id"), int(e["row"]), e.get("form")) if e.get("row", "").isdigit() else None
    if key is None or not e.get("record_id"):
        key = None
        probs.append("no record_id/row")
    if e.get("label_fix"):
        probs.append(f"label_fix {e['label_fix']!r} is not read: use the type/stem_fix/tense/person/gender/number/object columns")
    labels = {c: e.get(c) or None for c in LABEL_COLUMNS if c != "stem"}
    labels["stem"] = e.get("stem_fix") or e.get("stem") or None
    labels = _nominal(labels)
    status = e.get("status") or "confirmed"
    checked = bool(e.get("page")) or e.get("page_checked") == "yes"
    entry = {"status": status, "checked": checked, "labels": labels, **{c: e.get(c, "") for c in FIX_COLUMNS}}
    keep = {c: ("keep" if v is None else v) for c, v in labels.items()}
    return key, entry, probs + page_entry_problems(dict(e, **keep, status=status, page_checked="yes" if checked else ""))


def _norm_108(e, keys):
    """proposal_page_outcomes.tsv row: type, gender, number as given; tense/person/stem from the
    *_fix columns, blank = keep the gold label."""
    probs = []
    key = keys.get(e.get("proposal_no"))
    if key is None:
        probs.append(f"proposal_no {e.get('proposal_no')!r} is not one of the 108 proposals")
    outcome = e.get("outcome")
    status = {"checked": "confirmed", "flagged": "flag"}.get(outcome)
    if status is None:
        probs.append(f"outcome {outcome!r} is not checked or flagged")
    labels = dict.fromkeys(LABEL_COLUMNS)
    if status == "confirmed":
        labels.update(type=e.get("type", ""), gender=e.get("gender", ""), number=e.get("number", ""))
        for c in ("tense", "person", "stem"):
            labels[c] = e.get(c + "_fix") or None
        if labels["type"] in NOMINAL_TYPES:
            labels.update(tense="", person="", object="", stem="")
    entry = {"status": status, "checked": outcome == "checked", "labels": labels, **{c: e.get(c, "") for c in FIX_COLUMNS}}
    as_row = dict(e, status=status or "", page_checked="yes" if outcome == "checked" else "",
                  **{c: v or "" for c, v in labels.items() if c not in ("tense", "person", "stem")})
    as_row.update({c: labels[c] or ("keep" if labels["type"] == "verb" else "") for c in ("tense", "person", "stem")})
    if status == "flag":
        as_row.update(type="", gender="", number="")
    return key, entry, probs + page_entry_problems(as_row)


NORMALIZERS = {"gold_page_table.tsv": _norm_table, "gold_extra_fixes.tsv": _norm_extra,
               "proposal_page_outcomes.tsv": _norm_108}


def page_entry_problems(e):
    """Why a source row cannot be applied ([] = it can). "keep" in a label column means the source
    leaves that label as the gold has it. Used by the loader (rows with problems are not applied) and
    by scripts/gold_page_table.py validate."""
    g = lambda k: (e.get(k) or "").strip()
    probs = []
    status, checked = g("status"), g("page_checked")
    if status not in ("confirmed", "flag"):
        probs.append(f"status {status!r} is not confirmed or flag")
    if checked not in ("", "yes"):
        probs.append(f"page_checked {checked!r} is not empty or yes")
    for k in ("form_fix", "root_fix"):
        if g(k) and checked != "yes":
            probs.append(f"{k} given without page_checked=yes")
    if g("form_fix") and not is_single_ethiopic_word(g("form_fix")):
        probs.append(f"form_fix {g('form_fix')!r} is not one Ethiopic word")
    if g("root_fix"):
        why = root_problem(g("root_fix"))
        if why:
            probs.append(f"root_fix {g('root_fix')!r}: {why.replace('guard: ', '')}")
        elif g("root_fix") != geez_config.first_order_root(g("root_fix")):
            probs.append(f"root_fix {g('root_fix')!r} is not in 1st order (write {geez_config.first_order_root(g('root_fix'))})")
    if g("root_fix_scope") not in ("", "record", "row"):
        probs.append(f"root_fix_scope {g('root_fix_scope')!r} is not record or row")
    if status == "flag":
        filled = [k for k in LABEL_COLUMNS + ("form_fix",) if g(k) and g(k) != "keep"]
        if filled:
            probs.append("flagged row has " + ", ".join(filled) + " filled (a flagged row is never applied)")
        return probs
    t = g("type")
    ok = lambda k, allowed: g(k) in allowed or g(k) == "keep"
    if t == "keep":                           # labels given one by one (blank/"keep" = gold label)
        for k, allowed in (("tense", TENSES), ("stem", STEMS + ("aste",)), ("person", ("1", "2", "3")),
                           ("gender", ("m", "f", "c")), ("number", ("s", "p")), ("object", tuple(OBJECTS))):
            if g(k) not in ("", "keep") and g(k) not in allowed:
                probs.append(f"{k} {g(k)!r} is not one of {', '.join(sorted(allowed))}")
        return probs
    if t not in TYPES:
        probs.append(f"type {t!r} is not verb, noun or adjective")
    elif t == "verb":
        if not ok("tense", TENSES):
            probs.append(f"verb tense {g('tense')!r} is not one of {', '.join(TENSES)}")
        if "keep" not in (g("person"), g("gender"), g("number")) and (g("person"), g("gender"), g("number")) not in VERB_PERSONS:
            probs.append(f"person/gender/number {g('person')}{g('gender')}{g('number')} is not a verb person")
        if g("object") not in ("", "keep") and g("object") not in OBJECTS:
            probs.append(f"object {g('object')!r} is not a person like 3ms or 1cs")
        if g("stem") and not ok("stem", STEMS + ("aste",)):
            probs.append(f"stem {g('stem')!r} is not one of {', '.join(STEMS + ('aste',))}")
    else:
        for k in ("tense", "person", "object", "stem"):
            if g(k):
                probs.append(f"{t} with {k} {g(k)!r} (only verbs have {k})")
        if g("gender") not in ("", "m", "f"):
            probs.append(f"{t} gender {g('gender')!r} is not m, f or empty")
        if g("number") not in ("s", "p"):
            probs.append(f"{t} number {g('number')!r} is not s or p")
    return probs


def record_root_applies(fix, current, usable):
    """Does a page root_fix (without an explicit root_fix_scope) correct the whole record's root?
    Yes when the loader cannot use the record's root (skipped, joined, Latin...) or when the fix is a
    letter correction of it (same number of letters, at most one different: ጠ-ቀ-ሰ -> ጠ-ቀ-ለ).
    Otherwise (a different root such as a quoted verb, or a changed letter count) it is the row's own
    root, and the validator asks the maintainers to mark root_fix_scope=record if she means the record."""
    if not usable:
        return True
    a, b = fix.split("-"), (current or "").split("-")
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 1


class PageCorrections:
    """the three sources (gold_corrections/sources/), normalized per gold pair.

    .entries   (split, record_id, row, form) -> merged entry: status (confirmed/flag), checked,
               labels (value, or None = keep the gold label), form_fix, root_fix, root_fix_scope, sources
    .problems  [(source, line, form, [problems])]: rows not applied (bad values, no key)
    .conflicts {key: [(field, {value: [sources]})]}: pairs whose sources disagree; they stay out
    .record_fixes {(split, record_id): {root: [sources]}}: page-checked root_fix values that may be
               record-wide (not marked root_fix_scope=row); record_root() decides."""

    def __init__(self, folder, gold_dir=None):
        self.folder, self.gold_dir = folder, gold_dir
        self.sha256, self.rows = {}, collections.Counter()
        self.problems, self.conflicts = [], {}
        read = lambda p: _read_tsv(os.path.join(folder, p)) if os.path.exists(os.path.join(folder, p)) else []
        export = {(r["split"], r["export_line"]): (r["split"], r["record_id"], int(r["row"]), r["form"])
                  for r in read(EXPORT_KEYS) if r["export_line"]}
        proposals = {r["proposal_no"]: (r["split"], r["record_id"], int(r["row"]), r["form"]) for r in read(PROPOSAL_KEYS)}
        self.key_maps = key_maps = {"gold_page_table.tsv": export, "gold_extra_fixes.tsv": None,
                                    "proposal_page_outcomes.tsv": proposals}
        found = collections.defaultdict(list)            # key -> [(source, entry)]
        for name in PAGE_SOURCES:
            path = os.path.join(folder, "sources", name)
            if not os.path.exists(path):
                continue
            self.sha256[name] = _sha256(path)
            for line, e in enumerate(_read_tsv(path), 2):
                e = _clean(e)
                self.rows[name] += 1
                key, entry, probs = NORMALIZERS[name](e, key_maps[name])
                where = f"{name}:{line}"
                if probs or key is None:
                    self.problems.append((name, line, e.get("form", e.get("proposal_no", "")), probs or ["no key"]))
                    continue
                entry["sources"] = [where]
                found[key].append(entry)
        self.entries = {}
        for key, items in found.items():
            merged, conflicts = self._merge(items)
            if conflicts:
                self.conflicts[key] = conflicts
            self.entries[key] = merged
        self.flagged = self._read_flagged()           # key -> source line; these pairs stay out
        for key, where in self.flagged.items():
            e = self.entries.setdefault(key, {"sources": [], "status": "flag", "checked": False,
                                              "labels": {c: None for c in LABEL_COLUMNS},
                                              **{c: "" for c in FIX_COLUMNS}})
            e["status"] = "flag"
            e["sources"] = e["sources"] + [where]
        self.record_fixes = collections.defaultdict(lambda: collections.defaultdict(list))
        for (split, rid, _, _), e in self.entries.items():
            if e["root_fix"] and e["checked"] and e["root_fix_scope"] != "row" and e["status"] == "confirmed":
                self.record_fixes[(split, rid)][e["root_fix"]].extend(e["sources"])

    def _read_flagged(self):
        """flagged_rows.tsv -> {(split, record_id, row, form): "flagged_rows.tsv:<line>"}. The record is
        found by record_id when given, else by gold_file + file_line in the dataset's gold files; the
        form must be one of that row's variants. A row that does not resolve is a problem (not applied)."""
        path = os.path.join(self.folder, "sources", FLAGGED_SOURCE)
        if not os.path.exists(path):
            return {}
        self.sha256[FLAGGED_SOURCE] = _sha256(path)
        lines, out = {}, {}
        for line, e in enumerate(_read_tsv(path), 2):
            e = _clean(e)
            self.rows[FLAGGED_SOURCE] += 1
            probs, rec = [], None
            split, form = e.get("split", ""), e.get("form", "")
            if split not in ("train", "val", "test"):
                probs.append(f"split {split!r} is not train, val or test")
            try:
                row = int(e.get("row", ""))
            except ValueError:
                row = None
                probs.append(f"row {e.get('row')!r} is not a number")
            gf, fl = e.get("gold_file", ""), e.get("file_line", "")
            if gf and fl.isdigit() and self.gold_dir:
                if gf not in lines:
                    p = os.path.join(self.gold_dir, gf)
                    lines[gf] = open(p, encoding="utf-8").read().splitlines() if os.path.exists(p) else None
                ls = lines[gf]
                if ls is None:
                    probs.append(f"gold file {gf} not found")
                elif not 1 <= int(fl) <= len(ls):
                    probs.append(f"{gf} has no line {fl}")
                else:
                    rec = json.loads(ls[int(fl) - 1])
            rid = e.get("record_id") or (record_id(rec) if rec else "")
            if rec is not None and e.get("record_id") and record_id(rec) != e["record_id"]:
                probs.append(f"record_id {e['record_id']} is not the record at {gf}:{fl}")
            if not rid and not probs:
                probs.append("no record_id and no gold_file/file_line to find the record")
            if rec is not None and row is not None:
                cells = rec.get("table_data", [])
                if row >= len(cells):
                    probs.append(f"record has no row {row}")
                else:
                    kept, rejected = clean_variants(cells[row].get("geez_form"))
                    if form not in kept + rejected:
                        probs.append(f"form {form!r} is not a variant of row {row} ({' / '.join(kept + rejected)})")
            if probs:
                self.problems.append((FLAGGED_SOURCE, line, form, probs))
                continue
            out[(split, rid, row, form)] = f"{FLAGGED_SOURCE}:{line}"
        return out

    @staticmethod
    def _merge(items):
        """One pair's entries -> (merged entry, conflicts). A flag anywhere flags the pair. A fix
        ("" = none) or a label (None = keep the gold label; "" is a value: no gender, no object) that
        two sources give different values for is a conflict: the pair's status becomes "conflict"."""
        conflicts = []

        def pick(field, values):
            seen = collections.defaultdict(list)
            for v, src in values:
                seen[v].append(src)
            if len(seen) > 1:
                conflicts.append((field, dict(seen)))
            return next(iter(seen)) if len(seen) == 1 else None

        merged = {"sources": [s for i in items for s in i["sources"]],
                  "status": "flag" if any(i["status"] == "flag" for i in items) else "confirmed",
                  "checked": any(i["checked"] for i in items)}
        for c in FIX_COLUMNS:
            merged[c] = pick(c, [(i[c], i["sources"][0]) for i in items if i[c]]) or ""
        merged["labels"] = {c: pick(c, [(i["labels"][c], i["sources"][0]) for i in items if i["labels"][c] is not None])
                            for c in LABEL_COLUMNS}
        if conflicts and merged["status"] != "flag":
            merged["status"] = "conflict"
        return merged, conflicts

    def entry(self, split, rid, row, form):
        return self.entries.get((split, rid, row, form))

    def record_root_explicit(self, split, rid):
        """Did a source mark the record's (single) page root root_fix_scope=record?"""
        fixes = self.record_fixes.get((split, rid))
        if not fixes or len(fixes) > 1:
            return False
        root = next(iter(fixes))
        return any(e["root_fix_scope"] == "record" and e["root_fix"] == root
                   for k, e in self.entries.items() if k[:2] == (split, rid))

    def record_root(self, split, rid, current, usable):
        """The record-wide page root for this record, or None. current/usable: the root the root
        tables give the record and whether it loads. Several different values (a split record such as
        ስነነ / ቀረነ) give none: each pair then takes its own root_fix."""
        fixes = self.record_fixes.get((split, rid))
        if not fixes or len(fixes) > 1:
            return None
        root = next(iter(fixes))
        return root if self.record_root_explicit(split, rid) or record_root_applies(root, current, usable) else None
