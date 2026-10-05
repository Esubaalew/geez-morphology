"""Load-time corrections for the gold and synthetic data.

The data files stay byte-identical. train_byt5_morphology.py and evaluate_gold.py both read the
tables in <dataset>/data/corrections/ through this module:

  gold_root_corrections.tsv       per gold record: skip | fix (new_root) | split (forms) | keep
  gold_excluded_records.tsv       table rows/forms left out of this run
  synthetic_root_corrections.tsv  synthetic roots: fix (new_root) | skip | keep

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


def record_id(rec):
    key = f"{rec.get('page_number')}|{rec.get('title_or_lemma')}|{rec.get('raw_transcription')}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def form_key(v):
    """A form without "(...)" notes, 4th-order gutturals as 1st (how the proposals name forms)."""
    return geez_config.normalize_guttural_4th(re.sub(r"\([^)]*\)?", " ", v or "").strip(" :፡።()[]፣,"))


def variants(form_raw):
    """The trainer's variants of a geez_form cell."""
    return [v.strip(" :፡።()[]") for v in (form_raw or "").split("/") if v.strip(" :፡።()[]")]


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

    def __init__(self, folder, required=False):
        self.folder = folder
        paths = {k: os.path.join(folder, v) for k, v in TABLES.items()}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing and required:
            raise FileNotFoundError(f"correction tables missing: {missing}")
        self.paths = {k: p for k, p in paths.items() if os.path.exists(p)}
        self.sha256 = {TABLES[k]: _sha256(p) for k, p in self.paths.items()}
        self.gold, self.excluded, self.synthetic = {}, collections.defaultdict(set), {}
        self.synthetic_skip = set()
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
                elif r["action"] == "skip":
                    self.synthetic_skip.add(r["root"])

    @classmethod
    def from_dataset(cls, dataset_dir, required=False):
        return cls(os.path.join(dataset_dir, CORRECTIONS_DIR), required=required)

    @property
    def present(self):
        return len(self.paths) == len(TABLES)

    # ------------------------------------------------------------------ gold
    def gold_records(self, records, stats=None):
        """Corrected records (a split record may become several). stats: Counter of
        'records/pairs <kind>: <group>' counts."""
        stats = stats if stats is not None else collections.Counter()
        out = []
        for rec in records:
            out.extend(self._one(rec, stats))
        return out

    def _one(self, rec, stats):
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
        if t:
            if t["root"] != orig or t["headword"] != (rec.get("title_or_lemma") or ""):
                raise ValueError(f"correction table does not match the data at record {rid} "
                                 f"(page {t['page']}): rebuild data/corrections")
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

    # ------------------------------------------------------------- synthetic
    def synthetic_record(self, d):
        """-> (record, changed). Fixes features/analysis root and the surface letters.
        A root with a skip row gives (None, True): the caller drops the pair."""
        feats = d.get("features") if "features" in d else d.get("analysis")
        if feats and feats.get("root") in self.synthetic_skip:
            return None, True
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
