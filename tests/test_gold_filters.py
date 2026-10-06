"""gold_filters: load-time root corrections shared by the trainer and evaluate_gold.py."""
import collections
import json
import os
import sys
import tempfile

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)

import geez_config as gc
import gold_filters as G

DATASET = os.path.normpath(os.path.join(HERE, "..", "geez-dataset-v1"))
GOLD_COLS = ["record_id", "split", "line", "gold_all_line", "merho_line", "page", "headword", "root",
             "action", "new_root", "forms", "pairs", "group", "reason", "source"]


def rec(page, lemma, root, forms, raw="x"):
    return {"page_number": page, "title_or_lemma": lemma, "raw_transcription": raw, "root": root,
            "category": "verb_conjugation",
            "table_data": [{"geez_form": f, "tense": "past", "person": "3", "gender": "m", "number": "s"} for f in forms]}


def write_tables(folder, gold_rows, excluded_rows=(), synthetic_rows=()):
    os.makedirs(folder, exist_ok=True)
    def w(name, cols, rows):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    w("gold_root_corrections.tsv", GOLD_COLS, gold_rows)
    w("gold_excluded_records.tsv", ["record_id", "split", "line", "row", "page", "headword", "root", "form",
                                    "category", "proposals_line", "reason", "source"], excluded_rows)
    w("synthetic_root_corrections.tsv", ["root", "root_ethiopic", "action", "new_root", "new_root_ethiopic",
                                         "pairs", "evidence", "reason", "source"], synthetic_rows)


def row_for(r, action, new_root="", forms="", group="g"):
    return dict(record_id=G.record_id(r), page=r["page_number"], headword=r["title_or_lemma"], root=r["root"],
                action=action, new_root=new_root, forms=forms, group=group)


def test_root_problem_guard():
    assert G.root_problem("ቀ-ተ-ለ") is None
    assert G.root_problem("") == "guard: empty root"
    assert G.root_problem("ከ-ሰ / ከ-ደ-ሰ") == "guard: joined root"
    assert "Latin" in G.root_problem("ቀ-t-ለ")
    assert G.root_problem("ክ-ር-ስ-ት-ይ-ነ") == "guard: root is not 2-5 letters"
    assert G.root_problem("በ") == "guard: root is not 2-5 letters"
    assert G.root_problem("ጨ-ቀ-ለ") == "guard: letter outside the Ge'ez table"


def test_skip_fix_split_keep_normalize_and_exclude():
    a = rec(1, "አ", "", ["ሀ"], "a")                       # skip
    b = rec(2, "ብ", "ለ-ከ-መ", ["አልኰመ"], "b")               # fix
    c = rec(3, "ስ", "ን-ስ-ሕ-ለ-ል / ን-ዝ-ሀ-ለ-ል", ["ያንስሕላል", "አንዝሀለለ", "ዝዝ"], "c")   # split
    d = rec(4, "ድ", "ን-ስ-ሕ", ["ነስሐ", "ይነስሕ (ይነስሕ)"], "d")  # normalize only, one form excluded
    e = rec(5, "ኤ", "q-t-l", ["ቀተለ"], "e")                 # Latin, no row: to_ethiopic_root
    with tempfile.TemporaryDirectory() as tmp:
        write_tables(tmp, [row_for(a, "skip", group="empty_root"), row_for(b, "fix", "ለ-ኰ-መ", group="language_fix"),
                           row_for(c, "split", forms="ያንስሕላል=ነ-ሰ-ሐ-ለ-ለ|አንዝሀለለ=ነ-ዘ-ሀ-ለ-ለ")],
                     [dict(record_id=G.record_id(d), row=1, form="ይነስሕ")],
                     [dict(root="t-r-p", action="fix", new_root="t-r-p2")])
        C = G.Corrections(tmp, required=True)
        stats = collections.Counter()
        out = C.gold_records([a, b, c, d, e], stats)
    roots = [r["root"] for r in out]
    assert roots == ["ለ-ኰ-መ", "ነ-ሰ-ሐ-ለ-ለ", "ነ-ዘ-ሀ-ለ-ለ", "ነ-ሰ-ሐ", "ቀ-ተ-ለ"]
    assert out[0]["root_as_extracted"] == "ለ-ከ-መ"
    assert [r["geez_form"] for r in out[2]["table_data"]] == ["አንዝሀለለ"]       # ዝዝ: not settled, dropped
    assert stats["records skipped: empty_root"] == 1 and stats["pairs skipped: split form not settled"] == 1
    assert stats["records normalized: first_order"] == 1
    excl = out[3]["table_data"][1]
    assert G.is_excluded(excl, "ይነስሕ (ይነስሕ") and not G.is_excluded(out[3]["table_data"][0], "ነስሐ")
    assert a["root"] == "" and b["root"] == "ለ-ከ-መ"                              # inputs not mutated


def test_table_mismatch_raises():
    r = rec(1, "ሀ", "ቀ-ተ-ለ", ["ቀተለ"])
    bad = row_for(r, "fix", "ቀ-ተ-ለ"); bad["root"] = "ቀ-ጠ-ለ"
    with tempfile.TemporaryDirectory() as tmp:
        write_tables(tmp, [bad])
        try:
            G.Corrections(tmp).gold_records([r])
        except ValueError:
            return
    raise AssertionError("expected ValueError")


def test_guard_skips_without_tables():
    with tempfile.TemporaryDirectory() as tmp:
        C = G.Corrections(os.path.join(tmp, "none"))
        assert not C.present and C.sha256 == {}
        stats = collections.Counter()
        out = C.gold_records([rec(1, "ሀ", "", ["ሀ"]), rec(2, "ለ", "ከ-ሰ / ከ-ደ-ሰ", ["ተከሰ"]), rec(3, "ቀ", "ቀ-ተ-ል", ["ቀተለ"])], stats)
        assert [r["root"] for r in out] == ["ቀ-ተ-ለ"]
        assert stats["records skipped: guard: empty root"] == 1 and stats["records skipped: guard: joined root"] == 1
        try:
            G.Corrections(os.path.join(tmp, "none"), required=True)
        except FileNotFoundError:
            return
    raise AssertionError("expected FileNotFoundError")


def test_synthetic_pe_fix():
    with tempfile.TemporaryDirectory() as tmp:
        write_tables(tmp, [], [], [dict(root="t-r-p", action="fix", new_root="t-r-p2"), dict(root="t-p-s", action="keep"),
                                   dict(root="q-r-h2", action="keep")])
        C = G.Corrections(tmp)
    d, ch = C.synthetic_record({"features": {"root": "t-r-p", "tense": "past"}, "surface_form": "ታረጰት"})
    assert ch and d["features"]["root"] == "t-r-p2" and d["surface_form"] == "ታረፐት"
    assert gc.to_ethiopic_root(d["features"]["root"]) == "ተ-ረ-ፐ"
    d, ch = C.synthetic_record({"analysis": {"root": "t-p-s"}, "surface_form": "ተጰሰ"})
    assert not ch and d["surface_form"] == "ተጰሰ"
    d, ch = C.synthetic_record({"features": {"root": "q-r-h2"}, "surface_form": "ቃርሐ"})
    assert not ch and d["surface_form"] == "ቃርሐ"                 # read as written (no skips)
    with tempfile.TemporaryDirectory() as tmp:
        write_tables(tmp, [], [], [dict(root="q-r-h2", action="skip")])
        try:
            G.Corrections(tmp)
            assert False, "a skip row must be rejected"
        except ValueError as e:
            assert "not supported" in str(e)


def test_trainer_reads_synthetic_as_written():
    import train_byt5_morphology as T
    with tempfile.TemporaryDirectory() as tmp:
        para = os.path.join(tmp, "data", "paradigms"); os.makedirs(para)
        with open(os.path.join(para, "synthetic_morphology_train.jsonl"), "w", encoding="utf-8") as f:
            for root, form in (("q-r-h2", "ቃርሐ"), ("t-r-p", "ታረጰ"), ("q-t-l", "ቀተለ")):
                f.write(json.dumps({"surface_form": form, "features": {"root": root, "tense": "past", "person": "3",
                                    "gender": "m", "number": "s", "stem": "basic"}}, ensure_ascii=False) + "\n")
        write_tables(os.path.join(tmp, "data", "corrections"), [], [],
                     [dict(root="t-r-p", action="fix", new_root="t-r-p2"), dict(root="q-r-h2", action="keep")])
        stats = collections.Counter()
        samples = T.load_data(tmp, stats=stats)
    assert sorted(x["input_text"] for x in samples) == ["analyze: ቀተለ", "analyze: ቃርሐ", "analyze: ታረፐ"]
    assert stats["synthetic pairs fixed"] == 1 and not any(k.startswith("synthetic pairs skipped") for k in stats)


EXTRACTED_REF = "544271a"    # dataset commit with the gold files as extracted (before the in-place rewrite)
SPLIT_FILES = {"train": "merho_ges_gold_train.jsonl", "val": "merho_ges_gold_val.jsonl", "test": "merho_ges_gold_test.jsonl"}
# their names at EXTRACTED_REF (the train file the trainer read was merho_ges_gold_train_tense_filled.jsonl)
EXTRACTED_FILES = {"train": "merho_ges_gold_train_tense_filled.jsonl", "val": "merho_ges_gold_val.jsonl",
                   "test": "merho_ges_gold_test.jsonl"}


def extracted_records(name):
    """The gold file as extracted (git EXTRACTED_REF), or None without the dataset checkout."""
    import subprocess
    r = subprocess.run(["git", "-C", DATASET, "show", f"{EXTRACTED_REF}:data/paradigms/{name}"], capture_output=True)
    return [json.loads(l) for l in r.stdout.decode("utf-8").splitlines()] if r.returncode == 0 else None


def test_real_tables_on_the_dataset():
    path = os.path.join(DATASET, "data", "corrections", "gold_root_corrections.tsv")
    recs = extracted_records(EXTRACTED_FILES["train"]) if os.path.exists(path) else None
    if recs is None:
        return                                   # dataset checkout not next to the model repo
    # the root tables describe the files as extracted: load-time mode on the extracted train file
    C = G.Corrections.from_dataset(DATASET, required=True, page_dir=None, in_place=False)
    stats = collections.Counter()
    out = C.gold_records(recs, stats)
    assert not any(k.startswith("records skipped: guard") for k in stats)
    for r in out:
        assert G.root_problem(r["root"]) is None
        assert r["root"] == gc.first_order_root(r["root"])
    roots = {r["root"] for r in out}
    assert "አ-ኰ-ተ" in roots and "ቈ-መ-ሐ" in roots and "ጐ-ነ-ጸ-ጸ" in roots
    assert stats["records skipped: empty_root"] == 33
    excluded = sum(G.is_excluded(row, v) for r in out for row in r["table_data"] for v in G.variants(row.get("geez_form")))
    assert excluded == 78


def test_clean_gold_files_on_the_dataset():
    """Gold files corrected in place: read as written, every loaded root valid and in one split only."""
    if not os.path.exists(os.path.join(DATASET, "data", "paradigms", G.CLEAN_FLAGGED)):
        return                                   # no dataset checkout, or gold files not rewritten
    C = G.Corrections.from_dataset(DATASET, required=True)
    assert C.in_place and C.page is None and "gold files corrected in place" in C.describe()
    split_of, n = {}, collections.Counter()
    for split, name in SPLIT_FILES.items():
        with open(os.path.join(DATASET, "data", "paradigms", name), encoding="utf-8") as f:
            recs = [json.loads(l) for l in f]
        stats = collections.Counter()
        pairs = C.gold_pairs(recs, split, stats)
        n[split] = len(pairs)
        # a flagged pair never loads (it may already be out for another reason, e.g. its row's root)
        assert stats["file pairs not loaded: flagged (merho_ges_gold_flagged_rows.tsv)"] <= sum(1 for k in C.flagged_file if k[0] == split)
        assert not {(split, p["record_id"], p["row"], p["form"]) for p in pairs} & set(C.flagged_file)
        for p in pairs:
            assert G.root_problem(p["root"]) is None and p["root"] == gc.first_order_root(p["root"])
            assert split_of.setdefault(p["root"], split) == split, (p["root"], split_of[p["root"]], split)
    assert len(C.flagged_file) >= 13 and n["train"] > 2000 and n["val"] > 250 and n["test"] > 350
    assert split_of["ዐ-ፀ-መ"] == "train" and split_of["አ-በ-ሐ"] == "test" and split_of["ከ-ረ-በ"] == "val"
