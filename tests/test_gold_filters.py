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
                                   dict(root="q-r-h2", action="skip")])
        C = G.Corrections(tmp)
    d, ch = C.synthetic_record({"features": {"root": "t-r-p", "tense": "past"}, "surface_form": "ታረጰት"})
    assert ch and d["features"]["root"] == "t-r-p2" and d["surface_form"] == "ታረፐት"
    assert gc.to_ethiopic_root(d["features"]["root"]) == "ተ-ረ-ፐ"
    d, ch = C.synthetic_record({"analysis": {"root": "t-p-s"}, "surface_form": "ተጰሰ"})
    assert not ch and d["surface_form"] == "ተጰሰ"
    assert C.synthetic_record({"features": {"root": "q-r-h2"}, "surface_form": "ቃረሐ"}) == (None, True)


def test_trainer_skips_synthetic_root():
    import train_byt5_morphology as T
    with tempfile.TemporaryDirectory() as tmp:
        para = os.path.join(tmp, "data", "paradigms"); os.makedirs(para)
        with open(os.path.join(para, "synthetic_morphology_train.jsonl"), "w", encoding="utf-8") as f:
            for root, form in (("q-r-h2", "ቃረሐ"), ("t-r-p", "ታረጰ"), ("q-t-l", "ቀተለ")):
                f.write(json.dumps({"surface_form": form, "features": {"root": root, "tense": "past", "person": "3",
                                    "gender": "m", "number": "s", "stem": "basic"}}, ensure_ascii=False) + "\n")
        write_tables(os.path.join(tmp, "data", "corrections"), [], [],
                     [dict(root="t-r-p", action="fix", new_root="t-r-p2"), dict(root="q-r-h2", action="skip")])
        stats = collections.Counter()
        samples = T.load_data(tmp, stats=stats)
    assert sorted(x["input_text"] for x in samples) == ["analyze: ቀተለ", "analyze: ታረፐ"]
    assert stats["synthetic pairs skipped: q-r-h2"] == 1 and stats["synthetic pairs fixed"] == 1


def test_real_tables_on_the_dataset():
    path = os.path.join(DATASET, "data", "corrections", "gold_root_corrections.tsv")
    if not os.path.exists(path):
        return                                   # dataset checkout not next to the model repo
    C = G.Corrections.from_dataset(DATASET, required=True)
    stats = collections.Counter()
    with open(os.path.join(DATASET, "data", "paradigms", "merho_ges_gold_train_tense_filled.jsonl"), encoding="utf-8") as f:
        out = C.gold_records([json.loads(l) for l in f], stats)
    assert not any(k.startswith("records skipped: guard") for k in stats)
    for r in out:
        assert G.root_problem(r["root"]) is None
        assert r["root"] == gc.first_order_root(r["root"])
    roots = {r["root"] for r in out}
    assert "አ-ኰ-ተ" in roots and "ቈ-መ-ሐ" in roots and "ጐ-ነ-ጸ-ጸ" in roots
    assert stats["records skipped: empty_root"] == 33
    excluded = sum(G.is_excluded(row, v) for r in out for row in r["table_data"] for v in G.variants(row.get("geez_form")))
    assert excluded == 78
