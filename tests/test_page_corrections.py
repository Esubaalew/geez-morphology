"""the page corrections at load time (gold_filters.PageCorrections, Corrections.gold_pairs,
target_text), the nominal / object target shapes (analyzer.parse_neural_output) and evaluate_gold scoring."""
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import analyzer  # noqa: E402
import evaluate_gold as E  # noqa: E402
import gold_filters as G  # noqa: E402
import gold_page_table as T  # noqa: E402

TABLE_COLS = ["split", "line", "form", "root", "old_label", "status", "type", "tense", "person", "gender", "number",
              "object", "book_term", "note", "form_fix", "root_fix", "page_checked", "record_id", "row"]
V108_COLS = ["proposal_no", "outcome", "type", "gender", "number", "form_fix", "root_fix", "note",
             "stem_fix", "tense_fix", "person_fix"]
EXTRA_COLS = ["split", "gold_file", "file_line", "record_id", "row", "variant", "form", "form_fix", "root_fix",
              "label_fix", "page", "note"]


def record(root="ቀ-ተ-ለ", rows=None, lemma="ቀተለ", stem="basic"):
    rows = rows or [{"geez_form": "ቀተለ", "tense": "past", "person": "3", "gender": "m", "number": "s"},
                    {"geez_form": "ቀተልኩ", "tense": "past", "person": "1", "gender": "c", "number": "s"},
                    {"geez_form": "ቅቱል", "tense": "past", "person": "3", "gender": "m", "number": "s"}]
    return {"page_number": 50, "title_or_lemma": lemma, "raw_transcription": lemma + " ...", "root": root,
            "stem": stem, "category": "verb_conjugation", "table_data": rows}


def write(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(r.get(c, "") for c in cols) + "\n")


def table_row(rec, row, form, **kw):
    return dict({"split": "train", "form": form, "status": "confirmed", "record_id": G.record_id(rec), "row": str(row)}, **kw)


def corrections(table=(), extra=(), v108=(), proposals=(), excluded=()):
    """Corrections over a fresh temporary corrections dir (excluded table only) and page dir."""
    tmp = tempfile.mkdtemp()
    corr, page = os.path.join(tmp, "corrections"), os.path.join(tmp, "page")
    write(os.path.join(corr, "gold_excluded_records.tsv"), ["record_id", "row", "form"], list(excluded))
    for name, cols, rows in (("gold_page_table.tsv", TABLE_COLS, table), ("gold_extra_fixes.tsv", EXTRA_COLS, extra),
                             ("proposal_page_outcomes.tsv", V108_COLS, v108)):
        if rows:
            write(os.path.join(page, "sources", name), cols, list(rows))
    write(os.path.join(page, "keys", "proposal_keys.tsv"), ["proposal_no", "split", "record_id", "row", "form"],
          list(proposals))
    return G.Corrections(corr, required=False, page_dir=page)


def targets(corr, recs, split="train"):
    return {p["form"]: G.target_text(p) for p in corr.gold_pairs(recs, split)}


def test_no_page_corrections_keeps_the_old_targets():
    rec = record()
    t = targets(G.Corrections("/nonexistent", required=False), [rec])
    assert t["ቀተለ"] == "root: ቀ-ተ-ለ | tense: past | person: 3 | gender: m | number: s | stem: basic"
    assert len(t) == 3


def test_nominal_and_object_targets():
    rec = record()
    c = corrections(table=[table_row(rec, 2, "ቅቱል", type="adjective", gender="m", number="s"),
                       table_row(rec, 1, "ቀተልኩ", type="noun", number="p"),
                       table_row(rec, 0, "ቀተለ", type="verb", tense="past", person="3", gender="m", number="s",
                                 object="1cs")])
    t = targets(c, [rec])
    assert t["ቅቱል"] == "root: ቀ-ተ-ለ | type: adjective | gender: m | number: s"
    assert t["ቀተልኩ"] == "root: ቀ-ተ-ለ | type: noun | number: p"           # no gender in the book: left out
    assert t["ቀተለ"] == "root: ቀ-ተ-ለ | tense: past | person: 3 | gender: m | number: s | stem: basic | object: 1cs"


def test_flagged_pair_stays_out():
    rec = record()
    c = corrections(table=[table_row(rec, 2, "ቅቱል", status="flag")])
    assert "ቅቱል" not in targets(c, [rec])


def test_form_fix_needs_page_checked():
    rec = record()
    c = corrections(table=[table_row(rec, 2, "ቅቱል", type="adjective", gender="m", number="s", form_fix="ቅቱሉ",
                                 page_checked="yes"),
                       table_row(rec, 1, "ቀተልኩ", type="noun", number="s", form_fix="ቀተልከ")])
    t = targets(c, [rec])
    assert "ቅቱሉ" in t and "ቅቱል" not in t
    assert t["ቀተልኩ"].startswith("root: ቀ-ተ-ለ | tense: past")            # the unchecked row is not applied at all
    assert any("form_fix given without page_checked=yes" in m for _, _, _, ms in c.page.problems for m in ms)


def test_record_root_letter_correction_applies_to_every_row():
    rec = record(root="ቀ-ተ-ሰ")                                          # OCR ሰ for ለ
    c = corrections(table=[table_row(rec, 0, "ቀተለ", type="verb", tense="past", person="3", gender="m", number="s",
                                 root_fix="ቀ-ተ-ለ", page_checked="yes")])
    pairs = c.gold_pairs([rec], "train")
    assert {p["root"] for p in pairs} == {"ቀ-ተ-ለ"} and all(p["root_from_page"] for p in pairs)


def test_a_different_root_only_changes_its_own_pair():
    rec = record()
    c = corrections(table=[table_row(rec, 2, "ቅቱል", type="adjective", gender="m", number="s",
                                 root_fix="ገ-ዘ-ፈ", page_checked="yes")])
    roots = {p["form"]: p["root"] for p in c.gold_pairs([rec], "train")}
    assert roots == {"ቀተለ": "ቀ-ተ-ለ", "ቀተልኩ": "ቀ-ተ-ለ", "ቅቱል": "ገ-ዘ-ፈ"}


def test_split_record_restores_checked_pairs_only():
    rows = [{"geez_form": "መስነን / መቅርን / መቃርን", "tense": "", "person": "", "gender": "", "number": ""}]
    rec = record(root="s-n-n / q-r-n", rows=rows, lemma="ስነነ / ቀረነ")       # joined: the guard skips it
    assert G.Corrections("/nonexistent").gold_pairs([rec], "test") == []
    c = corrections(table=[dict(table_row(rec, 0, "መስነን", type="adjective", number="s", root_fix="ሰ-ነ-ነ",
                                      page_checked="yes"), split="test"),
                       dict(table_row(rec, 0, "መቅርን", type="adjective", number="s", root_fix="ቀ-ረ-ነ",
                                      page_checked="yes"), split="test")])
    roots = {p["form"]: p["root"] for p in c.gold_pairs([rec], "test")}
    assert roots == {"መስነን": "ሰ-ነ-ነ", "መቅርን": "ቀ-ረ-ነ"}                   # መቃርን: not checked, stays out


def test_excluded_pair_comes_back_only_when_page_checked():
    rec = record()
    ex = [{"record_id": G.record_id(rec), "row": "2", "form": "ቅቱል"}, {"record_id": G.record_id(rec), "row": "1", "form": "ቀተልኩ"}]
    c = corrections(excluded=ex,
                table=[table_row(rec, 2, "ቅቱል", type="adjective", gender="m", number="s", page_checked="yes"),
                       table_row(rec, 1, "ቀተልኩ", type="noun", number="s")])
    assert set(targets(c, [rec])) == {"ቀተለ", "ቅቱል"}


def test_sources_that_disagree_keep_the_pair_out():
    rec = record()
    c = corrections(table=[table_row(rec, 2, "ቅቱል", type="adjective", gender="m", number="s")],
                v108=[{"proposal_no": "7", "outcome": "checked", "type": "noun", "number": "s"}],
                proposals=[{"proposal_no": "7", "split": "train", "record_id": G.record_id(rec), "row": "2", "form": "ቅቱል"}])
    assert "ቅቱል" not in targets(c, [rec])
    (key, conflicts), = c.page.conflicts.items()
    assert key[3] == "ቅቱል" and conflicts[0][0] == "type"


def test_outcomes_keep_blank_labels_and_apply_fixes():
    rec = record(stem="passive")
    c = corrections(v108=[{"proposal_no": "1", "outcome": "checked", "type": "verb", "gender": "m", "number": "s",
                       "stem_fix": "causative", "person_fix": "3"},
                      {"proposal_no": "2", "outcome": "flagged"}],
                proposals=[{"proposal_no": "1", "split": "train", "record_id": G.record_id(rec), "row": "0", "form": "ቀተለ"},
                           {"proposal_no": "2", "split": "train", "record_id": G.record_id(rec), "row": "1", "form": "ቀተልኩ"}])
    t = targets(c, [rec])
    assert t["ቀተለ"] == "root: ቀ-ተ-ለ | tense: past | person: 3 | gender: m | number: s | stem: causative"
    assert "ቀተልኩ" not in t


def test_extra_source_keys_and_label_fix():
    rec = record()
    rid = G.record_id(rec)
    c = corrections(extra=[{"split": "train", "record_id": rid, "row": "0", "form": "ቀተለ", "form_fix": "ቀተሉ", "page": "50"},
                       {"split": "train", "record_id": rid, "row": "1", "form": "ቀተልኩ", "label_fix": "past 1cs", "page": "50"}])
    t = targets(c, [rec])
    assert "ቀተሉ" in t and t["ቀተሉ"].endswith("stem: basic")              # form fixed, gold labels kept
    assert any("label_fix" in m for _, _, _, ms in c.page.problems for m in ms)


@pytest.mark.parametrize("row, expect", [
    ({"status": "confirmed", "type": "noun", "number": "s", "tense": "past"}, "only verbs have tense"),
    ({"status": "confirmed", "type": "verb", "tense": "past", "person": "3", "gender": "x", "number": "s"}, "not a verb person"),
    ({"status": "confirmed", "type": "adjective", "number": "s", "root_fix": "ቁ-ተ-ለ", "page_checked": "yes"}, "1st order"),
    ({"status": "confirmed", "type": "adjective", "number": "s", "root_fix": "q-t-l", "page_checked": "yes"}, "Latin"),
    ({"status": "flag", "type": "noun"}, "flagged row has type"),
    ({"status": "maybe", "type": "noun", "number": "s"}, "not confirmed or flag"),
])
def test_entry_problems(row, expect):
    assert any(expect in m for m in G.page_entry_problems(row))


def test_record_root_rule():
    assert G.record_root_applies("ጠ-ቀ-ለ", "ጠ-ቀ-ሰ", True)
    assert not G.record_root_applies("ገ-ዘ-ፈ", "ዘ-ከ-ረ", True)
    assert not G.record_root_applies("መ-ነ-ደ-በ", "ነ-ደ-በ", True)
    assert G.record_root_applies("ሰ-ረ-ገ-ወ", "", False)
    assert T.missing_letters("ጠቀሰ", "ጠ-ቀ-ለ", "ጠ-ቀ-ሰ") == ["ለ"]
    assert T.missing_letters("ጠቀለቶ", "ጠ-ቀ-ለ", "ጠ-ቀ-ሰ") == []


def test_parse_nominal_and_object_targets():
    p = analyzer.parse_neural_output("root: ቀ-ተ-ለ | type: adjective | gender: f | number: s")
    assert p["type"] == "adjective" and p["gender"] == "f" and p["tense"] is None and p["stem"] is None
    p = analyzer.parse_neural_output("root: ቀ-ተ-ለ | type: noun | number: p")
    assert p["type"] == "noun" and p["gender"] is None
    p = analyzer.parse_neural_output("root: ቀ-ተ-ለ | tense: past | person: 3 | gender: m | number: s | stem: basic | object: 1cs")
    assert p["type"] == "verb" and p["object"] == "1cs"
    for bad in ("root: ቀ-ተ-ለ | type: verb | number: s",                   # a verb needs the six fields
                "root: ቀ-ተ-ለ | type: noun | number: s | tense: past",      # nominal with a verb field
                "root: ቀ-ተ-ለ | type: noun"):                              # no number
        assert analyzer.parse_neural_output(bad) is None


class _Fixed:
    def __init__(self, pred):
        self.pred = pred

    def predict(self, surface):
        return dict(self.pred)


def test_scoring_of_nominal_and_object_pairs():
    noun = {"surface": "ቅቱል", "root_keys": E.gold_root_keys("ቀ-ተ-ለ"), "verb_class": None,
            **E.normalize_row({"type": "noun", "number": "s"})}
    verb = {"surface": "ቀተለኒ", "root_keys": E.gold_root_keys("ቀ-ተ-ለ"), "verb_class": "Type A",
            **E.normalize_row({"type": "verb", "stem": "basic", "tense": "past", "person": "3", "gender": "m",
                               "number": "s", "object": "1cs"})}
    as_verb = {"root": "ቀ-ተ-ለ", **E.normalize_row({"stem": "basic", "tense": "past", "person": "3", "gender": "m",
                                                     "number": "s"}), "tier": "pattern", "raw": ""}
    totals, _ = E.score([noun, verb], _Fixed(as_verb))
    assert totals["type:noun"]["root"] == [1, 1] and totals["type:noun"]["type"] == [0, 1]
    assert totals["type:noun"]["exact"] == [0, 1] and totals["type:noun"]["tense"] == [0, 0]
    assert totals["type:verb"]["object"] == [0, 1] and totals["type:verb"]["exact"] == [0, 1]
    as_noun = {"root": "ቀ-ተ-ለ", **E.normalize_row({"type": "noun", "number": "s"}), "tier": "byt5", "raw": ""}
    totals, _ = E.score([noun], _Fixed(as_noun))
    assert totals["type:noun"]["exact"] == [1, 1]
