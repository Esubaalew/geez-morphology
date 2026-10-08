"""gold_filters: the gold files are read as written; flagged rows and rows with no root do not load."""
import collections
import json
import os
import sys
import tempfile

import pytest

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)

import gold_filters as G  # noqa: E402



def rec(root, rows, rid="aaaaaaaaaaaa", **kw):
    return dict({"record_id": rid, "page_number": 1, "root": root, "stem": "basic", "category": "verb_conjugation",
                 "table_data": rows}, **kw)


def verb(form, **kw):
    return dict({"geez_form": form, "tense": "past", "person": "3", "gender": "m", "number": "s"}, **kw)


def test_root_problem():
    assert G.root_problem("ቀ-ተ-ለ") is None
    assert G.root_problem("") == "empty root"
    assert G.root_problem("ከ-ሰ / ከ-ደ-ሰ") == "two roots joined"
    assert "Latin" in G.root_problem("ቀ-t-ለ")
    assert G.root_problem("ክ-ር-ስ-ት-ይ-ነ") == "root is not 2-5 letters"
    assert G.root_problem("በ") == "root is not 2-5 letters"
    assert G.root_problem("ጨ-ቀ-ለ") == "letter outside the Ge'ez table"


def test_pairs_as_written():
    r = rec("ቀ-ተ-ለ", [verb("ቀተለ / ቀቲል"),
                       verb("ቀተልዎ", object="3ms"),
                       verb("ተቀትለ", stem="passive", root="ቀ-ተ-ለ"),
                       {"geez_form": "ቅቱል", "type": "adjective", "gender": "m", "number": "s"},
                       {"geez_form": "ቀትል", "type": "noun", "gender": "", "number": "s"}])
    pairs = G.file_pairs([r], "train")
    assert [p["form"] for p in pairs] == ["ቀተለ", "ቀቲል", "ቀተልዎ", "ተቀትለ", "ቅቱል", "ቀትል"]
    t = [G.target_text(p) for p in pairs]
    assert t[0] == "root: ቀ-ተ-ለ | tense: past | person: 3 | gender: m | number: s | stem: basic"
    assert t[2].endswith("| stem: basic | object: 3ms")
    assert "stem: passive" in t[3]
    assert t[4] == "root: ቀ-ተ-ለ | type: adjective | gender: m | number: s"
    assert t[5] == "root: ቀ-ተ-ለ | type: noun | number: s"
    assert pairs[4]["tense"] is None and pairs[4]["stem"] is None


def test_labels_are_not_changed():
    """No load-time mapping: jussive, causative_intensive and 4th-order root letters stay as written."""
    r = rec("ቀ-ተ-ለ", [verb("ያስተቅትል", tense="jussive")], stem="causative_intensive")
    (p,) = G.file_pairs([r], "train")
    assert (p["tense"], p["stem"]) == ("jussive", "causative_intensive")
    (p,) = G.file_pairs([rec("ሃ-ለ-ወ", [verb("ሃለወ")])], "train")
    assert p["root"] == "ሃ-ለ-ወ"


def test_flagged_and_rootless_rows_do_not_load():
    stats = collections.Counter()
    records = [rec("ቀ-ተ-ለ", [verb("ቀተለ / ቀቲል"), verb("ቀተልኩ")]),
               rec("", [verb("ነገረ"), verb("ዐቀበ", root="ዐ-ቀ-በ"), verb("two words")], rid="bbbbbbbbbbbb"),
               rec("ሰ-ነ-አ", [verb("ሰነአ"), verb("ሐረሰ", root="ሐ-ረ-ሰ")], rid="cccccccccccc", root_unsettled=True)]
    flagged = {("train", "aaaaaaaaaaaa", 0, "ቀቲል"), ("val", "aaaaaaaaaaaa", 1, "ቀተልኩ")}   # 2nd key: other split
    pairs = G.file_pairs(records, "train", flagged, stats)
    assert [p["form"] for p in pairs] == ["ቀተለ", "ቀተልኩ", "ዐቀበ", "ሐረሰ"]
    assert stats["not loaded: flagged (merho_ges_gold_flagged_rows.tsv)"] == 1
    assert stats["not loaded: row has no root"] == 3          # ነገረ, "two words", ሰነአ (record root unsettled)
    assert stats["loaded"] == 4


@pytest.mark.parametrize("root,form", [("ቀ-t-ለ", "ቀተለ"), ("ከ-ሰ / ከ-ደ-ሰ", "ከሰ"), ("ቀ-ተ-ለ", "ቀተለ ብእሴ")])
def test_unclean_rows_raise(root, form):
    with pytest.raises(ValueError):
        G.file_pairs([rec(root, [verb(form)])], "train")


def test_read_flagged_rows_and_load_split():
    with tempfile.TemporaryDirectory() as tmp:
        para = os.path.join(tmp, "data", "paradigms")
        os.makedirs(para)
        with open(os.path.join(para, "merho_ges_gold_val.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec("ቀ-ተ-ለ", [verb("ቀተለ"), verb("ቀተልኩ")]), ensure_ascii=False) + "\n")
        assert [p["form"] for p in G.load_gold_split(tmp, "val")] == ["ቀተለ", "ቀተልኩ"]
        with open(os.path.join(para, G.FLAGGED_ROWS_FILE), "w", encoding="utf-8") as f:
            f.write("split\tgold_file\tfile_line\trecord_id\trow\tform\treason\n")
            f.write("val\tmerho_ges_gold_val.jsonl\t1\taaaaaaaaaaaa\t1\tቀተልኩ\tnot this verb\n")
        assert G.read_flagged_rows(tmp) == {("val", "aaaaaaaaaaaa", 1, "ቀተልኩ")}
        assert [p["form"] for p in G.load_gold_split(tmp, "val")] == ["ቀተለ"]


def test_trainer_reads_synthetic_as_written():
    import train_byt5_morphology as T
    with tempfile.TemporaryDirectory() as tmp:
        para = os.path.join(tmp, "data", "paradigms"); os.makedirs(para)
        with open(os.path.join(para, "synthetic_morphology_train.jsonl"), "w", encoding="utf-8") as f:
            for root, form in (("q-r-h2", "ቃርሐ"), ("t-r-p", "ታረጰ"), ("q-t-l", "ቀተለ")):
                f.write(json.dumps({"surface_form": form, "features": {"root": root, "tense": "past", "person": "3",
                                    "gender": "m", "number": "s", "stem": "basic"}}, ensure_ascii=False) + "\n")
        samples = T.load_data(tmp)
    assert sorted(x["input_text"] for x in samples) == ["analyze: ቀተለ", "analyze: ቃርሐ", "analyze: ታረጰ"]
    assert any(x["target_text"].startswith("root: ተ-ረ-ጰ |") for x in samples)

