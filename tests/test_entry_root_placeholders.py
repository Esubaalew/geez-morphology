"""entry_root() never returns the placeholder CHECK or a null written as a string,
and a fix row with new_root null means "root unknown, don't guess".

An unsettled root is null in the dataset, with the reason in root_notes (dictionary entry 7546: inferred_root null,
kidane_wold_root_fixes new_root null). An entry with a fix row gets the fix's root or None; it never falls
through to inferred_root or root. Without a fix row, entry_root() skips a null, empty or placeholder value
(inferred_root, then root); with none left it returns None.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import verb_classes  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402

BAD = {"CHECK", "check", "null", "None", "none", ""}
DICTIONARY = os.path.join(DATASET_DIR, "data", "geez_amharic_dictionary.jsonl")


@pytest.mark.parametrize("placeholder", ["CHECK", "null", "None", "", "  "])
def test_placeholder_is_skipped(placeholder):
    assert verb_classes.entry_root({"id": 1, "inferred_root": placeholder, "root": None}, {}) is None
    assert verb_classes.entry_root({"id": 1, "inferred_root": placeholder, "root": "q-t-l"}, {}) == "q-t-l"
    assert verb_classes.entry_root({"id": 1, "inferred_root": placeholder, "root": None}, {1: "q-t-l"}) == "q-t-l"
    # a fix row holding a placeholder is a fix row: no root, and no fall-through
    assert verb_classes.entry_root({"id": 1, "inferred_root": "q-t-l", "root": None}, {1: placeholder}) is None


def test_null_fix_row_stops_there():
    """A fix row with new_root null: root unknown, even when inferred_root and root have a value."""
    assert verb_classes.entry_root({"id": 1, "inferred_root": "q-t-l", "root": None}, {1: None}) is None
    assert verb_classes.entry_root({"id": 1, "inferred_root": "q-t-l", "root": "q-t-l"}, {1: None}) is None
    assert verb_classes.entry_root({"id": 1, "inferred_root": None, "root": None}, {1: None}) is None


def test_no_fix_row_falls_through():
    assert verb_classes.entry_root({"id": 2, "inferred_root": "q-t-l", "root": None}, {1: None}) == "q-t-l"
    assert verb_classes.entry_root({"id": 2, "inferred_root": None, "root": "q-t-l"}, {}) == "q-t-l"
    assert verb_classes.entry_root({"id": 1}, {}) is None


def test_root_fixes_keeps_null_rows(tmp_path):
    lex = tmp_path / "data" / "lexicon"
    lex.mkdir(parents=True)
    (lex / "kidane_wold_root_fixes.jsonl").write_text(
        json.dumps({"id": 1, "old_root": "q-t", "new_root": None}) + "\n"
        + json.dumps({"id": 2, "old_root": "q-t", "new_root": "q-t-l"}) + "\n", encoding="utf-8")
    fixes = verb_classes.root_fixes(str(tmp_path))
    assert fixes == {1: None, 2: "q-t-l"}
    assert verb_classes.entry_root({"id": 1, "inferred_root": "q-t"}, fixes) is None


def test_fix_wins():
    assert verb_classes.entry_root({"id": 1, "inferred_root": "t-l", "root": None}, {1: "q-t-l"}) == "q-t-l"


@pytest.mark.skipif(not os.path.exists(DICTIONARY), reason="no geez-dataset checkout")
def test_no_dictionary_entry_gives_a_placeholder_root():
    fixes = verb_classes.root_fixes(DATASET_DIR)
    assert fixes and not {v for v in fixes.values() if isinstance(v, str) and v.strip() in BAD}
    assert sorted(i for i, v in fixes.items() if v is None) == [7546]   # the only "root unknown" fix row
    seen = 0
    with open(DICTIONARY, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            assert e.get("inferred_root") != "CHECK", e.get("id")
            r = verb_classes.entry_root(e, fixes)
            assert r is None or (isinstance(r, str) and r.strip() not in BAD), (e.get("id"), r)
            if e.get("id") == 7546:
                seen += 1
                assert r is None   # merged entry: no root (inferred_root and new_root null, CHECK in root_notes)
    assert seen == 1
