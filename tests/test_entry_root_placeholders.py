"""entry_root() never returns the placeholder CHECK or a null written as a string.

An unsettled root is null in the dataset, with the reason in root_notes (dictionary entry 7546: inferred_root null,
kidane_wold_root_fixes new_root null). entry_root() skips a null, empty or placeholder value and takes the next
source (fix, then inferred_root, then root); with none left it returns None.
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
    assert verb_classes.entry_root({"id": 1, "inferred_root": placeholder, "root": None}, {1: "q-t-l"}) == "q-t-l"
    assert verb_classes.entry_root({"id": 1, "inferred_root": "q-t-l", "root": None}, {1: placeholder}) == "q-t-l"


def test_null_fix_is_skipped():
    assert verb_classes.entry_root({"id": 1, "inferred_root": None, "root": None}, {1: None}) is None
    assert verb_classes.entry_root({"id": 1, "inferred_root": "q-t-l", "root": None}, {1: None}) == "q-t-l"
    assert verb_classes.entry_root({"id": 1}, {}) is None


def test_fix_wins():
    assert verb_classes.entry_root({"id": 1, "inferred_root": "t-l", "root": None}, {1: "q-t-l"}) == "q-t-l"


@pytest.mark.skipif(not os.path.exists(DICTIONARY), reason="no geez-dataset checkout")
def test_no_dictionary_entry_gives_a_placeholder_root():
    fixes = verb_classes.root_fixes(DATASET_DIR)
    assert fixes and not {v for v in fixes.values() if isinstance(v, str) and v.strip() in BAD}
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
