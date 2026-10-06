"""Validator rules that accept a note with a reason (level "ok") instead of listing it to check."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import gold_page_table as V


def test_one_letter_change():
    assert V.one_letter_change("ጠ-ቀ-ሰ", "ጠ-ቀ-ለ")
    assert not V.one_letter_change("መ-ለ-ሐ", "መ-ለ-ተ-ሐ")      # letter count differs
    assert not V.one_letter_change("ረ-ፈ-ፈ", "ረ-መ-መ")          # two letters
    assert not V.one_letter_change("", "ረ-መ-መ")


def test_explain_missing():
    assert "labialized" in V.explain_missing("ጐሐ", "ገ-ወ-ሐ", ["ገ", "ወ"])
    assert "weak radical" in V.explain_missing("አብሐ", "በ-ወ-ሐ", ["ወ"])
    assert "weak radical" in V.explain_missing("ተከሰ", "ከ-የ-ሰ", ["የ"])
    assert V.explain_missing("አራኅ", "ረ-ኀ-በ", ["በ"]) is None       # needs the page
