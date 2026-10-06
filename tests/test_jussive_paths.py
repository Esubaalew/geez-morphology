"""Tense jussive: the clean gold files carry "subjunctive" and file_pairs reads
tense as written (no load-time mapping); the extracted-file path (544271a) still maps jussive to
subjunctive at load time (see test_train_script.test_load_counts_and_targets)."""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gold_filters as G


def _rec(tense):
    return {"root": "ቀ-ተ-ለ", "stem": "basic", "page_number": 30, "table_data": [
        {"geez_form": "ይቅትል", "tense": tense, "person": "3", "gender": "m", "number": "s"}]}


def test_clean_path_reads_tense_as_written():
    stats = collections.Counter()
    (p,) = G.file_pairs([_rec("subjunctive")], "train", stats=stats)
    assert p["tense"] == "subjunctive"
    assert "tense: subjunctive" in G.target_text(p)
    assert stats["file pairs with tense jussive (read as written)"] == 0


def test_clean_path_has_no_jussive_mapping():
    stats = collections.Counter()
    (p,) = G.file_pairs([_rec("jussive")], "train", stats=stats)
    assert p["tense"] == "jussive"                       # not mapped on this path
    assert "tense: jussive" in G.target_text(p)          # target_text no longer maps either
    assert stats["file pairs with tense jussive (read as written)"] == 1


def test_extracted_plain_pairs_still_map():
    (p,) = G.plain_pairs([_rec("jussive")])
    assert p["tense"] == "subjunctive"
