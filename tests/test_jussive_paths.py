"""Tense labels: the gold files carry "subjunctive", and the loader reads tense as written
(no load-time mapping from "jussive")."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dataset_loader as G  # noqa: E402


def _rec(tense):
    return {"record_id": "aaaaaaaaaaaa", "root": "ቀ-ተ-ለ", "stem": "basic", "page_number": 30, "table_data": [
        {"geez_form": "ይቅትል", "tense": tense, "person": "3", "gender": "m", "number": "s"}]}


def test_tense_read_as_written():
    (p,) = G.file_pairs([_rec("subjunctive")], "train")
    assert p["tense"] == "subjunctive"
    assert "tense: subjunctive" in G.target_text(p)


def test_no_jussive_mapping():
    (p,) = G.file_pairs([_rec("jussive")], "train")
    assert p["tense"] == "jussive"                       # not mapped
    assert "tense: jussive" in G.target_text(p)          # target_text does not map either
