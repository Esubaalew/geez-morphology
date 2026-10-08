"""Kidane Wold triple parsing, and the templates checked against reference forms."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import geez_config  # noqa: E402
import kw_verb_triples as kw  # noqa: E402
from engine import InterdigitationEngine, slots_from_spec  # noqa: E402

TRIPLES = [
    # (perfect, imperfect, subjunctive, class, root)
    ("ቀተለ", "ይቀትል", "ይቅትል", "type_a", "q-t-l"),
    ("ቀደሰ", "ይቄድስ", "ይቀድስ", "type_b", "q-d-s"),
    ("ባረከ", "ይባርክ", "ይባርክ", "type_c", "b-r-k"),
    ("ተለወ", "ይተሉ", "ይትሉ", "final_w", "t-l-w"),
    ("ሀለወ", "ይሄሉ", "ይሄሉ", "type_b_final_w", "h1-l-w"),
    ("ሰትየ", "ይሰቲ", "ይስተይ", "final_y", "s-t-y"),
    ("በከየ", "ይበኪ", "ይብኪ", "final_y_other", "b-k-y"),
    ("ወለደ", "ይወልድ", "ይለድ", "initial_w_drop", "w-l-d"),
    ("ቆመ", "ይቀውም", "ይቁም", "hollow", "q-w-m"),
    ("ሐረደ", "የሐርድ", "ይሐርድ", "type_a", "h2-r-d"),
    ("ዐረፈ", "የዐርፍ", "ይዐርፍ", "type_a", "a2-r-f"),
]


@pytest.mark.parametrize("p,i,s,cls,root", TRIPLES)
def test_triple_class_and_root(p, i, s, cls, root):
    got_cls, got_root, _ = kw.classify(p, i, s)
    assert got_cls == cls
    assert "-".join(got_root) == root


def test_triple_found_in_entry_text():
    rec = {"id": 1, "pos": "verb", "headword_geez": "ቀተለ",
           "amharic_definition": "ቀተለ ፡ (ቀተለ ፡ ይቀትል ፡ ይቅትል ፡፡ ዐረ ፡ ቀተለ) ፡ ገደለ ።"}
    t = kw.parse_entry(rec)
    assert t["triple"] == ["ቀተለ", "ይቀትል", "ይቅትል"] and t["verb_class"] == "type_a"


def test_restore_dropped_letter():
    rec = {"headword_geez": "ሠልዐ", "inferred_root": "s2-l"}
    assert kw.restore_dropped(rec) == "s2-l-a2"
    assert kw.restore_dropped({"headword_geez": "ቀተለ", "inferred_root": "q-t-l"}) is None


ENG = InterdigitationEngine(renderer=geez_config.render_unit)


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def forms(patterns, root, stem, tense, pgn):
    out = set()
    for p in patterns:
        f = p["features"]
        key = (f.get("person") or "") + (f.get("gender") or "") + (f.get("number") or "")
        t = {"jussive": "subjunctive"}.get(f.get("tense"), f.get("tense"))
        if f.get("verb_type") in ("strong_A", "quadriliteral") and f.get("stem") == stem \
                and t == tense and key == pgn:
            out.add(ENG.generate(root, slots_from_spec(p["pattern_spec"]), stem=stem))
    return out


# Reference forms (3ms; imperative 2ms)
REFERENCE = [
    ("patterns.json", ("q", "t", "l"), "passive", "past", "3ms", "ተቀትለ"),
    ("patterns.json", ("q", "t", "l"), "passive", "present", "3ms", "ይትቀተል"),
    ("patterns.json", ("q", "t", "l"), "passive", "subjunctive", "3ms", "ይትቀተል"),
    ("patterns.json", ("q", "t", "l"), "passive", "imperative", "2ms", "ተቀተል"),
    ("patterns.json", ("q", "t", "l"), "causative", "past", "3ms", "አቅተለ"),
    ("patterns.json", ("q", "t", "l"), "causative", "present", "3ms", "ያቀትል"),
    ("patterns.json", ("q", "t", "l"), "causative", "subjunctive", "3ms", "ያቅትል"),
    ("patterns.json", ("q", "t", "l"), "causative", "imperative", "2ms", "አቅትል"),
    ("patterns.json", ("q", "t", "l"), "basic", "imperative", "2ms", "ቅትል"),
    ("patterns_quadriliteral.json", ("d", "n", "g", "S2"), "basic", "past", "3ms", "ደንገፀ"),
    ("patterns_quadriliteral.json", ("d", "n", "g", "S2"), "basic", "present", "3ms", "ይደነግፅ"),
    ("patterns_quadriliteral.json", ("d", "n", "g", "S2"), "basic", "subjunctive", "3ms", "ይደንግፅ"),
    ("patterns_quadriliteral.json", ("d", "n", "g", "S2"), "basic", "imperative", "2ms", "ደንግፅ"),
]


@pytest.mark.parametrize("fname,root,stem,tense,pgn,expected", REFERENCE)
def test_reference_forms(fname, root, stem, tense, pgn, expected):
    assert forms(load(fname), root, stem, tense, pgn) == {expected}


def test_passive_perfect_keeps_ተቀተል_before_consonant_suffix():
    assert forms(load("patterns.json"), ("q", "t", "l"), "passive", "past", "2ms") == {"ተቀተልከ"}
