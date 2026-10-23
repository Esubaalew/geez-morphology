"""The unified rows (data/unified/geez_morphology_rows.jsonl): target_text for every word class,
analyzer.parse_neural_output for the same shapes, and load_unified on a small file."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyzer  # noqa: E402
import dataset_loader  # noqa: E402


def pair(**k):
    base = {"form": "x", "root": None, "type": "verb", "tense": None, "person": None, "gender": "", "number": "",
            "stem": None, "object": "", "possessor": "", "negated": False}
    base.update(k)
    return base


CASES = [
    # old shapes are unchanged
    (pair(root="ቀ-ተ-ለ", tense="past", person="3", gender="m", number="s", stem="basic"),
     "root: ቀ-ተ-ለ | tense: past | person: 3 | gender: m | number: s | stem: basic"),
    (pair(root="ቀ-ተ-ለ", type="noun", number="s"), "root: ቀ-ተ-ለ | type: noun | number: s"),
    (pair(root="ቀ-ተ-ለ", type="adjective", gender="m", number="p"), "root: ቀ-ተ-ለ | type: adjective | gender: m | number: p"),
    # additions to the old shapes
    (pair(root="ቀ-ተ-ለ", type="adjective", gender="m", number="s", possessor="3ms"),
     "root: ቀ-ተ-ለ | type: adjective | gender: m | number: s | possessor: 3ms"),
    (pair(root="ከ-ፈ-ለ", tense="past", person="3", gender="m", number="s", stem="basic", negated=True),
     "root: ከ-ፈ-ለ | tense: past | person: 3 | gender: m | number: s | stem: basic | negated: yes"),
    # new classes
    (pair(root="ቀ-ተ-ለ", type="infinitive", stem="basic"), "root: ቀ-ተ-ለ | type: infinitive | stem: basic"),
    (pair(root="ቀ-ተ-ለ", type="infinitive", stem="basic", possessor="3ms"),
     "root: ቀ-ተ-ለ | type: infinitive | stem: basic | possessor: 3ms"),
    (pair(root="ቀ-ተ-ለ", type="gerund", person="3", gender="m", number="s", stem="basic"),
     "root: ቀ-ተ-ለ | type: gerund | person: 3 | gender: m | number: s | stem: basic"),
    (pair(root="ቀ-ተ-ለ", type="nominal"), "root: ቀ-ተ-ለ | type: nominal"),
    (pair(root="ሕ-ል-መ", type="nominal", number="p"), "root: ሕ-ል-መ | type: nominal | number: p"),
    (pair(type="pronoun", person="3", gender="m", number="s"), "root: none | type: pronoun | person: 3 | gender: m | number: s"),
    (pair(type="preposition"), "root: none | type: preposition"),
    (pair(type="particle", possessor="3ms"), "root: none | type: particle | possessor: 3ms"),
]


@pytest.mark.parametrize("p,text", CASES)
def test_target_text(p, text):
    assert dataset_loader.target_text(p) == text


@pytest.mark.parametrize("p,text", CASES)
def test_target_text_parses_back(p, text):
    out = analyzer.parse_neural_output(text)
    assert out is not None, text
    assert out["root"] == p["root"]
    assert out["negated"] == bool(p["negated"])
    assert (out["possessor"] or "") == p["possessor"]
    if p["type"] != "verb":
        assert out["type"] == p["type"]


@pytest.mark.parametrize("text", [
    "root: none | type: noun | number: s",                       # a noun needs a root
    "root: ቀ-ተ-ለ | type: pronoun",                               # a pronoun has no root
    "root: ቀ-ተ-ለ | type: widget",                                # unknown class
    "root: none | type: preposition | stem: basic | mood: x",    # unknown key
    "root: ቀ-ተ-ለ | type: nominal | type: nominal",              # duplicate key
])
def test_malformed_generic_gives_none(text):
    assert analyzer.parse_neural_output(text) is None


def test_load_unified(tmp_path):
    rows = [
        {"id": "u1", "form": "ቀተለ", "root": "ቀ-ተ-ለ", "word_class": "verb", "stem": "basic", "tense": "past", "person": "3",
         "gender": "m", "number": "s", "object": None, "possessor": None, "negated": False, "source": "book", "book": "b",
         "page": 1, "status": "", "split": "train"},
        {"id": "u2", "form": "ቀተለ", "root": "ቀ-ተ-ለ", "word_class": "verb", "stem": "basic", "tense": "past", "person": "3",
         "gender": "m", "number": "s", "object": None, "possessor": None, "negated": False, "source": "gold", "book": "g",
         "page": 1, "status": "", "split": "train"},
        {"id": "u3", "form": "በ", "root": None, "word_class": "preposition", "stem": None, "tense": None, "person": None,
         "gender": None, "number": None, "object": None, "possessor": None, "negated": False, "source": "book", "book": "b",
         "page": 2, "status": "", "split": "train"},
        {"id": "u4", "form": "ምን", "root": None, "word_class": "noun", "stem": None, "tense": None, "person": None,
         "gender": None, "number": "s", "object": None, "possessor": None, "negated": False, "source": "book", "book": "b",
         "page": 2, "status": "CHECK", "split": "train"},
        {"id": "u5", "form": "ዕብ", "root": "ዐ-ብ-ዐ", "word_class": "nominal", "stem": None, "tense": None, "person": None,
         "gender": None, "number": None, "object": None, "possessor": None, "negated": False, "source": "book", "book": "b",
         "page": 3, "status": "", "split": "val"},
    ]
    d = tmp_path / "data" / "unified"
    d.mkdir(parents=True)
    with open(d / "geez_morphology_rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {}
    import collections
    stats = collections.Counter()
    tr = dataset_loader.load_unified(str(tmp_path), "train", stats=stats)
    assert [p["id"] for p in tr] == ["u1", "u3"]                       # book rows only, CHECK left out
    assert stats["not loaded: status CHECK"] == 1
    assert dataset_loader.target_text(tr[1]) == "root: none | type: preposition"
    both = dataset_loader.load_unified(str(tmp_path), "train", sources=("book", "gold"))
    assert [p["id"] for p in both] == ["u1", "u2", "u3"]
    assert [p["id"] for p in dataset_loader.load_unified(str(tmp_path), "val")] == ["u5"]
    assert dataset_loader.load_unified(str(tmp_path / "missing"), "train") == []


def test_load_unified_rootless_adjective(tmp_path):
    def adj(i, form):
        return {"id": i, "form": form, "root": None, "word_class": "adjective", "stem": None, "tense": None, "person": None,
                "gender": "m", "number": "s", "object": None, "possessor": None, "negated": False, "source": "book",
                "book": "b", "page": 1, "status": "", "split": "train"}
    d = tmp_path / "data" / "unified"
    d.mkdir(parents=True)
    with open(d / "geez_morphology_rows.jsonl", "w", encoding="utf-8") as f:
        for r in (adj("u1", "ኀነፋዊ"), adj("u2", "ሰማያዊ")):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (d / "rootless_derived.tsv").write_text("form\tbase\tref\nኀነፋዊ\tኀነፊ\tp\n", encoding="utf-8")
    import collections
    stats = collections.Counter()
    tr = dataset_loader.load_unified(str(tmp_path), "train", stats=stats)
    assert [p["id"] for p in tr] == ["u1"]                       # listed loads, unlisted is rejected
    assert tr[0]["root"] is None
    assert stats["not loaded: row has no root"] == 1
