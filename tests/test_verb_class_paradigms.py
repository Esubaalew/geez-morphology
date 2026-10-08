"""Checks for the class paradigms from the grammar check
(verb_class_paradigms.py) and their use in the generator and analyzer."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verb_class_paradigms as vcp  # noqa: E402
import verb_classes  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402


def test_templates_match_expected_tables():
    assert vcp.check()


@pytest.mark.parametrize("cls,lemma", list(vcp.EXPECTED))
def test_generator_rows_cover_expected_tables(cls, lemma):
    """paradigm_rows() (what the generator writes) gives every EXPECTED form."""
    cons = verb_classes.root_of_lemma(lemma)
    surfaces = {s for _, s in verb_classes.paradigm_rows(cons, cls)}
    for forms in vcp.EXPECTED[(cls, lemma)].values():
        for form in forms.split():
            assert form in surfaces, (cls, lemma, form)


def test_type_c_imperfect_keeps_both_labels():
    rows = verb_classes.paradigm_rows(("b", "r", "k"), "type_c")
    tenses = {f["tense"] for f, s in rows if s == "ይባርክ"}
    assert tenses == {"present", "subjunctive"}


def test_initial_w_drop_only_for_listed_roots():
    assert verb_classes.class_fits_root("initial_w_drop", ("w", "l", "d"))
    assert verb_classes.class_fits_root("initial_w_drop", ("w", "r", "d"))
    assert not verb_classes.class_fits_root("initial_w_drop", ("w", "q", "r"))


@pytest.fixture(scope="module")
def analyzer():
    if not os.path.exists(DATASET_DIR):
        pytest.skip("no geez-dataset checkout (the analyzer reads its dictionary)")
    from analyzer import HybridAnalyzer
    a = HybridAnalyzer(dataset_dir=DATASET_DIR, use_neural=False)  # rules only
    return a


# (surface, root, tense, person+gender+number) of the FIRST analysis
FIRST_ANALYSIS = [
    ("ቀደሰ", "q-d-s", "past", "3ms"),
    ("ይቄድስ", "q-d-s", "present", "3ms"),
    ("ይቀድስ", "q-d-s", "subjunctive", "3ms"),
    ("ቀድስ", "q-d-s", "imperative", "2ms"),
    ("ሰትየ", "s-t-y", "past", "3ms"),
    ("ሰተይከ", "s-t-y", "past", "2ms"),
    ("ይሰቲ", "s-t-y", "present", "3ms"),
    ("ይስተይ", "s-t-y", "subjunctive", "3ms"),
    ("ስተይ", "s-t-y", "imperative", "2ms"),
    ("ተለወ", "t-l-w", "past", "3ms"),
    ("ይተሉ", "t-l-w", "present", "3ms"),   # not q-t-l-like 3mp
    ("ይትሉ", "t-l-w", "subjunctive", "3ms"),
    ("ይቀትሉ", "q-t-l", "present", "3mp"),  # real strong 3mp still works
    ("ይለድ", "w-l-d", "subjunctive", "3ms"),
    ("ለድ", "w-l-d", "imperative", "2ms"),
    ("ይረድ", "w-r-d", "subjunctive", "3ms"),
    ("ረድ", "w-r-d", "imperative", "2ms"),
]


@pytest.mark.parametrize("surface,root,tense,pgn", FIRST_ANALYSIS)
def test_analyzer_first_reading(analyzer, surface, root, tense, pgn):
    r = analyzer.analyze(surface)[0]
    got = (r.root_string, r.tense, (r.person or "") + (r.gender or "") + (r.number or ""))
    assert got == (root, tense, pgn)


def test_analyzer_type_c_both_tenses(analyzer):
    tenses = {(r.root_string, r.tense) for r in analyzer.analyze("ይባርክ")}
    assert {("b-r-k", "present"), ("b-r-k", "subjunctive")} <= tenses


def test_no_jussive_label(analyzer):
    for w in ["ይቅትል", "ይቁም", "ይቀድስ", "ይባርክ"]:
        assert all(r.tense != "jussive" for r in analyzer.analyze(w))
