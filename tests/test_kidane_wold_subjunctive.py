"""Kidane Wold subjunctives written like the imperfect (docs/grammar_rules.md section 4).

1. A Kidane Wold triple whose subjunctive repeats the imperfect with a 6th-order
   guttural middle radical (ጸሕፈ ፡ ይጽሕፍ ፡ ይጽሕፍ) is an OCR collapse: the subjunctive is
   ይጽሐፍ and the imperative ጽሐፍ; the present is the triple's ይጽሕፍ. type_d roots follow
   their own rule (tests/test_type_d_subjunctive.py).
   In the corrected Kidane Wold transcription (dataset 47a773a) most of these triples are not
   collapsed: S-h2-f is ጸሐፈ ፡ ይጽሕፍ ፡ ይጽሐፍ, s-a-l and qh-h2-m are regular middle-guttural verbs, d-kh-l
   is off the allow-list. A triple that writes the middle-guttural forms gets the middle-guttural rule;
   test_patch_roots_follow_own_triple checks the 26 roots that changes.
2. Guttural first and middle radical: present keeps የ- and a 6th-order middle letter.
3./4. Derived stems and initial-weak roots are not changed.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import geez_script  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402
import generate_synthetic_data as gen  # noqa: E402
import verb_classes  # noqa: E402

DATASET = DATASET_DIR   # --dataset_dir, else $GEEZ_DATASET_DIR, else ../geez-dataset (conftest.py)
SYNTH = os.path.join(DATASET, "data", "paradigms", "synthetic_morphology_train.jsonl")


# Roots off the allow-list (data/synthetic_roots_allowed_v5.txt: no dictionary verb entry, gold root or Kidane Wold
# triple, or left out by hand) have no synthetic lines; their cases are skipped.
ALLOWED = {"-".join(r) for r in gen.read_allowed_roots_file()}


def need_allowed(root):
    if root not in ALLOWED:
        pytest.skip(f"{root} is not on data/synthetic_roots_allowed_v5.txt: no synthetic lines")


@pytest.fixture(scope="module")
def synth():
    if not os.path.exists(SYNTH):
        pytest.skip("synthetic data not generated")
    with open(SYNTH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def form(synth, root, tense, per, stem="basic"):
    for r in synth:
        f = r["features"]
        if (f["root"] == root and f["tense"] == tense and f.get("stem") == stem and "object" not in f
                and not f.get("negative") and f["person"] + (f.get("gender") or "") + f["number"] == per):
            return r["surface_form"]
    return None


def middle_order(surface, root):
    dec = [geez_script.decompose_char(ch) for ch in surface]
    js = [n for n in range(1, len(dec)) if dec[n][0] == root[2] and dec[n - 1][0] == root[1]]
    return dec[js[-1] - 1][1] if js else None


def test_collapsed_triples_found():
    if not os.path.exists(DATASET):
        pytest.skip("no geez-dataset checkout")
    found = verb_classes.kw_collapsed_subjunctive(DATASET)
    names = {"-".join(r) for r in found}
    # corrected Kidane Wold text: S-h2-f, d-kh-l, qh-h2-m, s-a-l and s-a-n are not collapsed
    assert names == {"T-a2-y", "d-kh-n", "d-kh-r", "f-kh-r", "n-kh-r", "z-kh-n", "z-kh-r"}
    assert not {"S-h2-f", "d-kh-l", "qh-h2-m", "s-a-l", "s-a-n"} & names
    assert found[("d", "kh", "n")]["imperfect"] == ("I", "I")


@pytest.mark.parametrize("tense,per,expect", [
    ("past", "3ms", "ጸሐፈ"),          # perfect not changed
    ("present", "3ms", "ይጽሕፍ"),      # from the triple (was ይጸሕፍ)
    ("present", "3fp", "ይጽሕፋ"),
    ("subjunctive", "3ms", "ይጽሐፍ"),  # was ይጽሕፍ (OCR collapse in the triple)
    ("subjunctive", "2fs", "ትጽሐፊ"),
    ("imperative", "2ms", "ጽሐፍ"),
    ("imperative", "2mp", "ጽሐፉ"),
])
def test_S_h2_f(synth, tense, per, expect):
    assert form(synth, "S-h2-f", tense, per) == expect


def test_S_h2_f_all_persons(synth):
    root = ("S", "h2", "f")
    rows = [r for r in synth if r["features"]["root"] == "S-h2-f" and r["features"].get("stem") == "basic"]
    for r in rows:
        t, s = r["features"]["tense"], r["surface_form"]
        if t == "present":
            assert middle_order(s, root) == "I", s
            assert geez_script.decompose_char(s[1])[1] == "I" or s.startswith("ኢ"), s   # ይጽ-, not ይጸ-
        elif t in ("subjunctive", "imperative"):
            assert middle_order(s, root) == "a", s


@pytest.mark.parametrize("root", ["d-kh-l", "d-kh-n", "d-kh-r", "s-a-n"])
def test_type_d_consistent(synth, root):
    need_allowed(root)                     # d-kh-l is off the list (spelling twin)
    r = tuple(root.split("-"))
    rows = [x["surface_form"] for x in synth if x["features"]["root"] == root
            and x["features"]["tense"] in ("subjunctive", "imperative")]
    assert rows and all(middle_order(s, r) == "a" for s in rows)
    assert middle_order(form(synth, root, "present", "3ms"), r) == "I"


def _w(*units):
    """Ethiopic word from (consonant symbol, vowel) units, so the test types no Ge'ez letters."""
    return "".join(geez_script.render_unit(c, v) for c, v in units)


@pytest.mark.parametrize("root", ["s-a-l"])   # qh-h2-m is off the allow-list (phantom root)
def test_regular_middle_guttural(synth, root):
    """Not type_d (s-a-l: perfect ሰአለ in the corrected text: a regular
    middle-guttural verb, ይስእል / ይስአል; qh-h2-m has no triple and stays Type A, its old one was an OCR misread,
    data/phantom_root_qh_h2_m.tsv): the middle-guttural forms in the basic stem."""
    need_allowed(root)
    c1, c2, c3 = root.split("-")
    assert form(synth, root, "past", "3ms") == _w((c1, "a"), (c2, "a"), (c3, "a"))
    assert form(synth, root, "present", "3ms") == _w(("y", "I"), (c1, "I"), (c2, "I"), (c3, "I"))
    assert form(synth, root, "subjunctive", "3ms") == _w(("y", "I"), (c1, "I"), (c2, "a"), (c3, "I"))
    assert form(synth, root, "imperative", "2ms") == _w((c1, "I"), (c2, "a"), (c3, "I"))
    r = (c1, c2, c3)
    rows = [x["surface_form"] for x in synth if x["features"]["root"] == root and x["features"].get("stem") == "basic"
            and x["features"]["tense"] in ("subjunctive", "imperative")]
    assert rows and all(middle_order(s, r) == "a" for s in rows)


# The 26 roots that get the middle-guttural rule from their own Kidane Wold triple, which writes
# ይስሕል ፡ ይስሐል and is class "unclear" (verb_classes.kw_middle_guttural_pattern).
PATCH_ROOTS = ["S-a2-q", "S-a2-r", "S-h2-b", "S-h2-f", "S-h2-m", "T-h2-n", "f-h1-q", "f-h2-l", "f-h2-q", "f-h2-r", "f-h2-s",
               "g-h2-f", "gw-h2-l", "l-h2-kh2", "m-a2-l", "m-a2-z", "n-a2-l", "n-h1-b", "q-h2-m", "r-h2-n", "s-a-l", "s-a2-r",
               "s-h2-T", "s-h2-b", "s-h2-l", "s2-a-b"]


@pytest.fixture(scope="module")
def synth_all(synth):
    """Training lines plus the lines held out for the gold val/test roots (g-h2-f is a held-out root)."""
    held = os.path.join(DATASET, "data", "synthetic", "synthetic_forms_for_val_test_roots.jsonl")
    if not os.path.exists(held):
        pytest.skip("held-out synthetic file not generated")
    with open(held, encoding="utf-8") as f:
        return synth + [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def triples():
    path = os.path.join(DATASET, "data", "lexicon", "kidane_wold_verb_triples.jsonl")
    if not os.path.exists(path):
        pytest.skip("no geez-dataset checkout")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x.get("root"):
                out.setdefault(x["root"], []).append(x)
    return out


def test_patch_roots_found(triples):
    assert {tuple(r.split("-")) for r in PATCH_ROOTS} <= verb_classes.kw_middle_guttural_pattern(DATASET)


# data/triple_check_exceptions.tsv, generator_kept: the book's m-a2-z
# subjunctive ends in the 1st order, a misprint for the 6th-order form the generator gives.
MISPRINT_SUBJUNCTIVE = {"m-a2-z": _w(("y", "I"), ("m", "I"), ("a2", "a"), ("z", "I"))}


@pytest.mark.parametrize("root", PATCH_ROOTS)
def test_patch_roots_follow_own_triple(synth_all, triples, root):
    need_allowed(root)
    xs = triples[root]
    assert form(synth_all, root, "present", "3ms") in {x["triple"][1] for x in xs}
    want = {MISPRINT_SUBJUNCTIVE[root]} if root in MISPRINT_SUBJUNCTIVE else {x["triple"][2] for x in xs}
    assert form(synth_all, root, "subjunctive", "3ms") in want


@pytest.mark.parametrize("root,pres", [("a-a-l", "የአእል"), ("a-a-n", "የአእን"), ("a-kh-s", "የአኅስ"),
                                       ("a-kh-z", "የአኅዝ"), ("h1-kh-t", "የሀኅት"), ("h2-kh-t", "የሐኅት")])
def test_doubly_guttural_present(synth, root, pres):
    need_allowed(root)
    assert form(synth, root, "present", "3ms") == pres
    r = tuple(root.split("-"))
    rows = [x["surface_form"] for x in synth if x["features"]["root"] == root
            and x["features"]["tense"] == "present" and x["features"].get("stem") == "basic"]
    assert rows and all(middle_order(s, r) == "I" for s in rows)


@pytest.mark.parametrize("root,tense,per,stem,expect", [
    ("m-h2-l", "present", "3ms", "passive", "ይትመሐል"),
    ("m-h2-l", "present", "3ms", "causative", "ያመሕል"),
])
def test_derived_stems_unchanged(synth, root, tense, per, stem, expect):
    need_allowed(root)
    assert form(synth, root, tense, per, stem) == expect


def test_initial_weak_unchanged(synth):
    assert form(synth, "w-h2-f", "present", "3ms") == "ይወሕፍ"
