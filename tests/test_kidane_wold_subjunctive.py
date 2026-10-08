"""Grammar decisions of 2026-10-05.

1. A Kidane Wold triple whose subjunctive repeats the imperfect with a 6th-order
   guttural middle radical (ጸሕፈ ፡ ይጽሕፍ ፡ ይጽሕፍ) is an OCR collapse: the subjunctive is
   ይጽሐፍ and the imperative ጽሐፍ; the present is the triple's ይጽሕፍ. type_d roots were
   already fixed by the Type D book rule.
2. Guttural first and middle radical: present keeps የ- and a 6th-order middle letter.
3./4. Derived stems and initial-weak roots are not changed.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import geez_config  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402
import generate_synthetic_fixed as gen  # noqa: E402
import verb_classes  # noqa: E402

DATASET = DATASET_DIR   # --dataset_dir, else $GEEZ_DATASET_DIR, else ../geez-dataset (conftest.py)
SYNTH = os.path.join(DATASET, "data", "paradigms", "synthetic_morphology_train.jsonl")


# Roots off the allow-list (data/synthetic_roots_allowed_6183c24_v2.txt, v2 decisions: no dictionary verb entry, gold
# root or Kidane Wold triple) have no synthetic lines any more; their cases are skipped.
ALLOWED = {"-".join(r) for r in gen.read_allowed_roots_file()}


def need_allowed(root):
    if root not in ALLOWED:
        pytest.skip(f"{root} is not on data/synthetic_roots_allowed_6183c24_v2.txt: no synthetic lines")


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
    dec = [geez_config.decompose_char(ch) for ch in surface]
    js = [n for n in range(1, len(dec)) if dec[n][0] == root[2] and dec[n - 1][0] == root[1]]
    return dec[js[-1] - 1][1] if js else None


def test_collapsed_triples_found():
    if not os.path.exists(DATASET):
        pytest.skip("no geez-dataset checkout")
    found = verb_classes.kw_collapsed_subjunctive(DATASET)
    names = {"-".join(r) for r in found}
    assert {"S-h2-f", "d-kh-l", "d-kh-n", "d-kh-r", "qh-h2-m", "s-a-l", "s-a-n"} <= names
    assert found[("S", "h2", "f")]["imperfect"] == ("I", "I")


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
            assert geez_config.decompose_char(s[1])[1] == "I" or s.startswith("ኢ"), s   # ይጽ-, not ይጸ-
        elif t in ("subjunctive", "imperative"):
            assert middle_order(s, root) == "a", s


@pytest.mark.parametrize("root", ["d-kh-l", "d-kh-n", "d-kh-r", "qh-h2-m", "s-a-l", "s-a-n"])
def test_type_d_consistent(synth, root):
    r = tuple(root.split("-"))
    rows = [x["surface_form"] for x in synth if x["features"]["root"] == root
            and x["features"]["tense"] in ("subjunctive", "imperative")]
    assert rows and all(middle_order(s, r) == "a" for s in rows)
    assert middle_order(form(synth, root, "present", "3ms"), r) == "I"


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
    assert form(synth, "w-h2-n", "present", "3ms") == "ይወሕን"
