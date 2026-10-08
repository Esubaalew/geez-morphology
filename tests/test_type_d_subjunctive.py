"""type_d subjunctive and imperative follow ፍሬ ግእዝ pp. 18, 63
(ሰአለ ይስእል ይስአል, ድሕነ ይድሕን ይድሐን, መሐለ ይምሕል ይምሐል): the guttural 2nd radical is
1st order before the last letter. Perfect and present stay as Kidane Wold writes them.
Root ወ-ዐ-አ is dropped (no clean Kidane Wold perfect)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import geez_config  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402
import verb_classes  # noqa: E402

DATASET = DATASET_DIR   # --dataset_dir, else $GEEZ_DATASET_DIR, else ../geez-dataset (conftest.py)
SYNTH = os.path.join(DATASET, "data", "paradigms", "synthetic_morphology_train.jsonl")


def rows_for(root, orders):
    return {(f["tense"], f["person"] + (f["gender"] or "") + f["number"]): form
            for f, form in verb_classes.paradigm_rows(root, "type_d", orders)}


@pytest.mark.parametrize("root,perf,pres,subj,imp", [
    (("s", "a", "l"), "ስእለ", "ይስእል", "ይስአል", "ስአል"),
    (("d", "kh", "n"), "ድኅነ", "ይድኅን", "ይድኀን", "ድኀን"),
    (("d", "kh", "l"), "ድኅለ", "ይድኅል", "ይድኀል", "ድኀል"),
    (("qh", "h2", "m"), "ቍሕመ", "ይቍሕም", "ይቍሐም", "ቍሐም"),
    (("s", "a", "n"), "ስእነ", "ይስእን", "ይስአን", "ስአን"),
])
def test_type_d_book_rule(root, perf, pres, subj, imp):
    orders = verb_classes.type_d_orders(DATASET).get(root)
    if orders is None:
        pytest.skip("no type_d triple")
    t = rows_for(root, orders)
    assert t[("past", "3ms")] == perf
    assert t[("present", "3ms")] == pres
    assert t[("subjunctive", "3ms")] == subj
    assert t[("imperative", "2ms")] == imp


def test_book_rule_orders():
    o = {"perfect": (6, 6), "imperfect": (6, 6), "subjunctive": (6, 6)}
    assert verb_classes.book_subjunctive(("s", "a", "l"), o)["subjunctive"] == (6, 1)
    assert verb_classes.book_subjunctive(("s", "a2", "r"), o)["subjunctive"] == (6, 4)   # ይስዓር
    assert verb_classes.book_subjunctive(("q", "t", "l"), o) == o
    assert verb_classes.book_subjunctive(("s", "a", "l"), o)["imperfect"] == (6, 6)


@pytest.fixture(scope="module")
def synth():
    if not os.path.exists(SYNTH):
        pytest.skip("synthetic data not generated")
    with open(SYNTH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_no_sixth_order_2nd_radical_in_type_d_subj_imp(synth):
    bad = []
    for r in synth:
        f = r["features"]
        if f.get("verb_type") != "strong_D" or f["tense"] not in ("subjunctive", "imperative"):
            continue
        root = f["root"].split("-")
        cons = [geez_config.decompose_char(ch) for ch in r["surface_form"]]
        second = [v for c, v in cons if c == root[1]]
        if not second or second[-1] == "I":
            bad.append(r["surface_form"])
    assert not bad, bad[:10]


def test_w_a2_a_dropped(synth):
    assert not [r for r in synth if r["features"]["root"] == "w-a2-a"]
