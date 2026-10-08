"""Grammar decisions of 2026-10-05.

1. After a guttural, 1st and 4th order are spelling variants (ዐ/ዓ ሐ/ሓ ኀ/ኃ ሀ/ሃ አ/ኣ).
   The data keeps the 1st order; the analyzer input and the gold-eval comparison treat
   each pair as equal (4th -> 1st, for these five letters only, at comparison time).
2. ፍሬ ግእዝ middle-guttural rule for three-radical roots that default to Type A:
   present ይምሕል (no doubling), subjunctive/jussive ይምሐል, imperative ምሐል.
   Roots with their own Kidane Wold triple, class-tagged roots, initial-weak roots and
   roots with a guttural first radical keep their forms. Derived stems are unchanged.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import geez_config  # noqa: E402
import gold_filters  # noqa: E402
import verb_classes  # noqa: E402

DATASET = gold_filters.default_dataset_dir()
SYNTH = os.path.join(DATASET, "data", "paradigms", "synthetic_morphology_train.jsonl")
# the generator's output is the training file plus the lines held out for the gold val/test roots (14:35)
HELD = os.path.join(DATASET, "data", "synthetic", "synthetic_forms_for_val_test_roots.jsonl")
GUTT = {"a", "a2", "h1", "h2", "kh"}


# ---- 1. spelling variants -------------------------------------------------

@pytest.mark.parametrize("fourth,first", [("ዓ", "ዐ"), ("ሓ", "ሐ"), ("ኃ", "ኀ"), ("ሃ", "ሀ"), ("ኣ", "አ")])
def test_normalize_guttural_4th_pairs(fourth, first):
    assert geez_config.normalize_guttural_4th(fourth) == first
    assert geez_config.gutturals_equal(fourth, first)


def test_normalize_touches_only_the_five_gutturals():
    for ch in "ባላማሳቃናጣፋዳጋዛካታ":                   # 4th order of non-gutturals
        assert geez_config.normalize_guttural_4th(ch) == ch
    for ch in "ሐሑሒሔሕሖዐዑዒዔዕዖ":                       # other orders untouched
        assert geez_config.normalize_guttural_4th(ch) == ch
    assert geez_config.normalize_guttural_4th("ሰምዓ") == "ሰምዐ"
    assert geez_config.normalize_guttural_4th("ኣምላክ") == "አምላክ"
    assert not geez_config.gutturals_equal("ባ", "በ")


@pytest.fixture(scope="module")
def analyzer():
    if not os.path.exists(DATASET):
        pytest.skip("no geez-dataset checkout (the analyzer reads its lexicon)")
    from analyzer import HybridAnalyzer
    return HybridAnalyzer()


def top(analyzer, word):
    r = analyzer.analyze(word)
    assert r, word
    x = r[0]
    return (x.root_string, x.tense, x.person, x.gender, x.number)


@pytest.mark.parametrize("variant,standard", [
    ("ሰምዓ", "ሰምዐ"),
    ("ይምሓል", "ይምሐል"),
    ("ምሓል", "ምሐል"),
])
def test_analyzer_reads_4th_order_variant_like_1st_order(analyzer, variant, standard):
    # The variant gets the standard spelling's best reading among its analyses. It can
    # also keep its own as-written readings first: a final 4th-order guttural is also a
    # real ending (ሰምዓ is the 3fp perfect as well as a spelling of ሰምዐ).
    readings = [(x.root_string, x.tense, x.person, x.gender, x.number) for x in analyzer.analyze(variant)]
    assert top(analyzer, standard) in readings


# ---- 2. middle-guttural rule ----------------------------------------------

@pytest.mark.parametrize("word,expect", [
    ("ይምሕል", ("m-h2-l", "present", "3", "m", "s")),
    ("ይምሐል", ("m-h2-l", "subjunctive", "3", "m", "s")),
    ("ምሐል", ("m-h2-l", "imperative", "2", "m", "s")),
])
def test_analyzer_middle_guttural(analyzer, word, expect):
    assert top(analyzer, word) == expect


@pytest.fixture(scope="module")
def synth():
    if not os.path.exists(SYNTH):
        pytest.skip("synthetic data not generated")
    out = []
    for path in (SYNTH, HELD):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out += [json.loads(line) for line in f]
    return out


def form(synth, root, tense, per, stem="basic"):
    for r in synth:
        f = r["features"]
        if (f["root"] == root and f["tense"] == tense and f.get("stem") == stem and "object" not in f
                and not f.get("negative")
                and f["person"] + (f.get("gender") or "") + f["number"] == per):
            return r["surface_form"]
    return None


@pytest.mark.parametrize("root,tense,per,expect", [
    ("m-h2-l", "past", "3ms", "መሐለ"),          # perfect unchanged
    ("m-h2-l", "present", "3ms", "ይምሕል"),      # not ይመሕል
    ("m-h2-l", "subjunctive", "3ms", "ይምሐል"),
    ("m-h2-l", "imperative", "2ms", "ምሐል"),
    ("m-h2-l", "imperative", "2mp", "ምሐሉ"),
    ("s2-a2-r", "present", "3ms", "ይሥዕር"),
    ("s2-a2-r", "subjunctive", "3fs", "ትሥዐር"),
    ("r-kh-b", "imperative", "2ms", "ርኀብ"),
    ("l-h1-k", "subjunctive", "3ms", "ይልሀክ"),
])
def test_generated_middle_guttural_forms(synth, root, tense, per, expect):
    assert form(synth, root, tense, per) == expect


@pytest.mark.parametrize("root,tense,per,stem,expect", [
    ("S-h2-f", "subjunctive", "3ms", "basic", "ይጽሐፍ"),     # triple's ይጽሕፍ is an OCR collapse
    ("S-h2-f", "present", "3ms", "basic", "ይጽሕፍ"),         # present from the triple
    ("s-a-l", "present", "3ms", "basic", "ይስእል"),          # type_d, Kidane Wold
    ("s-a-l", "subjunctive", "3ms", "basic", "ይስአል"),      # Type D book rule
    ("a-kh-z", "present", "3ms", "basic", "የአኅዝ"),         # guttural first radical: የ- rule kept
    ("m-h2-l", "present", "3ms", "passive", "ይትመሐል"),     # derived stems unchanged
    ("m-h2-l", "present", "3ms", "causative", "ያመሕል"),
])
def test_excluded_roots_and_derived_stems_unchanged(synth, root, tense, per, stem, expect):
    assert form(synth, root, tense, per, stem) == expect


def test_no_doubled_guttural_present_in_changed_roots(synth):
    kw = verb_classes.triple_roots(DATASET)
    bad, seen = [], set()
    for r in synth:
        f = r["features"]
        root = tuple(f["root"].split("-"))
        if (len(root) != 3 or f.get("verb_type") != "strong_A" or root[1] not in GUTT or root[0] in GUTT
                or root in kw or f.get("stem") != "basic"):
            continue
        dec = [geez_config.decompose_char(ch) for ch in r["surface_form"]]
        i = next((k for k, (c, v) in enumerate(dec) if c == root[0]), None)
        j = next((k for k in range(i + 1, len(dec)) if dec[k][0] == root[1]), None) if i is not None else None
        assert i is not None and j is not None, r
        seen.add(root)
        if f["tense"] == "present" and dec[i][1] == "a":
            bad.append(r["surface_form"])
        if f["tense"] in ("subjunctive", "imperative") and dec[j][1] != "a":
            bad.append(r["surface_form"])
    assert len(seen) == 155     # + f-a-m (ፈ-አ-መ): Type A since its Type C tag no longer comes from the extracted gold train
    assert bad == []
