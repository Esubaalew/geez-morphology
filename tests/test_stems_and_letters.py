"""fixes from the grammar check of the sheet (2026-10-05).

1. no Amharic-only letters (ሸ ቸ ኘ ኸ ዠ ጀ ጨ) in generated roots;
2. causative-intensive is አስተ- (stem "aste"): አስተቅተለ, አስተትረስኩ;
3/4. no doubly weak roots, no quads ending in w/y, no ዉ / word-final ው ይ on a
     weak radical outside the final-weak class paradigms;
5. type_d forms come from the root's own Kidane Wold triple (ስእለ ፡ ይስእል ፡ ይስእል);
6. analyzer: guttural-initial present prefix (የ/ተ/አ/ነ), type_d, aste.
"""
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import geez_config  # noqa: E402
import verb_classes  # noqa: E402
from engine import InterdigitationEngine, slots_from_spec  # noqa: E402

DATASET = os.path.abspath(os.path.join(HERE, "..", "geez-dataset-v1"))
SYNTH = os.path.join(DATASET, "data", "paradigms", "synthetic_morphology_train.jsonl")
ENG = InterdigitationEngine(renderer=geez_config.render_unit)


def patterns():
    with open(os.path.join(HERE, "patterns.json"), encoding="utf-8") as f:
        return json.load(f)


def forms(root, stem, tense, pgn):
    out = set()
    for p in patterns():
        f = p["features"]
        key = (f.get("person") or "") + (f.get("gender") or "") + (f.get("number") or "")
        if f.get("stem") == stem and f.get("tense") == tense and key == pgn:
            out.add(ENG.generate(root, slots_from_spec(p["pattern_spec"]), stem=stem))
    return out


# ---------------------------------------------------------------- 2. aste
@pytest.mark.parametrize("root,pgn,expected", [
    (("q", "t", "l"), "3ms", "አስተቅተለ"),
    (("t", "r", "s"), "1cs", "አስተትረስኩ"),
    (("g", "l", "S"), "3fs", "አስተግለጸት"),
])
def test_aste_past(root, pgn, expected):
    assert forms(root, "aste", "past", pgn) == {expected}


def test_no_amharic_as_prefix_or_old_stem_name():
    for p in patterns():
        assert not p["pattern_spec"].startswith("[አስ]-"), p["pattern_spec"]
        assert p["features"].get("stem") != "causative_intensive"


# ---------------------------------------------------------------- 1, 4 helpers
def test_amharic_letters_and_weak_count():
    for ch in "ሸቸኘዠ":
        cons, _ = geez_config.decompose_char(ch)
        assert verb_classes.has_amharic_letter(("b", cons, "r"))
    assert not verb_classes.has_amharic_letter(("q", "t", "l"))
    assert set("ሸቸኘኸዠጀጨ") == set(verb_classes.AMHARIC_ONLY_LETTERS)
    assert verb_classes.weak_count(("h2", "y", "w")) == 2      # ሐይወ
    assert verb_classes.weak_count(("y", "w", "s")) == 2       # የ-ወ-ሰ
    assert verb_classes.weak_count(("q", "w", "m")) == 1


# ---------------------------------------------------------------- generated data
@pytest.fixture(scope="module")
def synth():
    if not os.path.exists(SYNTH):
        pytest.skip("synthetic data not generated")
    with open(SYNTH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


AMH = re.compile("[\u1238-\u123F\u1278-\u127F\u1298-\u129F\u12B8-\u12C5\u12E0-\u12E7\u1300-\u1307\u1328-\u132F]")


def test_no_amharic_only_letters_in_data(synth):
    bad = [r["surface_form"] for r in synth
           if AMH.search(r["surface_form"]) or verb_classes.has_amharic_letter(r["features"]["root"].split("-"))]
    assert not bad, bad[:10]


def test_no_doubly_weak_or_quad_final_weak(synth):
    bad = set()
    for r in synth:
        root = r["features"]["root"].split("-")
        if verb_classes.weak_count(root) >= 2 or (len(root) == 4 and root[-1] in ("w", "y")):
            bad.add(r["features"]["root"])
    assert not bad, sorted(bad)[:10]


def test_no_strong_template_output_on_weak_radical(synth):
    """ዉ, or a word-final 6th-order ው/ይ, only in the final-weak class paradigms
    (ተለወ: ተለዉ; ሰትየ: ይስተይ), never from the strong templates (ፈቅድዉ, ይየብው, እወልይ)."""
    bad = []
    for r in synth:
        f = r["features"]
        root = f["root"].split("-")
        if f.get("verb_type") == "weak_final" or not ({"w", "y"} & set(root)):
            continue
        s = r["surface_form"]
        if "ዉ" in s or s[-1] in "ውይ":
            bad.append((f["root"], s))
    assert not bad, bad[:10]


def test_aste_rows_start_with_aste(synth):
    rows = [r for r in synth if r["features"].get("stem") == "aste"]
    assert rows
    assert all(re.match("^(ኢ)?አስተ", r["surface_form"]) for r in rows)
    assert not [r for r in synth if r["features"].get("stem") == "causative_intensive"]


def test_type_d_matches_triples(synth):
    """Each type_d root's 3ms perfect and 3ms imperfect are its own Kidane Wold triple
    (ስእለ ፡ ይስእል is what Kidane Wold writes for ሰ-አ-ለ). the subjunctive
    follows the ፍሬ ግእዝ rule instead (test_type_d_subjunctive.py)."""
    path = os.path.join(DATASET, verb_classes.TRIPLES_FILE)
    triples = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x.get("verb_class") == "type_d":
                triples[x["root"]] = x["triple"]
    got = {}
    for r in synth:
        f = r["features"]
        if f.get("verb_type") == "strong_D" and f["person"] == "3" and f["gender"] == "m" \
                and f["number"] == "s" and "object" not in f and not f.get("negative"):
            got.setdefault(f["root"], {})[f["tense"]] = r["surface_form"]
    assert got
    for root, t in got.items():
        assert [t["past"], t["present"]] == triples[root][:2], root


# ---------------------------------------------------------------- 6. analyzer
@pytest.fixture(scope="module")
def analyzer():
    if not os.path.exists(DATASET):
        pytest.skip("dataset folder not found")
    from analyzer import HybridAnalyzer
    a = HybridAnalyzer()
    a.checkpoint_dir = None   # rules only, as in the rules-only eval; never load the model
    return a


def best(a, word):
    r = a.analyze(word)
    assert r, word
    x = r[0]
    return (x.root_string, x.stem, x.tense, x.person, x.number)


@pytest.mark.parametrize("word,expected", [
    ("የኀፍር", ("kh-f-r", "basic", "present", "3", "s")),
    ("ይኀፍር", ("kh-f-r", "basic", "subjunctive", "3", "s")),
    ("ነኀፍር", ("kh-f-r", "basic", "present", "1", "p")),
    ("ተኀፍሩ", ("kh-f-r", "basic", "present", "2", "p")),
    ("ትቀትሉ", ("q-t-l", "basic", "present", "2", "p")),
    ("ጽሕቀ", ("S-h2-q", "basic", "past", "3", "s")),
    ("ይጽሕቅ", ("S-h2-q", "basic", "present", "3", "s")),
    ("ይጽሐቅ", ("S-h2-q", "basic", "subjunctive", "3", "s")),
    ("ድኅንከ", ("d-kh-n", "basic", "past", "2", "s")),
    ("አስተቀተለ", ("q-t-l", "aste", "past", "3", "s")),
])
def test_analyzer(analyzer, word, expected):
    assert best(analyzer, word) == expected


def test_template_matcher_guttural(analyzer):
    """Template route alone: የ- copy for guttural first radicals only; the ይ-present
    template no longer takes a guttural first radical (ይኀፍር is subjunctive)."""
    pats = analyzer.compiled_patterns
    got = {(r.root_string, r.stem, r.tense) for r in analyzer._pattern_analyze("የሐምስ", pats, False)}
    assert ("h2-m-s", "basic", "present") in got
    got = {(r.root_string, r.stem, r.tense) for r in analyzer._pattern_analyze("ይሐምስ", pats, False)}
    assert ("h2-m-s", "basic", "present") not in got
    assert ("h2-m-s", "basic", "subjunctive") in got
    assert not [r for r in analyzer._pattern_analyze("የቀትል", pats, False) if r.root_string == "q-t-l"]


def test_no_amharic_as_stem(analyzer):
    stems = {r.stem for r in analyzer.analyze("አስትረስኩ")}
    assert "causative_intensive" not in stems


# ---------------------------------------------------------------- 7. evaluation
def test_eval_aste_mapping_and_exclusions():
    import evaluate_gold as E
    assert E.ASTE_FORM.match("ያስተሰነግል") and E.ASTE_FORM.match("አስተሰነገለ")
    assert not E.ASTE_FORM.match("አንጸብረቀ") and not E.ASTE_FORM.match("አቅየዐየዐ")
    path = os.path.join(HERE, "notes", "gold_proposals_2026-10-05.jsonl")
    if not os.path.exists(path):
        pytest.skip("proposals not written")
    ex = E.load_exclusions(path)
    assert ("ብሕትው", "b-ḥ-t-w") in ex          # not a verb
    assert ("ይሰናአሉ", "ሰ-ነ-አ-ለ (s-n-'-l)") not in ex   # correct as is (C)
