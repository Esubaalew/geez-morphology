"""Reference forms for the generator and the rules-only analyzer.

The forms follow standard classical grammar (Lambdin, Dillmann). Roots are written with the
Latin consonant symbols of engine.py. The generation cases are looked up in the synthetic data of
the dataset checkout (training file and held-out file), 3ms basic stem, no object, not negative.
Known gaps are marked xfail with the reason.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from conftest import DATASET_DIR  # noqa: E402

SYNTHETIC_FILES = [
    os.path.join(DATASET_DIR, "data", "paradigms", "synthetic_morphology_train.jsonl"),
    os.path.join(DATASET_DIR, "data", "synthetic", "synthetic_forms_for_val_test_roots.jsonl"),
]

# (root, tense, expected 3ms surface form)
GENERATION = [
    # strong, type A
    (('q', 't', 'l'), 'past', 'ቀተለ'),
    (('q', 't', 'l'), 'present', 'ይቀትል'),
    (('q', 't', 'l'), 'subjunctive', 'ይቅትል'),
    # type B (doubled middle radical, not written)
    (('q', 'd', 's'), 'past', 'ቀደሰ'),
    (('q', 'd', 's'), 'present', 'ይቄድስ'),
    # type C (long a after first radical)
    (('b', 'r', 'k'), 'past', 'ባረከ'),
    (('b', 'r', 'k'), 'present', 'ይባርክ'),
    # hollow w: contracts in perfect and subjunctive, NOT in imperfect
    (('q', 'w', 'm'), 'past', 'ቆመ'),
    (('q', 'w', 'm'), 'present', 'ይቀውም'),
    (('q', 'w', 'm'), 'subjunctive', 'ይቁም'),
    pytest.param(('m', 'w', 't'), 'past', 'ሞተ', marks=pytest.mark.xfail(reason='m-w-t is not on the allow-list, so the generator does not conjugate it')),
    pytest.param(('m', 'w', 't'), 'subjunctive', 'ይሙት', marks=pytest.mark.xfail(reason='m-w-t is not on the allow-list, so the generator does not conjugate it')),
    # hollow y
    (('s2', 'y', 'm'), 'past', 'ሤመ'),
    (('s2', 'y', 'm'), 'subjunctive', 'ይሢም'),
    # middle w that does NOT contract (type B)
    (('f', 'w', 's'), 'past', 'ፈወሰ'),
    # final weak: must not contract in the perfect
    (('t', 'l', 'w'), 'past', 'ተለወ'),
    pytest.param(('h1', 'l', 'w'), 'past', 'ሀለወ', marks=pytest.mark.xfail(reason='h1-l-w is not on the allow-list, so the generator does not conjugate it')),
    pytest.param(('h1', 'l', 'w'), 'present', 'ይሄሉ', marks=pytest.mark.xfail(reason='h1-l-w is not on the allow-list, so the generator does not conjugate it')),
    pytest.param(('s', 't', 'y'), 'past', 'ሰተየ', marks=pytest.mark.xfail(reason='the final-y paradigm writes the 3ms perfect with a 6th-order 2nd radical')),
]

# Wrong forms an older engine produced; none of them may appear as a synthetic surface form.
MUST_NOT_GENERATE = ['ፎሰ', 'ተሎ', 'ሀሎ', 'ሰቴ']

# (surface word, expected root, expected features of the best analysis)
ANALYSIS = [
    ('ቀተለ', 'q-t-l', {'tense': 'past', 'person': '3', 'number': 's'}),
    ('ቀተልኩ', 'q-t-l', {'tense': 'past', 'person': '1'}),
    ('ቀተልነ', 'q-t-l', {'tense': 'past', 'person': '1', 'number': 'p'}),
    ('ቀተሎ', 'q-t-l', {'object': '3ms'}),   # object fused into last letter
    ('ይቀትሎ', 'q-t-l', {'tense': 'present', 'object': '3ms'}),
    ('ይቀትለኒ', 'q-t-l', {'object': '1s'}),
    ('ቀተልክዎ', 'q-t-l', {'person': '1', 'object': '3ms'}),   # standard spelling
    ('ቀተልኩዎ', 'q-t-l', {'person': '1', 'object': '3ms'}),   # common variant
    ('ቀተልክዮ', 'q-t-l', {'person': '2', 'gender': 'f', 'object': '3ms'}),
    ('እቀትል', 'q-t-l', {'person': '1', 'number': 's'}),
    ('ንቀትል', 'q-t-l', {'person': '1', 'number': 'p'}),
    ('ተቀትለ', 'q-t-l', {'stem': 'passive'}),
    ('አቅተለ', 'q-t-l', {'stem': 'causative'}),
    ('ኢይቀትል', 'q-t-l', {'negative': True}),
    ('ወይቀትል', 'q-t-l', {}),   # ወ- is the conjunction "and"
    ('ቆመ', 'q-w-m', {}),   # two surface consonants
    ('ወሰነ', 'w-s-n', {}),
    ('ባረከ', 'b-r-k', {}),
    ('ተከለ', 't-k-l', {}),
    ('ወለደ', 'w-l-d', {}),
]


@pytest.fixture(scope="module")
def synthetic_rows():
    if not all(os.path.exists(p) for p in SYNTHETIC_FILES):
        pytest.skip("no synthetic data in the dataset checkout")
    rows = []
    for path in SYNTHETIC_FILES:
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    return rows


@pytest.fixture(scope="module")
def forms_3ms(synthetic_rows):
    """{(root, tense): set of 3ms basic forms without object or negation}"""
    out = {}
    for r in synthetic_rows:
        f = r["features"]
        if (f.get("person"), f.get("gender"), f.get("number")) != ("3", "m", "s"):
            continue
        if f.get("stem") != "basic" or f.get("object") or f.get("negative"):
            continue
        out.setdefault((tuple(f["root"].split("-")), f["tense"]), set()).add(r["surface_form"])
    return out


@pytest.mark.parametrize("root,tense,expected", GENERATION)
def test_generation(forms_3ms, root, tense, expected):
    assert expected in forms_3ms.get((root, tense), set())


def test_must_not_generate(synthetic_rows):
    bad = sorted({r["surface_form"] for r in synthetic_rows} & set(MUST_NOT_GENERATE))
    assert not bad


@pytest.fixture(scope="module")
def analyzer():
    from analyzer import DICTIONARY_FILE, HybridAnalyzer
    if not os.path.exists(os.path.join(DATASET_DIR, DICTIONARY_FILE)):
        pytest.skip("no geez-dataset checkout")
    return HybridAnalyzer(dataset_dir=DATASET_DIR, use_neural=False)


@pytest.mark.parametrize("word,root,feats", ANALYSIS)
def test_analysis(analyzer, word, root, feats):
    results = analyzer.analyze(word)
    assert results, word
    best = results[0]
    assert best.root_string == root
    for key, value in feats.items():
        assert getattr(best, key) == value, (key, getattr(best, key))
