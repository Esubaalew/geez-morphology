"""Expected Ge'ez forms for checking the contraction fix and the HybridAnalyzer.

Drop into the geez-morphology repo as tests/test_expected_forms.py.
Forms follow standard classical grammar (Lambdin / Dillmann conventions).
Roots are written with Latin letters to match engine.py's consonant symbols.
"""
import pytest

# (root, tense/stem, expected 3ms surface form)
GENERATION = [
    # strong, type A
    (("q", "t", "l"), "perfect", "ቀተለ"),
    (("q", "t", "l"), "imperfect", "ይቀትል"),
    (("q", "t", "l"), "jussive", "ይቅትል"),
    # type B (doubled middle radical, not written)
    (("q", "d", "s"), "perfect", "ቀደሰ"),
    (("q", "d", "s"), "imperfect", "ይቄድስ"),
    # type C (long a after first radical)
    (("b", "r", "k"), "perfect", "ባረከ"),
    (("b", "r", "k"), "imperfect", "ይባርክ"),
    # hollow w: contracts in perfect and jussive, NOT in imperfect
    (("q", "w", "m"), "perfect", "ቆመ"),
    (("q", "w", "m"), "imperfect", "ይቀውም"),
    (("q", "w", "m"), "jussive", "ይቁም"),
    (("m", "w", "t"), "perfect", "ሞተ"),
    (("m", "w", "t"), "jussive", "ይሙት"),
    # hollow y
    (("s2", "y", "m"), "perfect", "ሤመ"),
    (("s2", "y", "m"), "jussive", "ይሢም"),
    # middle w that does NOT contract (type B)
    (("f", "w", "s"), "perfect", "ፈወሰ"),
    # final weak: must not contract in the perfect
    (("t", "l", "w"), "perfect", "ተለወ"),
    (("h1", "l", "w"), "perfect", "ሀለወ"),
    (("h1", "l", "w"), "imperfect", "ይሄሉ"),
    (("s", "t", "y"), "perfect", "ሰትየ"),
]

# Wrong outputs the current engine produces; none of these may come back.
MUST_NOT_GENERATE = ["ፎሰ", "ተሎ", "ሀሎ", "ሰቴ"]

# (surface word, expected root, expected features subset)
ANALYSIS = [
    ("ቀተለ", "q-t-l", {"tense": "past", "person": "3", "number": "s"}),
    ("ቀተልኩ", "q-t-l", {"tense": "past", "person": "1"}),
    ("ቀተልነ", "q-t-l", {"tense": "past", "person": "1", "number": "p"}),
    ("ቀተሎ", "q-t-l", {"object": "3ms"}),         # object fused into last letter
    ("ይቀትሎ", "q-t-l", {"tense": "present", "object": "3ms"}),
    ("ይቀትለኒ", "q-t-l", {"object": "1s"}),
    ("ቀተልክዎ", "q-t-l", {"person": "1", "object": "3ms"}),   # standard spelling
    ("ቀተልኩዎ", "q-t-l", {"person": "1", "object": "3ms"}),   # common variant
    ("ቀተልክዮ", "q-t-l", {"person": "2", "gender": "f", "object": "3ms"}),
    ("እቀትል", "q-t-l", {"person": "1", "number": "s"}),
    ("ንቀትል", "q-t-l", {"person": "1", "number": "p"}),
    ("ተቀትለ", "q-t-l", {"stem": "passive"}),
    ("አቅተለ", "q-t-l", {"stem": "causative"}),
    ("ኢይቀትል", "q-t-l", {"negative": True}),
    ("ወይቀትል", "q-t-l", {}),                       # ወ- is the conjunction "and"
    ("ቆመ", "q-w-m", {}),                           # two surface consonants
    # radicals that look like affixes; must NOT be stripped
    ("ወሰነ", "w-s-n", {}),
    ("ባረከ", "b-r-k", {}),
    ("ተከለ", "t-k-l", {}),
    ("ወለደ", "w-l-d", {}),
]


@pytest.mark.parametrize("root,tense,expected", GENERATION)
def test_generation(root, tense, expected):
    pytest.skip("wire to engine once templates are keyed by tense and class")


@pytest.mark.parametrize("word,root,feats", ANALYSIS)
def test_analysis(word, root, feats):
    pytest.skip("wire to HybridAnalyzer.analyze(word, use_hybrid=True)")
