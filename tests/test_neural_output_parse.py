"""Strict parse of the ByT5 fallback output (analyzer.parse_neural_output).
A prediction must have exactly the six training fields; anything else gives no answer."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyzer  # noqa: E402

GOOD = "root: ቀ-ተ-ለ | tense: present | person: 3 | gender: m | number: s | stem: basic"


def test_well_formed():
    p = analyzer.parse_neural_output(GOOD)
    assert p["root_consonants"] == ["q", "t", "l"]
    assert (p["tense"], p["person"], p["gender"], p["number"], p["stem"]) == ("present", "3", "m", "s", "basic")


def test_empty_values_mean_none():
    p = analyzer.parse_neural_output("root: ቀ-ተ-ለ | tense: past | person: 1 | gender:  | number: s | stem: basic")
    assert p is not None and p["gender"] is None
    p = analyzer.parse_neural_output("root: ቀ-ተ-ለ | tense: None | person: 3 | gender: m | number: s | stem: basic")
    assert p is not None and p["tense"] is None


@pytest.mark.parametrize("text", [
    "",
    "root: ቀ-ተ-ለ | tense: present | person: 3",                                  # truncated
    "root: ቀ-ተ-ለ | tense: present | person: 3 | gender: m | number: s",           # missing stem
    GOOD + " | mood: x",                                                            # unknown key
    GOOD.replace("tense:", "tens:"),                                                # misspelled key
    GOOD + " | stem: passive",                                                      # duplicate key
    GOOD.replace(" | person", " person"),                                           # missing separator
    "root:  | tense: present | person: 3 | gender: m | number: s | stem: basic",    # empty root
    "root: q-t-l | tense: present | person: 3 | gender: m | number: s | stem: basic",  # not Ethiopic
    "garbage output",
])
def test_malformed_gives_none(text):
    assert analyzer.parse_neural_output(text) is None


class _Inputs(dict):
    def to(self, device):
        return self


class _Tok:
    def __init__(self, text):
        self.text = text

    def __call__(self, *a, **k):
        return _Inputs()

    def decode(self, *a, **k):
        return self.text


class _Model:
    def generate(self, **k):
        return [[0]]


def _need_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")


def _fallback(text):
    a = analyzer.HybridAnalyzer.__new__(analyzer.HybridAnalyzer)
    a._ensure_neural_model = lambda: True
    a._tokenizer, a._model, a.device = _Tok(text), _Model(), "cpu"
    a.engine = None   # lemma synthesis falls back to the word itself
    a.lemma_spec = None
    return a._neural_fallback("ቀተለ")


def test_fallback_rejects_partial_output():
    _need_torch()
    assert _fallback("root: ቀ-ተ-ለ | tense: present") == []
    assert _fallback(GOOD + " | extra: 1") == []


def test_fallback_accepts_full_output():
    _need_torch()
    r = _fallback(GOOD)
    assert len(r) == 1
    assert (r[0].root_string, r[0].tense, r[0].person, r[0].stem) == ("q-t-l", "present", "3", "basic")
