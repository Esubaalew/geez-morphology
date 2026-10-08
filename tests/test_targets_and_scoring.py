"""Target shapes for nouns, adjectives and object suffixes: what the trainer writes
(gold_filters.target_text), what the analyzer parses back (analyzer.parse_neural_output) and how
evaluate_gold scores them."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyzer  # noqa: E402
import evaluate_gold as E  # noqa: E402


def test_parse_nominal_and_object_targets():
    p = analyzer.parse_neural_output("root: ቀ-ተ-ለ | type: adjective | gender: f | number: s")
    assert p["type"] == "adjective" and p["gender"] == "f" and p["tense"] is None and p["stem"] is None
    p = analyzer.parse_neural_output("root: ቀ-ተ-ለ | type: noun | number: p")
    assert p["type"] == "noun" and p["gender"] is None
    p = analyzer.parse_neural_output("root: ቀ-ተ-ለ | tense: past | person: 3 | gender: m | number: s | stem: basic | object: 1cs")
    assert p["type"] == "verb" and p["object"] == "1cs"
    for bad in ("root: ቀ-ተ-ለ | type: verb | number: s",                   # a verb needs the six fields
                "root: ቀ-ተ-ለ | type: noun | number: s | tense: past",      # nominal with a verb field
                "root: ቀ-ተ-ለ | type: noun"):                              # no number
        assert analyzer.parse_neural_output(bad) is None


class _Fixed:
    def __init__(self, pred):
        self.pred = pred

    def predict(self, surface):
        return dict(self.pred)


def test_scoring_of_nominal_and_object_pairs():
    noun = {"surface": "ቅቱል", "root_keys": E.gold_root_keys("ቀ-ተ-ለ"), "verb_class": None,
            **E.normalize_row({"type": "noun", "number": "s"})}
    verb = {"surface": "ቀተለኒ", "root_keys": E.gold_root_keys("ቀ-ተ-ለ"), "verb_class": "Type A",
            **E.normalize_row({"type": "verb", "stem": "basic", "tense": "past", "person": "3", "gender": "m",
                               "number": "s", "object": "1cs"})}
    as_verb = {"root": "ቀ-ተ-ለ", **E.normalize_row({"stem": "basic", "tense": "past", "person": "3", "gender": "m",
                                                     "number": "s"}), "tier": "pattern", "raw": ""}
    totals, _ = E.score([noun, verb], _Fixed(as_verb))
    assert totals["type:noun"]["root"] == [1, 1] and totals["type:noun"]["type"] == [0, 1]
    assert totals["type:noun"]["exact"] == [0, 1] and totals["type:noun"]["tense"] == [0, 0]
    assert totals["type:verb"]["object"] == [0, 1] and totals["type:verb"]["exact"] == [0, 1]
    as_noun = {"root": "ቀ-ተ-ለ", **E.normalize_row({"type": "noun", "number": "s"}), "tier": "byt5", "raw": ""}
    totals, _ = E.score([noun], _Fixed(as_noun))
    assert totals["type:noun"]["exact"] == [1, 1]
