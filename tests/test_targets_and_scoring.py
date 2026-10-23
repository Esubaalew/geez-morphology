"""Target shapes for nouns, adjectives and object suffixes: what the trainer writes
(dataset_loader.target_text), what the analyzer parses back (analyzer.parse_neural_output) and how
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
    noun = {"surface": "ቅቱል", "root": "ቀ-ተ-ለ", "verb_class": None,
            **E.normalize_row({"type": "noun", "number": "s"})}
    verb = {"surface": "ቀተለኒ", "root": "ቀ-ተ-ለ", "verb_class": "Type A",
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


def test_root_accuracy_is_an_exact_match():
    assert E.root_matches("ቀ-ተ-ለ", "ቀ-ተ-ለ")
    assert E.root_matches("q-t-l", "ቀ-ተ-ለ")                 # the analyzer's symbols, written in Ethiopic
    assert not E.root_matches("ቅ-ት-ል", "ቀ-ተ-ለ")             # not the 1st order: wrong
    assert not E.root_matches("ቀ-ጠ-ለ", "ቀ-ተ-ለ")
    assert not E.root_matches("", "ቀ-ተ-ለ") and not E.root_matches(None, "ቀ-ተ-ለ")


def test_root_counts_per_split():
    pairs = [{"surface": f, "root": "ቀ-ተ-ለ", "verb_class": None, **E.normalize_row({"type": "noun", "number": "s"})}
             for f in ("ቅቱል", "ቀታሊ", "ቀትል")]
    answers = iter(["ቀ-ተ-ለ", "ቀ-ጠ-ለ", "ቀ-ተ-ለ"])

    class _Seq:
        def predict(self, surface):
            return {"root": next(answers), **E.normalize_row({"type": "noun", "number": "s"}), "tier": "byt5", "raw": ""}

    rows = []
    totals, _ = E.score(pairs, _Seq(), rows)
    assert totals["ALL"]["root"] == [2, 3] and totals["ALL"]["rows"] == [0, 3]
    assert [r["correct"]["root"] for r in rows] == [True, False, True]


def test_no_root_report():
    def row(gold, said):
        return {"gold": {"root": gold}, "predicted": {"root": said}}
    scored = [row(None, None), row(None, None), row(None, "ቀ-ተ-ለ"),        # 3 gold none, 2 answered none
              row("ቀ-ተ-ለ", None), row("ቀ-ተ-ለ", "ቀ-ተ-ለ"), row("ገ-ብ-ረ", "ገ-ብ-ረ"), row("ነ-ገ-ረ", "ነ-ገ-ረ")]
    rep = E.no_root_report(scored)
    assert (rep["gold_none"], rep["predicted_none"]) == (3, 2)
    assert abs(rep["recall"] - 2 / 3) < 1e-9 and rep["recall_wilson95"] == E.wilson(2, 3)
    assert (rep["gold_real"], rep["false_none"], rep["false_none_rate"]) == (4, 1, 0.25)
    assert rep["none_replies"] == 3 and abs(rep["precision"] - 2 / 3) < 1e-9
    empty = E.no_root_report([])
    assert empty["recall"] is None and empty["precision"] is None and empty["recall_wilson95"] == (None, None)
