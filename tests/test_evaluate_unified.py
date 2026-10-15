"""evaluate_gold: unified val/test splits, words without root, possessor, negated and the Wilson interval."""
import collections
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset_loader  # noqa: E402
import evaluate_gold as ev  # noqa: E402


def test_wilson_interval():
    lo, hi = ev.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert ev.wilson(0, 0) == (None, None)
    assert ev.wilson(10, 10)[1] == 1.0


def test_rootless_gold_matches_only_a_reply_without_root():
    assert ev.root_matches(None, None) and not ev.root_matches("ቀ-ተ-ለ", None)
    assert not ev.root_matches(None, "ቀ-ተ-ለ")


def test_negated_and_possessor_are_normalized():
    r = ev.normalize_row({"type": "verb", "negated": True, "possessor": "3ms"})
    assert r["negated"] == "yes" and r["possessor"] == "3ms"
    assert ev.normalize_row({"type": "noun"})["negated"] is None
    r1 = ev.normalize_row({"type": "verb", "object": "1s", "possessor": "1p"})
    assert r1["object"] == "1cs" and r1["possessor"] == "1cp"


class Oracle:
    """Answers with the gold target text, read back through parse_prediction (the trainer's target format)."""
    def __init__(self, pairs):
        self.by_form = {}
        for p in pairs:
            self.by_form.setdefault(p["surface"], p)

    def predict(self, surface):
        p = self.by_form[surface]
        t = {"type": p["type"], "root": p["root"], "tense": p["tense"], "person": p["person"], "gender": p["gender"],
             "number": p["number"], "stem": p["stem"], "object": p["object"], "possessor": p["possessor"],
             "negated": p["negated"] == "yes", "form": surface}
        text = dataset_loader.target_text(t)
        pred = ev.parse_prediction(text)
        return {"root": pred.get("root"), **ev.normalize_row(pred), "tier": "byt5", "raw": text}


@pytest.fixture
def dataset_dir(request):
    d = request.config.getoption("--dataset_dir", default=None) or dataset_loader.default_dataset_dir()
    if not os.path.exists(os.path.join(d, dataset_loader.UNIFIED_FILE)):
        pytest.skip("no unified file in the dataset checkout")
    return d


@pytest.mark.parametrize("split", ["unified_val", "unified_test"])
def test_oracle_scores_100_percent_on_unified_splits(dataset_dir, split):
    stats = collections.Counter()
    pairs = ev.load_gold_pairs(dataset_dir, split, stats)
    assert pairs and stats["loaded"] == len(pairs)
    totals, _ = ev.score(pairs, Oracle(pairs))
    root_c, root_n = totals["ALL"]["root"]
    ex_c, ex_n = totals["ALL"]["exact"]
    assert root_c == root_n == len(pairs)
    assert ex_n - ex_c == ev.duplicate_surplus(pairs)   # the only misses are spellings with two analyses
    assert any(g.startswith("book:") for g in totals)


def test_root_variant_groups(dataset_dir):
    train = ev.folded_train_roots(dataset_dir)
    pairs = ev.load_gold_pairs(dataset_dir, "test", collections.Counter())
    ev.tag_root_variants(pairs, train)
    kinds = collections.Counter(p["rootvar"] for p in pairs)
    assert kinds["disjoint"] > 0 and set(kinds) <= {"disjoint", "variant_in_train"}
    totals, _ = ev.score(pairs, Oracle(pairs))
    assert totals["rootvar:disjoint"]["rows"][1] == kinds["disjoint"]


def test_dev_unified_flag_and_source_groups(dataset_dir):
    gold = ev.load_gold_pairs(dataset_dir, "val", collections.Counter())
    book = ev.load_gold_pairs(dataset_dir, "unified_val", collections.Counter())
    for p in book:
        p["split"] = "val"
    pairs = gold + book
    totals, _ = ev.score(pairs, Oracle(pairs))
    assert totals["source:gold"]["rows"][1] == len(gold) and totals["source:book"]["rows"][1] == len(book)
    assert totals["ALL"]["rows"][1] == len(pairs)


def test_hybrid_neural_fallback_preserves_unified_classes_possessor_and_negated():
    pytest.importorskip("torch")   # _neural_fallback imports torch and returns [] without it
    import analyzer
    a = analyzer.HybridAnalyzer.__new__(analyzer.HybridAnalyzer)
    a._ensure_neural_model = lambda: True

    class _Inputs(dict):
        def to(self, dev):
            return self

    class _Tok:
        def __init__(self, text):
            self.text = text
        def __call__(self, *args, **kw):
            return _Inputs()
        def decode(self, *args, **kw):
            return self.text

    class _Model:
        def generate(self, **kw):
            return [[0]]

    a._model, a.device, a.engine, a.lemma_spec = _Model(), "cpu", None, None
    a._tokenizer = _Tok("root: ቀ-ተ-ለ | type: infinitive | stem: basic | possessor: 3ms | negated: yes")
    res = a._neural_fallback("ኢቀቲሎቱ")
    assert len(res) == 1
    assert res[0].pos == "infinitive" and res[0].possessor == "3ms" and res[0].negative is True

    a._tokenizer = _Tok("root: none | type: pronoun | person: 3 | gender: m | number: s")
    res_pro = a._neural_fallback("ውእቱ")
    assert len(res_pro) == 1
    assert res_pro[0].pos == "pronoun" and res_pro[0].root_string == "" and res_pro[0].person == "3"
