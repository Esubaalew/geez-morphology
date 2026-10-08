"""Root-only replies ("root: <root>", "root: none"): analyzer.parse_neural_output, evaluate_gold.parse_prediction,
the analyzer's model fallback and predict_roots.py, with generation replaced by a fixed function.
Ge'ez letters are built with geez_script."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import analyzer  # noqa: E402
import dataset_loader as D  # noqa: E402
import evaluate_gold as E  # noqa: E402
import geez_script  # noqa: E402
import predict_roots as P  # noqa: E402

ROOT = "-".join(geez_script.render_unit(c, "a") for c in ("q", "t", "l"))
WORD = geez_script.render_unit("q", "a") + geez_script.render_unit("t", "a") + geez_script.render_unit("l", "a")


def test_parse_root_only_reply():
    p = analyzer.parse_neural_output(f"root: {ROOT}")
    assert p["type"] == D.ROOT_ONLY_TASK and p["root"] == ROOT and p["root_consonants"] == ["q", "t", "l"]
    assert all(p[k] is None for k in ("tense", "person", "gender", "number", "stem", "object"))
    assert analyzer.parse_neural_output(D.target_text({"root": ROOT, "task": D.ROOT_ONLY_TASK}))["root"] == ROOT


@pytest.mark.parametrize("text", ["root: none", "root: None", " root: none "])
def test_parse_no_root_reply(text):
    p = analyzer.parse_neural_output(text)
    assert p["type"] == D.ROOT_ONLY_TASK and p["root"] is None and p["root_consonants"] == []


@pytest.mark.parametrize("text", [
    "root: ",                                       # empty root
    "root: q-t-l",                                  # not Ethiopic
    "root: none | type: noun | number: s",         # "none" only in the root-only shape
    f"root: {ROOT} | type: noun",                   # nominal without number
    f"root: {ROOT} | root: {ROOT}",                 # duplicate key
])
def test_malformed_root_only_gives_none(text):
    assert analyzer.parse_neural_output(text) is None


def test_evaluate_gold_reads_root_only_replies():
    assert E.parse_prediction(f"root: {ROOT}") == {"root": ROOT}
    assert E.parse_prediction("root: none") == {"root": None}
    assert E.root_matches(E.parse_prediction(f"root: {ROOT}")["root"], ROOT)


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


def _fallback(text):
    pytest.importorskip("torch")
    a = analyzer.HybridAnalyzer.__new__(analyzer.HybridAnalyzer)
    a._ensure_neural_model = lambda: True
    a._tokenizer, a._model, a.device = _Tok(text), _Model(), "cpu"
    a.engine = None
    a.lemma_spec = None
    return a._neural_fallback(WORD)


def test_fallback_on_root_only_replies():
    r = _fallback(f"root: {ROOT}")
    assert len(r) == 1 and r[0].root_string == "q-t-l" and r[0].pos is None and r[0].tense is None
    assert _fallback("root: none") == []


def _replies(table):
    seen = []

    def generate(texts):
        seen.append(list(texts))
        return [table[t] for t in texts]
    return generate, seen


def test_predict_roots_task_root_and_analyze():
    words = [WORD, WORD + WORD, "x"]
    generate, seen = _replies({f"root: {WORD}": f"root: {ROOT}", f"root: {WORD + WORD}": "root: none",
                               "root: x": "garbage"})
    assert P.predict_roots(words, generate, batch_size=2) == [(WORD, ROOT), (WORD + WORD, "none"), ("x", "")]
    assert [len(b) for b in seen] == [2, 1]
    generate, seen = _replies({f"analyze: {WORD}": f"root: {ROOT} | type: noun | number: s"})
    assert P.predict_roots([WORD], generate, task="analyze") == [(WORD, ROOT)]
    assert P.root_of_reply("root: " + ROOT.replace("-", " ")) == ROOT       # letters joined by "-"


def test_predict_roots_main_writes_tsv(tmp_path, monkeypatch):
    words = tmp_path / "words.txt"
    words.write_text(f"{WORD}\n\n{WORD + WORD}\n", encoding="utf-8")
    generate, _ = _replies({f"root: {WORD}": f"root: {ROOT}", f"root: {WORD + WORD}": "root: none"})
    calls = []
    monkeypatch.setattr(P, "load_generator", lambda model_dir, n, device=None: calls.append(n) or generate)
    out = tmp_path / "roots.tsv"
    assert P.main([str(words), "--output", str(out), "--model_dir", str(tmp_path)]) == 0
    assert out.read_text(encoding="utf-8") == f"{WORD}\t{ROOT}\n{WORD + WORD}\tnone\n"
    assert calls == [P.MAX_NEW_TOKENS["root"]]
