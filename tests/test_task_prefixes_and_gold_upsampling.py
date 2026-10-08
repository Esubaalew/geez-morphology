"""Task prefixes, gold upsampling, words without root and the root-accuracy helpers of train_byt5_morphology.py.

The first tests build a small dataset folder whose letters come from geez_script and whose words without root
come from data/closed_class_words_without_root.tsv at run time. The later tests read the dataset checkout.
"""
import collections
import json
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import analyzer  # noqa: E402
import dataset_loader as D  # noqa: E402
import evaluate_gold  # noqa: E402
import generate_synthetic_data as gen  # noqa: E402
import geez_script  # noqa: E402
import split_heldout_synthetic as H  # noqa: E402
import train_byt5_morphology as T  # noqa: E402
import verb_classes  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402

PROCLITIC_LETTERS = ("w", "l", "b", "z")   # first radicals that are also proclitics


def _word(symbols, vowels):
    return "".join(geez_script.render_unit(c, v) for c, v in zip(symbols, vowels))


def _root(symbols):
    return "-".join(geez_script.render_unit(c, "a") for c in symbols)


def _dataset(tmp, n_syn=40):
    """Synthetic verbs of one root, gold train with 3 verb rows, 2 noun rows and 1 adjective row, a gold
    val form and a gold train form (row with a root) taken from the words-without-root file, and a gold train
    row without root whose form is another word of that file."""
    roots = sorted(r for r in gen.read_allowed_roots_file() if len(r) == 3)
    A, B, C = roots[:3]
    words = sorted(D.read_words_without_root())
    vowels = list(geez_script.VOWEL_ORDERS)
    para = os.path.join(tmp, "data", "paradigms")
    os.makedirs(para)
    with open(os.path.join(para, "synthetic_morphology_train.jsonl"), "w", encoding="utf-8") as f:
        for i in range(n_syn):
            f.write(json.dumps({"surface_form": _word(A, "eoe") + _word(B[:1], vowels[i % len(vowels)]) * (i // len(vowels) + 1),
                                "features": {"root": "-".join(A), "tense": "past", "person": "3", "gender": "m",
                                             "number": "s", "stem": "basic"}}, ensure_ascii=False) + "\n")
    verb = {"tense": "past", "person": "3", "gender": "m", "number": "s"}
    train = [
        {"record_id": "a" * 12, "root": _root(A), "stem": "basic", "table_data": [
            dict(verb, geez_form=_word(A, "aaa")), dict(verb, geez_form=_word(A, "aea")),
            dict(verb, geez_form=_word(A, "eaa")),
            {"geez_form": _word(A, "aoa"), "type": "noun", "number": "s"},
            {"geez_form": _word(A, "ioa"), "type": "noun", "number": "p"},
            {"geez_form": _word(A, "uia"), "type": "adjective", "gender": "m", "number": "s"},
            {"geez_form": words[1], "type": "noun", "number": "s"}]},
        {"record_id": "b" * 12, "root": None, "table_data": [{"geez_form": words[2], "type": "noun"}]},
    ]
    val = [{"record_id": "c" * 12, "root": _root(B), "table_data": [{"geez_form": words[0], "type": "noun"}]}]
    test = [{"record_id": "d" * 12, "root": _root(C), "table_data": [{"geez_form": _word(C, "ooo"), "type": "noun"}]}]
    for split, recs in (("train", train), ("val", val), ("test", test)):
        with open(os.path.join(para, D.gold_file_name(split)), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return words


def test_input_prefix_follows_the_task():
    assert D.input_text("x") == "analyze: x" == D.input_text("x", D.ANALYZE_TASK)
    assert D.input_text("x", D.ROOT_ONLY_TASK) == "root: x"
    assert D.input_text("x", D.ROOT_ONLY_TASK, single_prefix=True) == "analyze: x"
    assert D.task_of({}) == D.ANALYZE_TASK and D.task_of({"task": D.ROOT_ONLY_TASK}) == D.ROOT_ONLY_TASK
    pair = {"form": "x", "root": "r", "task": D.ROOT_ONLY_TASK}
    assert T.root_only_sample(pair, "dictionary")["input_text"] == "root: x"
    assert T.root_only_sample(pair, "dictionary", single_prefix=True)["input_text"] == "analyze: x"
    assert T.root_only_sample(pair, "dictionary")["target_text"] == "root: r"


def test_no_root_target_is_read_as_no_value():
    assert D.target_text({"form": "x", "root": D.NO_ROOT, "task": D.ROOT_ONLY_TASK}) == "root: none"
    assert D.NO_ROOT in analyzer.NO_VALUE and D.NO_ROOT in verb_classes.ROOT_PLACEHOLDERS
    assert evaluate_gold.clean_label(D.NO_ROOT) is None


def test_gold_upsampling_in_train_only():
    with tempfile.TemporaryDirectory() as tmp:
        _dataset(tmp)
        samples = T.load_data(tmp)
        assert collections.Counter(s["source"] for s in samples) == {"synthetic": 40, "gold_verb": 3, "gold_nominal": 4}
        plain = T.split_samples(samples)
        assert plain == T.split_samples(samples, 1, 1)
        train, evals, n_syn, n_gold = T.split_samples(samples, 4, 5)
        assert (n_syn, n_gold) == (40, 7) and evals == plain[1]
        assert not any(s["is_gold"] for s in evals)
        assert T.counts_by_source(train) == {"synthetic": 36, "gold_verb": 12, "gold_nominal": 20}
        assert train == T.split_samples(samples, 4, 5)[0]


def test_words_without_root_and_their_leak_gate():
    with tempfile.TemporaryDirectory() as tmp:
        words = _dataset(tmp)
        assert not set(words) & gen.read_blocked_forms_file()
        stats = collections.Counter()
        pairs = D.load_words_without_root(tmp, stats)
        assert stats == {"loaded": len(words) - 2, "not loaded: form is a gold val/test form": 1,
                         "not loaded: form has a root in the gold train file": 1}
        assert {p["form"] for p in pairs} == set(words) - {words[0], words[1]}       # words[2]: gold row without root
        assert all(D.target_text(p) == "root: none" and p["task"] == D.ROOT_ONLY_TASK for p in pairs)
        samples = T.load_words_without_root_samples(tmp)
        assert all(s["input_text"].startswith("root: ") and s["source"] == "word_without_root" for s in samples)
        bad = os.path.join(tmp, "bad.tsv")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("word\tlabel\tkind\tnote\n" + words[3] + "\tx\tname\t\n")
        with pytest.raises(ValueError):
            D.load_words_without_root(tmp, path=bad)


def _main_export(tmp, *flags):
    out = os.path.join(tmp, "out")
    assert T.main(["--dataset_dir", tmp, "--load_only", "--export_dir", out, *flags]) == 0
    with open(os.path.join(out, T.EXPORT_FILES["train"]), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_single_prefix_and_no_extras_give_the_full_analysis_rows_only():
    with tempfile.TemporaryDirectory() as tmp:
        _dataset(tmp)
        rows = _main_export(tmp, "--no_dictionary_pairs", "--single_prefix", "--gold_upsample", "1",
                            "--gold_nominal_upsample", "1", "--no_words_without_root")
        assert len(rows) == 36 + 7
        assert all(set(r) == {"input_text", "target_text", "is_gold"} and r["input_text"].startswith("analyze: ")
                   for r in rows)
        rows = _main_export(tmp, "--no_dictionary_pairs")
        root_only = [r for r in rows if r.get("task") == D.ROOT_ONLY_TASK]
        assert len(rows) == 36 + 3 * 4 + 4 * 5 + len(root_only) and len(root_only) == 55
        assert all(r["input_text"].startswith("root: ") and r["target_text"] == "root: none" for r in root_only)
        rows = _main_export(tmp, "--no_dictionary_pairs", "--single_prefix")
        assert all(r["input_text"].startswith("analyze: ") for r in rows)


def test_export_row_keeps_the_old_shape_for_full_analysis():
    s = {"input_text": "analyze: x", "target_text": "t", "is_gold": True, "task": D.ANALYZE_TASK, "source": "gold_verb"}
    assert T.export_row(s) == {"input_text": "analyze: x", "target_text": "t", "is_gold": True}
    r = {"input_text": "root: x", "target_text": "root: none", "is_gold": False, "task": D.ROOT_ONLY_TASK,
         "source": "word_without_root"}
    assert T.export_row(r) == r


def test_root_accuracy_helpers():
    assert T.root_in_output("root: a-b-c | tense: past") == "a-b-c"
    assert T.root_in_output("root: a-b-c") == "a-b-c" and T.root_in_output("tense: past") == ""
    items = [("analyze: x", "a-b-c"), ("analyze: y", "d-e-f"), ("root: z", "g-h-i")]
    answers = {"analyze: x": "root: a-b-c | tense: past", "analyze: y": "root: d-e | tense: past",
               "root: z": "root: g-h-i"}
    seen = []

    def predict(texts):
        seen.append(len(texts))
        return [answers[t] for t in texts]
    assert T.root_accuracy(predict, items, 2) == pytest.approx(2 / 3) and seen == [2, 1]
    assert T.root_accuracy(predict, [], 2) == 0.0
    samples = [{"input_text": "root: z", "target_text": "root: g-h-i"}]
    assert T.sample_items(samples) == [("root: z", "g-h-i")]
    with tempfile.TemporaryDirectory() as tmp:
        _dataset(tmp)
        items = T.root_accuracy_items(tmp, 1000)
        assert len(items) == 1 and items[0][0].startswith("analyze: ") and T.root_accuracy_items(tmp, 0) == []


def test_parse_args_new_flags():
    a = T.parse_args([])
    assert (a.gold_upsample, a.gold_nominal_upsample, a.root_accuracy_rows, a.eval_steps) == (4, 5, 0, 0)
    assert not (a.single_prefix or a.no_words_without_root or a.metric_for_best_model)
    a = T.parse_args(["--root_accuracy_rows", "1000", "--metric_for_best_model", "root_accuracy"])
    assert a.metric_for_best_model == "root_accuracy"
    for bad in (["--metric_for_best_model", "root_accuracy"], ["--gold_upsample", "0"]):
        with pytest.raises(SystemExit):
            T.parse_args(bad)


# ---------------------------------------------------------------- the dataset checkout

def _need_dataset():
    if not os.path.exists(D.gold_file_path(DATASET_DIR, "val")):
        pytest.skip(f"no dataset checkout at {DATASET_DIR}")


def test_checkout_words_without_root_are_not_gold_eval_forms():
    _need_dataset()
    words = D.read_words_without_root()
    assert not set(words) & H.gold_eval_forms(DATASET_DIR)
    stats = collections.Counter()
    assert [p["form"] for p in D.load_words_without_root(DATASET_DIR, stats)] == list(words)
    assert stats == {"loaded": len(words)}


def test_checkout_proclitic_letter_verbs_are_trained_unsplit():
    """Verbs whose first radical is w, l, b or z, with forms that start with that radical, are in the gold
    train and synthetic pairs, so a leading letter of this kind is not always a proclitic."""
    _need_dataset()
    found = collections.Counter()
    for s in T.load_data(DATASET_DIR):
        if s["source"] not in ("synthetic", "gold_verb"):
            continue
        first = geez_script.root_symbols(T.root_in_output(s["target_text"]))[0]
        form = s["input_text"][len(D.TASK_PREFIXES[D.ANALYZE_TASK]):]
        if first in PROCLITIC_LETTERS and geez_script.decompose_char(form[0])[0] == first:
            found[(s["source"], first)] += 1
    assert all(found[(source, c)] > 0 for source in ("synthetic", "gold_verb") for c in PROCLITIC_LETTERS)
