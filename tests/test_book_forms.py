"""data/lexicon/book_forms.jsonl: forms printed in the other books load by split and status."""
import collections
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dataset_loader  # noqa: E402
import evaluate_gold  # noqa: E402
import train_byt5_morphology as T  # noqa: E402

ROWS = [
    {"root": "ቀ-ተ-ለ", "form": "ቀተልክዎ", "type": "verb", "person": "1", "gender": "c", "number": "s", "tense": "past",
     "stem": "basic", "object": "3ms", "status": "", "split": "train"},
    {"root": "ቀ-ጸ-ረ", "form": "ቀጻርያን", "type": "adjective", "gender": "m", "number": "p", "status": "", "split": "train"},
    {"root": "ሀ-ለ-ወ", "form": "ሀልዉ", "type": "verb", "person": "2", "gender": "m", "number": "p",
     "tense": "imperative", "stem": "basic", "status": "", "split": "heldout"},
    {"root": "ቀ-ተ-ለ", "form": "ቀተልዎ", "type": "verb", "status": "CHECK", "split": "train"},
    {"root": None, "form": "ውእቱ", "type": "pronoun", "status": "NO_ROOT", "split": "train"},
    {"root": "ፈ-ተ-ሐ", "form": "ፈትሐ", "type": "verb", "person": "3", "gender": "m", "number": "s", "status": "",
     "split": "train"},   # no tense: does not load
]


def _dataset(tmp):
    os.makedirs(os.path.join(tmp, "data", "lexicon"))
    with open(os.path.join(tmp, dataset_loader.BOOK_FORMS_FILE), "w", encoding="utf-8") as f:
        for r in ROWS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return tmp


def test_load_by_split_and_status():
    with tempfile.TemporaryDirectory() as tmp:
        stats = collections.Counter()
        train = dataset_loader.load_book_forms(_dataset(tmp), "train", stats)
        assert [p["form"] for p in train] == ["ቀተልክዎ", "ቀጻርያን"]
        assert stats["not loaded: status CHECK"] == 1 and stats["not loaded: status NO_ROOT"] == 1
        assert stats["not loaded: type or labels missing"] == 1
        assert [p["form"] for p in dataset_loader.load_book_forms(tmp, "heldout")] == ["ሀልዉ"]


def test_targets_and_samples():
    with tempfile.TemporaryDirectory() as tmp:
        samples = T.load_book_form_samples(_dataset(tmp))
        assert [s["target_text"] for s in samples] == [
            "root: ቀ-ተ-ለ | tense: past | person: 1 | gender: c | number: s | stem: basic | object: 3ms",
            "root: ቀ-ጸ-ረ | type: adjective | gender: m | number: p"]
        assert all(s["source"] == "book_forms" and s["is_gold"] for s in samples)


def test_books_eval_split():
    with tempfile.TemporaryDirectory() as tmp:
        pairs = evaluate_gold.load_gold_pairs(_dataset(tmp), "books")
        assert [(p["surface"], p["root"], p["tense"]) for p in pairs] == [("ሀልዉ", "ሀ-ለ-ወ", "imperative")]


def test_missing_file_gives_no_pairs():
    with tempfile.TemporaryDirectory() as tmp:
        assert dataset_loader.load_book_forms(tmp, "train") == []
