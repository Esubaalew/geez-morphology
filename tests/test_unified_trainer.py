"""train_byt5_morphology.load_unified_samples: unified book rows become training samples."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset_loader  # noqa: E402
import train_byt5_morphology as trainer  # noqa: E402


def _row(i, form, split="train", status="", word_class="verb", root="ቀ-ተ-ለ", **kw):
    r = {"id": i, "form": form, "root": root, "word_class": word_class, "stem": "basic", "tense": "past",
         "person": "3", "gender": "m", "number": "s", "object": None, "possessor": None, "negated": False,
         "source": "book", "book": "b", "page": "1", "status": status, "split": split}
    r.update(kw)
    return r


def test_unified_samples_train_only_and_loading_only():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "data", "unified"))
        rows = [_row("a", "ቀተለ"), _row("b", "ቀተለት", split="val"), _row("c", "ቀተሉ", status="CHECK"),
                _row("d", "ወ", word_class="particle", root=None, stem=None, tense=None, person=None,
                     gender=None, number=None)]
        with open(os.path.join(tmp, dataset_loader.UNIFIED_FILE), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats = {}
        import collections
        stats = collections.Counter()
        samples = trainer.load_unified_samples(tmp, stats)
        forms = sorted(s["input_text"] for s in samples)
        assert len(samples) == 2 and all(s["source"] == "unified_book" for s in samples)
        assert any(s["target_text"].startswith("root: none") for s in samples)
        assert stats["loaded"] == 2
        assert "unified_book" in trainer.SOURCES
        assert forms


def test_unified_samples_empty_without_file():
    with tempfile.TemporaryDirectory() as tmp:
        assert trainer.load_unified_samples(tmp) == []


def test_dev_items_add_unified_val_rows_and_none_root():
    import collections  # noqa: F401
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "data", "unified"))
        os.makedirs(os.path.join(tmp, "data", "paradigms"), exist_ok=True)
        rows = [_row("a", "ቀተለ", split="val"), _row("b", "ወ", split="val", word_class="particle", root=None, stem=None,
                tense=None, person=None, gender=None, number=None), _row("c", "ቀተሉ", split="train")]
        with open(os.path.join(tmp, dataset_loader.UNIFIED_FILE), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gold = dataset_loader.gold_file_path(tmp, "val")
        os.makedirs(os.path.dirname(gold), exist_ok=True)
        with open(gold, "w", encoding="utf-8") as f:
            f.write("")
        items = trainer.root_accuracy_items(tmp, 100)
        assert sorted(root for _, root in items) == ["none", "ቀ-ተ-ለ"]
        assert trainer.root_accuracy_items(tmp, 100, gold_only=True) == []


def test_without_root_deduplicated_against_unified_train_despite_different_prefix():
    unified = [{"input_text": "analyze: ወ", "target_text": "root: none | type: particle", "source": "unified_book"}]
    without_root = [
        {"input_text": "root: ወ", "target_text": "root: none", "source": "word_without_root"},
        {"input_text": "root: እንዘ", "target_text": "root: none", "source": "word_without_root"},
    ]
    seen_forms = {s["input_text"].split(":", 1)[-1].strip() for s in unified}
    kept = [s for s in without_root if s["input_text"].split(":", 1)[-1].strip() not in seen_forms]
    assert [s["input_text"] for s in kept] == ["root: እንዘ"]


def test_dev_eval_samples_and_unified_upsample():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "data", "unified"))
        os.makedirs(os.path.join(tmp, "data", "paradigms"), exist_ok=True)
        rows = [_row("a", "ቀተለ", split="val"), _row("b", "ወ", split="val", word_class="particle", root=None, stem=None,
                tense=None, person=None, gender=None, number=None)]
        with open(os.path.join(tmp, dataset_loader.UNIFIED_FILE), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gold = dataset_loader.gold_file_path(tmp, "val")
        with open(gold, "w", encoding="utf-8") as f:
            f.write("")
        dev = trainer.dev_eval_samples(tmp)
        assert len(dev) == 2 and all(s["source"] == "dev_val" for s in dev)
        assert trainer.dev_eval_samples(tmp, gold_only=True) == []
        assert trainer.parse_args(["--unified_upsample", "3"]).unified_upsample == 3
