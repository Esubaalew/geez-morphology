"""Root-level gate: a synthetic root that is a gold val/test root up to spelling is held out like the gold root.

- split_heldout_synthetic.fold_root folds a 1st-order root with geez_script.fold_spelling: the labiovelar codes
  (qh, kh2, gw, khw) fold to their plain codes and the twin codes (s2, S2, h2, kh, a2) to s, S, h1, h1, a.
- Gold-backed: gold train and gold val/test roots that differ only by such codes (read by record_id) fold to one root.
- A line whose folded root is a folded gold val/test root moves to the held-out files with its analyze line,
  in order, under the split of that gold root; the other lines do not move. --check fails on such a line in the
  training pair, and the dataset's training pair has none.
Roots are written with the Latin consonant symbols; the test types no Ge'ez letters.
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

import geez_script
import split_heldout_synthetic as H
from conftest import DATASET_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARADIGMS = os.path.join(DATASET_DIR, "data", "paradigms")

# Tests that read the dataset checkout: --dataset_dir, else $GEEZ_DATASET_DIR, else ../geez-dataset (conftest.py)
needs_dataset = pytest.mark.skipif(
    not os.path.isdir(PARADIGMS),
    reason=f"no geez-dataset checkout at {DATASET_DIR}; set GEEZ_DATASET_DIR or pass --dataset_dir")
GOLD_FILES = {"train": "merho_ges_gold_train.jsonl", "val": "merho_ges_gold_val.jsonl", "test": "merho_ges_gold_test.jsonl"}

# (gold train record_id, gold val/test record_id): the record roots differ only by labiovelar or twin codes
GOLD_ROOT_PAIRS = [
    ("624fff1d0291", "7a01b50bb505"),   # q-S-l / qh-S-l
    ("4ae71fecf407", "a588575abb7f"),   # g-r-h1 / gw-r-h2
]


def _eth(root):
    return geez_script.first_order_root(geez_script.to_ethiopic_root(root))


def _record_roots():
    roots = {}
    for name in GOLD_FILES.values():
        with open(os.path.join(PARADIGMS, name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                roots[rec["record_id"]] = rec.get("root") or ""
    return roots


@needs_dataset
def test_gold_root_pairs_fold_to_one_root():
    roots = _record_roots()
    for train_id, eval_id in GOLD_ROOT_PAIRS:
        a, b = _eth(roots[train_id]), _eth(roots[eval_id])
        assert a != b and H.fold_root(a) == H.fold_root(b), (train_id, eval_id)
        assert H.fold_root(b) in H.gold_eval_roots_folded(DATASET_DIR)


def test_fold_root_codes():
    plain = {}
    for symbol, cp in geez_script.CONSONANTS.items():
        folded = geez_script.decompose_char(H.fold_root(chr(cp)))[0]
        if folded != symbol:
            plain[symbol] = folded
    assert plain == {"qh": "q", "kh2": "k", "gw": "g", "khw": "h1", "s2": "s", "S2": "S", "h2": "h1", "kh": "h1", "a2": "a"}
    assert H.fold_root(_eth("q-t-l")) == _eth("q-t-l")
    assert H.fold_root(_eth("d-kh2-m")) == H.fold_root(_eth("d-k-m")) != H.fold_root(_eth("d-g-m"))


def _line(root, form):
    f = {"gender": "m", "number": "s", "person": "3", "stem": "basic", "tense": "past", "verb_type": "strong_A", "root": root}
    return json.dumps({"features": f, "surface_form": form}, ensure_ascii=False) + "\n", \
        json.dumps({"surface_form": form, "analysis": f}, ensure_ascii=False) + "\n"


def _word(*units):
    return "".join(geez_script.render_unit(c, v) for c, v in units)


def _setup(tmp):
    os.makedirs(os.path.join(tmp, "ds", "data", "paradigms"))
    os.makedirs(os.path.join(tmp, "full"))
    val = {"record_id": "v1", "root": _eth("w-S-a"), "table_data": [{"geez_form": _word(("w", "a"), ("S", "a"), ("a", "a"))}]}
    test = {"record_id": "t1", "root": _eth("qh-l-w"), "table_data": [{"geez_form": _word(("qh", "a"), ("l", "a"), ("w", "a"))}]}
    for name, rec in ((GOLD_FILES["val"], val), (GOLD_FILES["test"], test)):
        with open(os.path.join(tmp, "ds", "data", "paradigms", name), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    lines = [
        _line("g-b-r", _word(("g", "a"), ("b", "I"), ("r", "a"))),        # kept
        _line("w-S2-a", _word(("w", "a"), ("S2", "I"), ("a", "a"))),      # twin of the val root: held out
        _line("q-l-w", _word(("q", "a"), ("l", "I"), ("w", "o"))),        # plain code of the test root: held out
        _line("w-s-a", _word(("w", "a"), ("s", "I"), ("a", "a"))),        # another root: kept
    ]
    with open(os.path.join(tmp, "full", H.TRAIN), "w", encoding="utf-8") as ft, \
            open(os.path.join(tmp, "full", H.ANALYZE), "w", encoding="utf-8") as fa:
        for t, a in lines:
            ft.write(t)
            fa.write(a)
    return lines


def _check(tmp):
    return subprocess.run([sys.executable, os.path.join(ROOT, "split_heldout_synthetic.py"), "--full_dir", os.path.join(tmp, "full"),
                           "--dataset_dir", os.path.join(tmp, "ds"), "--check"], capture_output=True, text=True, cwd=ROOT)


def test_split_holds_out_folded_root_lines():
    with tempfile.TemporaryDirectory() as tmp:
        lines = _setup(tmp)
        ds = os.path.join(tmp, "ds")
        moved, _, dropped = H.split_into_dataset(os.path.join(tmp, "full"), ds)
        tr, an, htr, han = [open(p, encoding="utf-8").read() for p in H.targets(ds)]
        assert tr == lines[0][0] + lines[3][0] and an == lines[0][1] + lines[3][1]
        assert htr == lines[1][0] + lines[2][0] and han == lines[1][1] + lines[2][1]
        assert dict(moved) == {("val", _eth("w-S2-a")): 1, ("test", _eth("q-l-w")): 1} and dropped == []
        ok = _check(tmp)
        assert ok.returncode == 0, ok.stdout + ok.stderr
        assert "folded root is a folded gold val/test root: 0" in ok.stdout
        assert H.folded_root_lines(ds) == {}


def test_check_fails_on_a_folded_root_in_the_training_pair():
    with tempfile.TemporaryDirectory() as tmp:
        lines = _setup(tmp)
        ds = os.path.join(tmp, "ds")
        H.split_into_dataset(os.path.join(tmp, "full"), ds)
        for p, line in zip(H.targets(ds)[:2], lines[2]):                  # a line written past the split
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
        assert sum(len(v) for v in H.folded_root_lines(ds).values()) == 2
        bad = _check(tmp)
        assert bad.returncode == 1
        assert "folded root is a folded gold val/test root: 2" in bad.stdout


@needs_dataset
def test_dataset_training_pair_has_no_folded_gold_root():
    assert H.folded_root_lines(DATASET_DIR) == {}
