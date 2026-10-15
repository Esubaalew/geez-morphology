"""Near-duplicate filter: training lines whose surface form equals a gold val/test form up to spelling are dropped.

- geez_script.fold_spelling makes every labiovelar letter plain and puts every twin letter (s2, S2, h2, kh, a2) in its
  kept row (s, S, h1, h1, a), vowel order kept. Its tables come from the Unicode names and from CONSONANTS.
- Gold-backed: pairs of gold forms (read from the gold files by record_id) that differ only by such letters fold to
  the same string.
- split_heldout_synthetic drops the matching lines from the training pair only; the other lines keep their bytes
  and order, the held-out lines are not filtered, and a second run drops nothing (idempotent).
- After the split of the dataset's training pair no training line folds to a gold val/test form.
Forms are built from consonant symbols or read from the gold files; the test types no Ge'ez letters.
"""
import json
import os
import shutil
import tempfile
import unicodedata

import geez_script
import split_heldout_synthetic as H
from conftest import DATASET_DIR

PARADIGMS = os.path.join(DATASET_DIR, "data", "paradigms")
GOLD_FILES = {"train": "merho_ges_gold_train.jsonl", "val": "merho_ges_gold_val.jsonl", "test": "merho_ges_gold_test.jsonl"}

# ((record_id, row, variant), (record_id, row, variant)): two gold forms that differ only by labiovelar or twin letters
GOLD_FOLD_PAIRS = [
    (("624fff1d0291", 1, 0), ("7a01b50bb505", 1, 0)),   # q-S-l train / qh-S-l test: labiovelar 6th order
    (("4ae71fecf407", 0, 0), ("a588575abb7f", 0, 0)),   # g-r-h1 train / gw-r-h2 test: labiovelar 1st order and h1/h2
    (("662ebd7932ec", 0, 0), ("7908db250bfb", 1, 0)),   # h2 against the labiovelar of kh
    (("d1693b26c18e", 0, 1), ("d1693b26c18e", 0, 2)),   # s2 / s, two spellings of one cell
    (("a2d0b6feed4e", 2, 0), ("9761133fe07c", 5, 0)),   # h2 / kh
    (("f69e0ca24611", 1, 0), ("084c4e9397e6", 0, 0)),   # a / a2
    (("3e3c776de4ca", 1, 0), ("e35dfd544f74", 5, 0)),   # S / S2
]
LABIOVELAR_PAIRS = 3     # the first three pairs have a labiovelar letter on one side


def _gold_records():
    recs = {}
    for name in GOLD_FILES.values():
        with open(os.path.join(PARADIGMS, name), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                recs[rec["record_id"]] = rec
    return recs


def _gold_form(recs, record_id, row, variant):
    return H.form_variants(recs[record_id]["table_data"][row]["geez_form"])[variant]


def test_gold_pairs_fold_to_one_form():
    recs = _gold_records()
    for n, (a, b) in enumerate(GOLD_FOLD_PAIRS):
        fa, fb = _gold_form(recs, *a), _gold_form(recs, *b)
        assert fa != fb and len(fa) == len(fb), (a, b)
        assert geez_script.fold_spelling(fa) == geez_script.fold_spelling(fb), (a, b)
        has_labiovelar = any(ch in geez_script.LABIOVELAR_TO_PLAIN for ch in fa + fb)
        assert has_labiovelar == (n < LABIOVELAR_PAIRS), (a, b)


def test_fold_tables_and_vowel_order():
    lab, twin = geez_script.LABIOVELAR_TO_PLAIN, geez_script.TWIN_TO_ONE_ROW
    assert len(lab) == 6 * 5 and len(twin) == 5 * 7
    for letter, plain in lab.items():
        name = unicodedata.name(letter)
        assert "".join(name.rsplit("W", 1)) == unicodedata.name(plain), name
    for twin_symbol, kept_symbol in geez_script.TWIN_ROWS.items():
        for order in range(7):
            letter = geez_script.render_unit(twin_symbol, geez_script.ORDER_TO_VOWEL[order])
            assert twin[letter] == geez_script.render_unit(kept_symbol, geez_script.ORDER_TO_VOWEL[order])
    for symbol in ("qh", "kh2", "gw", "khw"):                         # the labiovelar rows with a consonant symbol
        for vowel in ("a", "i", "A", "e", "I"):
            letter = geez_script.render_unit(symbol, vowel)
            assert letter in lab and geez_script.decompose_char(geez_script.fold_spelling(letter))[1] == vowel
    for cp, out in geez_script.SPELLING_FOLD.items():
        assert cp % 8 == ord(out) % 8                                  # vowel order kept
        assert ord(out) not in geez_script.SPELLING_FOLD               # one step reaches the folded letter
    plain = "".join(geez_script.render_unit(c, "a") for c in ("q", "t", "l", "s", "S", "h1", "a"))
    assert geez_script.fold_spelling(plain) == plain
    assert geez_script.fold_spelling("q-t-l 3ms") == "q-t-l 3ms" and geez_script.fold_spelling("") == ""


def test_fold_is_idempotent():
    letters = "".join(chr(cp) for cp in range(0x1200, 0x1380))
    once = geez_script.fold_spelling(letters)
    assert geez_script.fold_spelling(once) == once
    for rec in _gold_records().values():
        for row in rec["table_data"]:
            for form in H.form_variants(row.get("geez_form") or ""):
                folded = geez_script.fold_spelling(form)
                assert geez_script.fold_spelling(folded) == folded


def _word(*units):
    return "".join(geez_script.render_unit(c, v) for c, v in units)


def _line(root, form):
    f = {"gender": "m", "number": "s", "person": "3", "stem": "basic", "tense": "past", "verb_type": "strong_A", "root": root}
    return json.dumps({"features": f, "surface_form": form}, ensure_ascii=False) + "\n", \
        json.dumps({"surface_form": form, "analysis": f}, ensure_ascii=False) + "\n"


def test_split_drops_near_duplicates_from_training_pair_only():
    gold_val = _word(("qh", "I"), ("S", "u"), ("l", "I"))             # labiovelar first letter
    gold_test = _word(("n", "a"), ("s", "a"), ("k", "a"))
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "ds", "data", "paradigms"))
        os.makedirs(os.path.join(tmp, "full"))
        # the gold records carry roots of their own, so the root gate does not move the training lines below
        val = {"record_id": "v1", "root": geez_script.to_ethiopic_root("m-r-t"), "table_data": [{"geez_form": gold_val}]}
        test = {"record_id": "t1", "root": geez_script.to_ethiopic_root("b-r-k"), "table_data": [{"geez_form": gold_test}]}
        for name, rec in ((GOLD_FILES["val"], val), (GOLD_FILES["test"], test)):
            with open(os.path.join(tmp, "ds", "data", "paradigms", name), "w", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        lines = [
            _line("g-b-r", _word(("g", "a"), ("b", "I"), ("r", "a"))),             # kept
            _line("q-S-l", _word(("q", "I"), ("S", "u"), ("l", "I"))),             # plain letter of gold_val: dropped
            _line("n-s2-k", _word(("n", "a"), ("s2", "a"), ("k", "a"))),           # twin of gold_test: dropped
            _line("q-S-l", _word(("q", "a"), ("S", "u"), ("l", "I"))),             # other vowel order: kept
            _line("b-r-k", _word(("n", "a"), ("s2", "a"), ("k", "a"))),            # gold test root: held out, not filtered
            _line("n-g-r", _word(("n", "a"), ("g", "a"), ("r", "a"))),             # kept
        ]
        with open(os.path.join(tmp, "full", H.TRAIN), "w", encoding="utf-8") as ft, \
                open(os.path.join(tmp, "full", H.ANALYZE), "w", encoding="utf-8") as fa:
            for t, a in lines:
                ft.write(t)
                fa.write(a)
        ds = os.path.join(tmp, "ds")
        report = os.path.join(tmp, "near_duplicates.jsonl")
        moved, _, dropped = H.split_into_dataset(os.path.join(tmp, "full"), ds, near_duplicate_report=report)
        tr, an, htr, han = [open(p, encoding="utf-8").read() for p in H.targets(ds)]
        assert dropped == [lines[1][0], lines[2][0]]
        assert tr == "".join(lines[i][0] for i in (0, 3, 5)) and an == "".join(lines[i][1] for i in (0, 3, 5))
        assert htr == lines[4][0] and han == lines[4][1] and sum(moved.values()) == 1
        assert H.near_duplicate_lines(ds) == {} and H.gold_form_lines(ds) == {}
        with open(report, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f]
        assert [(r["root"], r["gold_val_test_forms"][0]["record_id"]) for r in rows] == [("q-S-l", "v1"), ("n-s2-k", "t1")]
        assert all(r["folded_form"] == geez_script.fold_spelling(r["gold_val_test_forms"][0]["form"]) for r in rows)
        _, differ, _ = H.split_into_dataset(os.path.join(tmp, "full"), ds, write=False)
        assert differ == []                                                   # reproducible
        # a second pass over the filtered files drops nothing and changes nothing
        kept, again = H.drop_near_duplicates(tr.splitlines(keepends=True), H.gold_eval_forms_folded(ds))
        assert again == [] and "".join(kept) == tr
        # a near-duplicate planted in the dataset training file is found by the check
        with open(H.targets(ds)[0], "a", encoding="utf-8") as f:
            f.write(lines[1][0])
        assert sum(len(v) for v in H.near_duplicate_lines(ds).values()) == 1


def test_dataset_training_pair_has_no_near_duplicate_after_split():
    """The dataset's training pair, split again with its gold val/test files, has no line that folds to a gold form,
    and the lines it keeps are the dataset's lines in order."""
    with tempfile.TemporaryDirectory() as tmp:
        full, ds = os.path.join(tmp, "full"), os.path.join(tmp, "ds")
        os.makedirs(full)
        os.makedirs(os.path.join(ds, "data", "paradigms"))
        for name in (H.TRAIN, H.ANALYZE):
            shutil.copyfile(os.path.join(PARADIGMS, name), os.path.join(full, name))
        for name in (GOLD_FILES["val"], GOLD_FILES["test"]):
            shutil.copyfile(os.path.join(PARADIGMS, name), os.path.join(ds, "data", "paradigms", name))
        _, _, dropped = H.split_into_dataset(full, ds)
        assert H.near_duplicate_lines(ds) == {}
        with open(os.path.join(full, H.TRAIN), encoding="utf-8") as f:
            before = f.readlines()
        with open(H.targets(ds)[0], encoding="utf-8") as f:
            after = f.readlines()
        dropped_set = set(dropped)
        assert after == [x for x in before if x not in dropped_set]
