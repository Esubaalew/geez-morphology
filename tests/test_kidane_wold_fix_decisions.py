"""kidane_wold_triples.build() keeps the decisions recorded in the previous root fixes file.

A row of the previous fixes file with new_root null means "root unknown, don't guess" (entry 7546). A rebuild
writes that entry's row with new_root null and that row's root_notes; if the rebuild finds no row for the
entry, the previous row is kept as it is. root_notes of a previous row are kept on the rebuilt row.
Headwords are built from root symbols with geez_script, never typed.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import geez_script as gc  # noqa: E402
import kidane_wold_triples as kw  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402

NOTE = "CHECK: merged entry"


def word(*units):
    return "".join(gc.render_unit(c, v) for c, v in units)


def entry(i, hw, inferred, pos="noun"):
    return {"id": i, "headword_geez": hw, "pos": pos, "inferred_root": inferred, "amharic_definition": ""}


def run(tmp_path, entries, previous=None):
    d = tmp_path / "dict.jsonl"
    d.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    prev = None
    if previous is not None:
        prev = tmp_path / "previous_fixes.jsonl"
        prev.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in previous), encoding="utf-8")
    out = tmp_path / "fixes.jsonl"
    kw.build(str(d), str(tmp_path / "triples.jsonl"), str(out), previous_fixes_path=str(prev) if prev else None)
    return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]


def test_restored_root_without_decision(tmp_path):
    hw = word(("m", "I"), ("a", "u"), ("a2", "I"))
    assert run(tmp_path, [entry(1, hw, "m-a")]) == [
        {"id": 1, "headword": hw, "pos": "noun", "old_root": "m-a", "new_root": "m-a-a2",
         "method": "restored_dropped_letter", "evidence": "headword"}]


def test_null_decision_on_a_rebuilt_row(tmp_path):
    hw = word(("m", "I"), ("a", "u"), ("a2", "I"))
    prev = [{"id": 1, "headword": hw, "pos": "noun", "old_root": "m-a", "new_root": None,
             "method": "restored_dropped_letter", "evidence": "headword", "root_notes": NOTE}]
    assert run(tmp_path, [entry(1, hw, "m-a")], prev) == prev


def test_null_decision_kept_when_the_rebuild_finds_no_row(tmp_path):
    hw = word(("m", "I"), ("a", "u"), ("a2", "I"))
    prev = [{"id": 1, "headword": hw, "pos": "noun", "old_root": "m-a", "new_root": None,
             "method": "restored_dropped_letter", "evidence": "headword", "root_notes": NOTE}]
    other = word(("q", "a"), ("t", "a"), ("l", "a"))
    rows = run(tmp_path, [entry(5, other, None), entry(1, hw, None), entry(7, other, None)], prev)
    assert rows == prev


def test_root_notes_kept_on_a_rebuilt_row(tmp_path):
    hw = word(("m", "I"), ("a", "u"), ("a2", "I"))
    prev = [{"id": 1, "headword": hw, "pos": "noun", "old_root": "m-a", "new_root": "m-a-a2",
             "method": "restored_dropped_letter", "evidence": "headword", "root_notes": "checked"}]
    assert run(tmp_path, [entry(1, hw, "m-a")], prev) == prev


def test_rows_without_a_decision_are_not_carried(tmp_path):
    hw = word(("m", "I"), ("a", "u"), ("a2", "I"))
    prev = [{"id": 9, "headword": hw, "pos": "noun", "old_root": "m-a", "new_root": "m-a-a2",
             "method": "restored_dropped_letter", "evidence": "headword"}]
    assert run(tmp_path, [entry(1, hw, None)], prev) == []


FIXES = os.path.join(DATASET_DIR, "data", "lexicon", "kidane_wold_root_fixes.jsonl")
KW = os.path.join(DATASET_DIR, "data", "lexicon", "classical_dictionary_kidane_wold.jsonl")


@pytest.mark.skipif(not os.path.exists(FIXES), reason="no geez-dataset checkout")
def test_rebuild_on_the_dataset_keeps_the_7546_row(tmp_path):
    with open(FIXES, encoding="utf-8") as f:
        committed = {json.loads(l)["id"]: l.rstrip("\n") for l in f}
    out = tmp_path / "fixes.jsonl"
    kw.build(KW, str(tmp_path / "triples.jsonl"), str(out), previous_fixes_path=FIXES)
    rebuilt = {json.loads(l)["id"]: l for l in out.read_text(encoding="utf-8").splitlines()}
    assert rebuilt[7546] == committed[7546]
    assert json.loads(rebuilt[7546])["new_root"] is None and json.loads(rebuilt[7546])["root_notes"].startswith("CHECK:")
