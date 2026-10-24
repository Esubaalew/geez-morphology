"""Held-out clitic spellings must be protected even when their bare forms differ."""
import json
import sys
import pytest

import dataset_loader as D
import generate_synthetic_data as G
import split_heldout_synthetic as S
import training_mixture as TM
from scripts.audit_train_leakage import heldout
import train_byt5_morphology as T
from scripts import audit_train_leakage as audit


def test_clitic_holdouts_protect_training_and_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "gold_eval_forms", lambda _: set())
    monkeypatch.setattr(S, "gold_eval_roots", lambda _: {})
    monkeypatch.setattr(G, "read_blocked_forms_file", lambda: set())
    monkeypatch.setattr(G, "read_held_out_roots_file", lambda: set())
    monkeypatch.setattr(D, "held_out_synthetic_roots", lambda _: set())
    monkeypatch.setattr(D, "load_gold_split", lambda *a: [])
    path = tmp_path / TM.CLITIC_FILE
    path.parent.mkdir(parents=True)
    row = dict(id="clitic-test", form="ወመኄርው", root="ኀ-ረ-ወ", word_class="verb",
               source="clitic_rule", status="", split="test", book="test", page=1)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    gate = TM.HeldOut(str(tmp_path))
    assert gate.leaks(row["form"], "ኀ-የ-ረ-ወ") == "form"
    assert gate.leaks("ሌላ", row["root"]) == "root"
    assert (row["form"], row["root"]) in heldout(str(tmp_path))["test"]


def test_audit_uses_selected_unified_file(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "load_gold_split", lambda *a: [])
    monkeypatch.setattr(D, "UNIFIED_FILE", "corrected.jsonl")
    row = dict(id="corrected", form="ሰላም", root=None, word_class="noun", number="s",
               source="book", status="", split="val", book="test", page=1)
    (tmp_path / "corrected.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert (row["form"], None) in heldout(str(tmp_path))["val"]


def test_legacy_trainer_filters_gold_homographs_and_all_sources():
    rows = [dict(input_text="analyze: ወመኄርው", target_text="root: ኀ-የ-ረ-ወ", is_gold=True),
            dict(input_text="root: ሌላ", target_text="root: ኀ-ረ-ወ", is_gold=False),
            dict(input_text="analyze: ቀተለ", target_text="root: ቀ-ተ-ለ", is_gold=True)]
    assert T.filter_heldout_samples(rows, [("ወመኄርው", "ኀ-ረ-ወ")]) == rows[2:]


def test_audit_rejects_spelling_variant_forms(tmp_path, monkeypatch):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(dict(input_text="analyze: ሰላም", target_text="root: none")))
    monkeypatch.setattr(D, "require_dataset_dir", lambda d: d)
    monkeypatch.setattr(audit, "heldout", lambda _: {"test": [("ሠላም", None)]})
    monkeypatch.setattr(sys, "argv", ["audit", "--dataset_dir", str(tmp_path), "--train", str(path)])
    with pytest.raises(SystemExit) as exc:
        audit.main()
    assert exc.value.code == 1
