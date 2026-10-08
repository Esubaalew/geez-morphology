"""split_heldout_synthetic.py: lines whose root is a gold val/test root move to the held-out files, in order, with
their analyze mirror; nothing is lost. The trainer reads data/paradigms/synthetic_morphology_train.jsonl only."""
import json
import os
import tempfile

import split_heldout_synthetic as H
import train_byt5_morphology as T


def _line(root, form, tense="past"):
    f = {"gender": "m", "number": "s", "person": "3", "stem": "basic", "tense": tense, "verb_type": "strong_A", "root": root}
    return json.dumps({"features": f, "surface_form": form}, ensure_ascii=False) + "\n", \
        json.dumps({"surface_form": form, "analysis": f}, ensure_ascii=False) + "\n"


def _setup(tmp):
    os.makedirs(os.path.join(tmp, "ds", "data", "paradigms"))
    os.makedirs(os.path.join(tmp, "full"))
    val = {"root": "ሰ-መ-ዐ", "table_data": [{"geez_form": "ሰምዐ"}]}
    test = {"root": "", "table_data": [{"geez_form": "ቀተለ", "root": "ቀ-ተ-ለ / ቀ-ጠ-ለ"}]}
    for name, rec in (("merho_ges_gold_val.jsonl", val), ("merho_ges_gold_test.jsonl", test)):
        with open(os.path.join(tmp, "ds", "data", "paradigms", name), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    lines = [_line("g-b-r", "ገብረ"), _line("s-m-a2", "ሰምዐ"), _line("q-t-l", "ቀተለ"), _line("q-T-l", "ቀጠለ"), _line("n-g-r", "ነገረ")]
    with open(os.path.join(tmp, "full", H.TRAIN), "w", encoding="utf-8") as ft, \
            open(os.path.join(tmp, "full", H.ANALYZE), "w", encoding="utf-8") as fa:
        for t, a in lines:
            ft.write(t); fa.write(a)
    return lines


def test_split_moves_val_test_root_lines_in_order():
    with tempfile.TemporaryDirectory() as tmp:
        lines = _setup(tmp)
        ds = os.path.join(tmp, "ds")
        moved, _ = H.split_into_dataset(os.path.join(tmp, "full"), ds)
        tr, an, htr, han = [open(p, encoding="utf-8").read() for p in H.targets(ds)]
        assert tr == lines[0][0] + lines[4][0] and an == lines[0][1] + lines[4][1]
        assert htr == lines[1][0] + lines[2][0] + lines[3][0] and han == lines[1][1] + lines[2][1] + lines[3][1]
        assert sum(moved.values()) == 3 and moved[("val", "ሰ-መ-ዐ")] == 1
        _, differ = H.split_into_dataset(os.path.join(tmp, "full"), ds, write=False)
        assert differ == []                                      # reproducible


def test_trainer_reads_the_clean_train_file_only():
    assert T.SYNTHETIC_FILE == os.path.join("data", "paradigms", "synthetic_morphology_train.jsonl")
    assert H.HELDOUT_DIR not in T.SYNTHETIC_FILE
