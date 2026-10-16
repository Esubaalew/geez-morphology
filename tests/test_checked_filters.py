import csv
import os

import extra_synthetic
import training_mixture


def test_checked_cells_load_with_corrected_forms():
    pairs = extra_synthetic.checked_pairs()
    assert len(pairs) == 247
    forms = {p["form"] for p in pairs}
    assert "ይትሰተይ" in forms and "ይትስተይ" not in forms
    assert "ይትተለው" in forms and "ይትትሉ" not in forms


def test_dictionary_root_check_drops_unpassed_roots():
    with open(training_mixture.DICTIONARY_ROOT_CHECK_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    bad = [r for r in rows if r["verdict"] != "OK"][:3]
    good = [r for r in rows if r["verdict"] == "OK"][:1]
    pairs = [{"record_id": r["record_id"], "from_forms": r["from_forms"] == "True", "form": r["form"]} for r in bad + good]
    assert [p["form"] for p in training_mixture.dictionary_root_check(pairs, __import__("collections").Counter())] == \
        [good[0]["form"]]
