"""qh-h2-m is a phantom root and is off the allow-list.

Its only book source was a misread of Kidane Wold entry 360, which the printed page shows with a plain first letter: the triple of 360
gives q-h2-m (data/phantom_root_qh_h2_m.tsv). So the generator makes no qh-h2-m lines, and no root field of the
Kidane Wold tables (verb triples, rejected triples, root fixes) is qh-h2-m: entry 360 has q-h2-m in the triples and
the root fixes. q-h2-m stays on the list and keeps its lines.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from conftest import DATASET_DIR  # noqa: E402
import generate_synthetic_data as gen  # noqa: E402

ALLOWED = {"-".join(r) for r in gen.read_allowed_roots_file()}
FILES = [os.path.join("data", "paradigms", "synthetic_morphology_train.jsonl"),
         os.path.join("data", "paradigms", "synthetic_morphology_analyze.jsonl"),
         os.path.join("data", "synthetic", "synthetic_forms_for_val_test_roots.jsonl"),
         os.path.join("data", "synthetic", "synthetic_forms_for_val_test_roots_analyze.jsonl")]


def roots_in(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            x = x.get("record", x)
            r = (x.get("features") or {}).get("root") or x.get("root")
            out[r] = out.get(r, 0) + 1
    return out


def test_qh_h2_m_off_the_list():
    assert "qh-h2-m" not in ALLOWED
    assert "q-h2-m" in ALLOWED


KW_TABLES = [os.path.join("data", "lexicon", "kidane_wold_verb_triples.jsonl"),
             os.path.join("data", "lexicon", "kidane_wold_verb_triples_rejected.jsonl"),
             os.path.join("data", "lexicon", "kidane_wold_root_fixes.jsonl")]


def root_values(path):
    """Every root field (key "root" or "*_root") of a JSONL file, per entry id."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            out.setdefault(x.get("id"), set()).update(v for k, v in x.items() if k == "root" or k.endswith("_root"))
    return out


@pytest.mark.parametrize("rel", FILES)
def test_no_qh_h2_m_lines(rel):
    path = os.path.join(DATASET_DIR, rel)
    if not os.path.exists(path):
        pytest.skip("no geez-dataset checkout")
    assert "qh-h2-m" not in roots_in(path)


def test_q_h2_m_keeps_its_lines():
    paths = [os.path.join(DATASET_DIR, FILES[0]), os.path.join(DATASET_DIR, FILES[2])]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("synthetic data not generated")
    # q-h2-m is a book-only val root, so its lines are in the held-out file, not the training file
    assert sum(roots_in(p).get("q-h2-m", 0) for p in paths) > 0


@pytest.mark.parametrize("rel", KW_TABLES)
def test_no_qh_h2_m_in_kidane_wold_tables(rel):
    path = os.path.join(DATASET_DIR, rel)
    if not os.path.exists(path):
        pytest.skip("no geez-dataset checkout")
    assert not any("qh-h2-m" in v for v in root_values(path).values())


def test_entry_360_is_q_h2_m():
    """The book source of the phantom: entry 360's triple ቀሐመ ይቅሕም ይቅሐም gives q-h2-m."""
    triples = os.path.join(DATASET_DIR, KW_TABLES[0])
    fixes = os.path.join(DATASET_DIR, KW_TABLES[2])
    if not (os.path.exists(triples) and os.path.exists(fixes)):
        pytest.skip("no geez-dataset checkout")
    assert "q-h2-m" in root_values(triples)[360]
    assert "q-h2-m" in root_values(fixes)[360]
