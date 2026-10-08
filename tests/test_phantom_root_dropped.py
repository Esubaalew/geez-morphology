"""qh-h2-m is a phantom root and is off the allow-list.

Its only book source was a misread of Kidane Wold entry 360, which the scan shows with a plain first letter: the triple of 360
gives q-h2-m (data/phantom_root_qh_h2_m.tsv). So the generator makes no qh-h2-m lines, and the combined file
verbs_rbha_anqets.jsonl has no qh-h2-m lines. q-h2-m stays on the list and keeps its lines.
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


@pytest.mark.parametrize("rel", FILES + [os.path.join("data", "verbs_rbha_anqets.jsonl")])
def test_no_qh_h2_m_lines(rel):
    path = os.path.join(DATASET_DIR, rel)
    if not os.path.exists(path):
        pytest.skip("no geez-dataset checkout")
    assert "qh-h2-m" not in roots_in(path)


def test_q_h2_m_keeps_its_lines():
    path = os.path.join(DATASET_DIR, FILES[0])
    if not os.path.exists(path):
        pytest.skip("synthetic data not generated")
    assert roots_in(path).get("q-h2-m", 0) > 0
