"""data/heldout_test_roots_c7bc587.txt: the gold test roots the synthetic data was generated without.

The list was made once from dataset commit c7bc587 (data/paradigms/merho_ges_gold_test.jsonl) and is
committed, so the generator needs no git history of the dataset. Any change to the file shows here.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_synthetic_fixed as gen  # noqa: E402

SHA256 = "8a6b64053cddd0d54301acc4142e3aa9e2b25cc88bb6334ce5a05dd061b7c88d"


def test_file_is_unchanged():
    with open(gen.HELD_OUT_ROOTS_FILE, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == SHA256


def test_185_roots():
    roots = gen.read_held_out_roots_file()
    assert len(roots) == 185
    assert all(2 <= len(r) <= 5 for r in roots)
    assert ("q", "t", "l") not in roots and ("w", "d", "d") in roots   # ወ-ደ-ደ is a c7bc587 test root
