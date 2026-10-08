"""Checks on a geez-dataset checkout (--dataset_dir, else $GEEZ_DATASET_DIR, else ../geez-dataset); skipped without one.

- the loader gives the same pair counts as the dataset's own scripts/check_dataset.py;
- no row of merho_ges_gold_flagged_rows.tsv loads;
- a root is in one gold split only;
- no synthetic training line has a gold val or test root (those lines are held out in data/synthetic/).
"""
import json
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import dataset_loader as G  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402
import split_heldout_synthetic as H  # noqa: E402

DATASET = DATASET_DIR   # --dataset_dir, else $GEEZ_DATASET_DIR, else ../geez-dataset (conftest.py)
CHECKER = os.path.join(DATASET, "scripts", "check_dataset.py")

pytestmark = pytest.mark.skipif(not os.path.exists(G.gold_file_path(DATASET, "train")),
                                reason="no geez-dataset checkout")


@pytest.fixture(scope="module")
def gold():
    return {split: G.load_gold_split(DATASET, split) for split in G.SPLITS}


def test_loader_counts_match_check_dataset(gold):
    if not os.path.exists(CHECKER):
        pytest.skip("dataset checkout without scripts/check_dataset.py")
    out = subprocess.run([sys.executable, CHECKER], cwd=DATASET, capture_output=True, text=True).stdout
    m = re.search(r"pairs that load: train (\d+), val (\d+), test (\d+)", out)
    assert m, out
    assert {s: len(p) for s, p in gold.items()} == dict(zip(G.SPLITS, map(int, m.groups())))


def test_flagged_rows_do_not_load(gold):
    flagged = G.read_flagged_rows(DATASET)
    assert flagged
    loaded = {(s, p["record_id"], p["row"], p["form"]) for s, pairs in gold.items() for p in pairs}
    assert not loaded & flagged


def test_a_root_is_in_one_gold_split(gold):
    roots = {split: {p["root"] for p in pairs} for split, pairs in gold.items()}
    assert not roots["train"] & roots["val"]
    assert not roots["train"] & roots["test"]
    assert not roots["val"] & roots["test"]


def test_synthetic_train_has_no_val_or_test_root(gold):
    eval_roots = set(H.gold_eval_roots(DATASET))
    assert {p["root"] for s in ("val", "test") for p in gold[s]} <= eval_roots
    with open(os.path.join(DATASET, "data", "paradigms", H.TRAIN), encoding="utf-8") as f:
        train_roots = {H.line_root(json.loads(line)) for line in f}
    assert train_roots and not train_roots & eval_roots
