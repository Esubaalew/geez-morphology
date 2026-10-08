"""The analyzer reads its dictionary from the dataset folder it is given.

- a missing dictionary is an error, not a silent run without a lexicon;
- --dataset_dir alone is enough: no $GEEZ_DATASET_DIR and no ../geez-dataset needed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyzer  # noqa: E402
import evaluate_gold  # noqa: E402
import dataset_loader  # noqa: E402
from conftest import DATASET_DIR  # noqa: E402

HAS_DICTIONARY = os.path.exists(os.path.join(DATASET_DIR, analyzer.DICTIONARY_FILE))


def test_missing_dictionary_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dictionary not found"):
        analyzer.GeezMorphologicalAnalyzer(dataset_dir=str(tmp_path))


@pytest.mark.skipif(not HAS_DICTIONARY, reason="no geez-dataset checkout")
def test_dataset_dir_alone_is_enough(monkeypatch, tmp_path):
    # No $GEEZ_DATASET_DIR, and the ../geez-dataset default points at a folder that does not exist.
    monkeypatch.delenv(dataset_loader.DATASET_DIR_ENV, raising=False)
    monkeypatch.setattr(dataset_loader, "SCRIPT_DIR", str(tmp_path / "geez-morphology"))
    assert not os.path.exists(dataset_loader.default_dataset_dir())

    system = evaluate_gold.HybridSystem(DATASET_DIR, no_neural=True)   # what --dataset_dir gives
    assert system.a.dataset_dir == os.path.abspath(DATASET_DIR)
    assert system.a.lexicon and system.a.root_index   # the dictionary was read
    assert system.predict("ቀተለ")["root"] == "q-t-l"
