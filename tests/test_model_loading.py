"""train_byt5_morphology.model_to_load: the final model at the top of the trainer's output folder wins over
its checkpoint-<step> folders (save_total_limit keeps older checkpoints than the final step).
The analyzer finds its model the same way (analyzer.find_model_dir) and stops when there is none."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import analyzer  # noqa: E402
import train_byt5_morphology as T  # noqa: E402


def test_final_model_then_latest_checkpoint():
    with tempfile.TemporaryDirectory() as out:
        for step in (900, 1000):
            os.makedirs(os.path.join(out, f"checkpoint-{step}"))
        os.makedirs(os.path.join(out, "checkpoint-x"))
        assert T.model_to_load(out) == os.path.join(out, "checkpoint-1000")
        open(os.path.join(out, "config.json"), "w").close()
        assert T.model_to_load(out) == out
        assert T.model_to_load(os.path.join(out, "checkpoint-900")) == os.path.join(out, "checkpoint-900")


def test_default_names():
    assert os.path.basename(T.DEFAULT_OUTPUT_DIR) == "byt5_morphology_model"
    assert T.parse_args([]).output_dir == T.DEFAULT_OUTPUT_DIR
    assert set(T.EXPORT_FILES.values()) == {"train_pairs.jsonl", "eval_pairs_synthetic.jsonl"}


def test_analyzer_takes_the_model_folder_from_the_same_rule():
    with tempfile.TemporaryDirectory() as out:
        os.makedirs(os.path.join(out, "checkpoint-500"))
        open(os.path.join(out, "checkpoint-500", "config.json"), "w").close()
        assert analyzer.find_model_dir(out) == os.path.join(out, "checkpoint-500")
        open(os.path.join(out, "config.json"), "w").close()
        assert analyzer.find_model_dir(out) == out


def test_missing_model_folder_raises():
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "byt5_morphology_model")
        with pytest.raises(FileNotFoundError, match="ByT5 model not found"):
            analyzer.find_model_dir(missing)
        with pytest.raises(FileNotFoundError, match="ByT5 model not found"):
            analyzer.HybridAnalyzer(model_dir=missing)     # neural on: the model must be there
        os.makedirs(missing)                               # a folder with no model in it
        with pytest.raises(FileNotFoundError, match="ByT5 model not found"):
            analyzer.HybridAnalyzer(model_dir=missing)
