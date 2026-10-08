"""evaluate_gold.model_to_load: the final model at the top of the trainer's output folder wins over
its checkpoint-<step> folders (save_total_limit keeps older checkpoints than the final step)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import evaluate_gold as E  # noqa: E402
import train_byt5_morphology as T  # noqa: E402


def test_final_model_then_latest_checkpoint():
    with tempfile.TemporaryDirectory() as out:
        for step in (900, 1000):
            os.makedirs(os.path.join(out, f"checkpoint-{step}"))
        os.makedirs(os.path.join(out, "checkpoint-x"))
        assert E.model_to_load(out) == os.path.join(out, "checkpoint-1000")
        open(os.path.join(out, "config.json"), "w").close()
        assert E.model_to_load(out) == out
        assert E.model_to_load(os.path.join(out, "checkpoint-900")) == os.path.join(out, "checkpoint-900")


def test_default_names():
    assert os.path.basename(T.DEFAULT_OUTPUT_DIR) == "byt5_morphology_model"
    assert T.parse_args([]).output_dir == T.DEFAULT_OUTPUT_DIR
    assert set(T.EXPORT_FILES.values()) == {"train_pairs.jsonl", "eval_pairs_synthetic.jsonl"}
