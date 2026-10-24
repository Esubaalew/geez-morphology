"""train_byt5_morphology.model_to_load: the newest of the top-level final model and
checkpoint-<step> folders. A finished run's top-level model still wins over older
save_total_limit checkpoints; a resumed/extended run's newer checkpoint wins over a
stale top-level model. The analyzer finds its model the same way (analyzer.find_model_dir)
and stops when there is none."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import analyzer  # noqa: E402
import train_byt5_morphology as T  # noqa: E402


def test_tpu_run_selects_exported_best_model(tmp_path):
    best = tmp_path / "best"
    best.mkdir()
    (best / "config.json").write_text("{}")
    assert T.model_to_load(str(tmp_path)) == str(best)
    assert analyzer.find_model_dir(str(tmp_path)) == str(best)
    (tmp_path / "config.json").write_text("{}")
    assert T.model_to_load(str(tmp_path)) == str(tmp_path)


def test_final_model_then_latest_checkpoint():
    with tempfile.TemporaryDirectory() as out:
        for step in (900, 1000):
            os.makedirs(os.path.join(out, f"checkpoint-{step}"))
        os.makedirs(os.path.join(out, "checkpoint-x"))
        assert T.model_to_load(out) == os.path.join(out, "checkpoint-1000")
        open(os.path.join(out, "config.json"), "w").close()
        assert T.model_to_load(out) == out
        assert T.model_to_load(os.path.join(out, "checkpoint-900")) == os.path.join(out, "checkpoint-900")


def test_newer_checkpoint_wins_over_stale_final_model():
    with tempfile.TemporaryDirectory() as out:
        open(os.path.join(out, "config.json"), "w").close()
        with open(os.path.join(out, "trainer_state.json"), "w") as f:
            f.write('{"global_step": 1000}')
        os.makedirs(os.path.join(out, "checkpoint-2000"))
        open(os.path.join(out, "checkpoint-2000", "config.json"), "w").close()
        assert T.model_to_load(out) == os.path.join(out, "checkpoint-2000")


def test_finished_run_top_level_wins_over_older_checkpoint():
    with tempfile.TemporaryDirectory() as out:
        os.makedirs(os.path.join(out, "checkpoint-1000"))
        open(os.path.join(out, "checkpoint-1000", "config.json"), "w").close()
        open(os.path.join(out, "config.json"), "w").close()
        with open(os.path.join(out, "trainer_state.json"), "w") as f:
            f.write('{"global_step": 2000}')
        assert T.model_to_load(out) == out


def test_newer_checkpoint_mtime_wins_when_top_level_has_no_step():
    with tempfile.TemporaryDirectory() as out:
        open(os.path.join(out, "config.json"), "w").close()
        os.utime(os.path.join(out, "config.json"), (1_000_000, 1_000_000))
        ckpt = os.path.join(out, "checkpoint-500")
        os.makedirs(ckpt)
        cfg = os.path.join(ckpt, "config.json")
        open(cfg, "w").close()
        os.utime(cfg, (2_000_000, 2_000_000))
        assert T.model_to_load(out) == ckpt


def test_default_names():
    assert os.path.basename(T.DEFAULT_OUTPUT_DIR) == "byt5_morphology_model"
    assert T.parse_args([]).output_dir == T.DEFAULT_OUTPUT_DIR
    assert T.parse_args(["--init_from", "a2/best"]).init_from == "a2/best"
    assert set(T.EXPORT_FILES.values()) == {"train_pairs.jsonl", "eval_pairs_synthetic.jsonl",
                                            "eval_pairs_dictionary_nominal.jsonl"}


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
