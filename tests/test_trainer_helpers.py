"""Tests for train_byt5_morphology.py helpers (no model, no torch needed)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import train_byt5_morphology as T


def _dataset(tmp, n_syn=20, gold_name="merho_ges_gold_train.jsonl"):
    para = os.path.join(tmp, "data", "paradigms")
    os.makedirs(para)
    with open(os.path.join(para, "synthetic_morphology_train.jsonl"), "w", encoding="utf-8") as f:
        for i in range(n_syn):
            f.write(json.dumps({"surface_form": f"ይቀትል{i}", "features": {
                "root": "q-t-l", "tense": "present", "person": "3", "gender": "m",
                "number": "s", "stem": "basic"}}, ensure_ascii=False) + "\n")
    if gold_name:
        rec = {"record_id": "aaaaaaaaaaaa", "root": "ቀ-ተ-ለ", "stem": "basic", "table_data": [
            {"geez_form": "አስተቀተለ / አስተቀተለ፡", "tense": "past", "person": "3", "gender": "m", "number": "s",
             "stem": "aste"},
            {"geez_form": "ያስተቅትል", "tense": "subjunctive", "person": "3", "gender": "m", "number": "s",
             "stem": "aste"},
            {"geez_form": "ቀተለ", "tense": "past", "person": "3", "gender": "m", "number": "s"},
            {"geez_form": "", "tense": "past"}]}
        with open(os.path.join(para, gold_name), "w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return tmp


def test_load_counts_and_targets():
    with tempfile.TemporaryDirectory() as tmp:
        samples = T.load_data(_dataset(tmp))
        gold = [s for s in samples if s["is_gold"]]
        assert len(samples) == 24 and len(gold) == 4
        syn = [s for s in samples if not s["is_gold"]][0]
        assert syn["target_text"] == "root: ቀ-ተ-ለ | tense: present | person: 3 | gender: m | number: s | stem: basic"
        targets = sorted(s["target_text"] for s in gold)
        assert sum("stem: aste" in t for t in targets) == 3          # the row's own stem
        assert sum("stem: basic" in t for t in targets) == 1         # the record's stem
        assert sum("tense: subjunctive" in t for t in targets) == 1


def test_sample_limit_and_determinism():
    with tempfile.TemporaryDirectory() as tmp:
        _dataset(tmp, n_syn=50)
        a = T.load_data(tmp, limit=10)
        b = T.load_data(tmp, limit=10)
        assert sum(not s["is_gold"] for s in a) == 10
        assert [s["input_text"] for s in a] == [s["input_text"] for s in b]


def test_split_gold_all_in_train_eval_synthetic_only():
    with tempfile.TemporaryDirectory() as tmp:
        samples = T.load_data(_dataset(tmp, n_syn=100))
        train, evals, n_syn, n_gold = T.split_samples(samples)
        assert (n_syn, n_gold) == (100, 4)
        assert len(evals) == 10 and len(train) == 94
        assert not any(s["is_gold"] for s in evals)
        assert sum(s["is_gold"] for s in train) == 4


def test_missing_synthetic_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            T.load_data(tmp)
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


def _ckpt(folder, step, state=True):
    path = os.path.join(folder, f"checkpoint-{step}")
    os.makedirs(path)
    if state:
        open(os.path.join(path, "trainer_state.json"), "w").write("{}")
    return path


def test_latest_checkpoint_numeric_order_and_requires_state():
    with tempfile.TemporaryDirectory() as tmp:
        _ckpt(tmp, 900)
        best = _ckpt(tmp, 1000)
        _ckpt(tmp, 2000, state=False)        # half-written: ignored
        os.makedirs(os.path.join(tmp, "checkpoint-x"))
        assert T.latest_checkpoint(tmp) == best
        assert T.latest_checkpoint(os.path.join(tmp, "missing")) is None


def test_resolve_resume_prefers_output_dir():
    with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as prev:
        prev_ckpt = _ckpt(prev, 5000)
        assert T.resolve_resume(out, None) is None
        assert T.resolve_resume(out, prev) == prev_ckpt          # folder of checkpoints
        assert T.resolve_resume(out, prev_ckpt) == prev_ckpt     # a checkpoint itself
        own = _ckpt(out, 6000)
        assert T.resolve_resume(out, prev) == own                # the run's own progress wins


def test_resolve_resume_bad_path_raises():
    with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as empty:
        try:
            T.resolve_resume(out, empty)
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")


def test_optimizer_steps():
    assert T.optimizer_steps(1000, 16, 1, 1, 3, 0) == 63 * 3
    assert T.optimizer_steps(1000, 8, 2, 1, 3, 0) == 63 * 3      # same effective batch
    assert T.optimizer_steps(1000, 8, 1, 2, 3, 0) == 63 * 3      # two GPUs
    assert T.optimizer_steps(1000, 16, 1, 1, 3, 5) == 5


def test_parse_args_defaults():
    a = T.parse_args([])
    assert (a.lr, a.warmup_ratio, a.epochs, a.batch_size, a.grad_accum) == (3e-4, 0.03, 3, 16, 1)
    assert (a.save_steps, a.save_total_limit, a.sample_limit) == (1000, 2, 0)
