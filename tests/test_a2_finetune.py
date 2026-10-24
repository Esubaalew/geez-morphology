import json
import pytest
import a2_finetune as A


def checkpoint(tmp_path):
    p = tmp_path / "best"
    p.mkdir()
    for name in ("config.json", "model.safetensors"):
        (p / name).write_text("{}")
    manifest = tmp_path / "seen.json"
    manifest.write_text(json.dumps({"forms": [], "roots": []}))
    return p, manifest


def test_profile_keeps_global_batch_on_two_workers(tmp_path, monkeypatch):
    ck, seen = checkpoint(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "2")
    args = A.parse_args(["--dataset_dir", str(tmp_path), "--init_from", str(ck),
                         "--output_dir", str(tmp_path / "new"), "--dev_seen_manifest", str(seen)])
    cmd = A.trainer_args(args)
    assert cmd[cmd.index("--gradient_accumulation") + 1] == "16"
    assert cmd[cmd.index("--lr") + 1] == "0.0001"
    assert "--baseline_first" in cmd
    assert "--gradient_checkpointing" in cmd
    assert cmd[cmd.index("--bf16") + 1] == "no"


def test_profile_rejects_overwriting_a2(tmp_path):
    ck, seen = checkpoint(tmp_path)
    with pytest.raises(SystemExit):
        A.parse_args(["--dataset_dir", str(tmp_path), "--init_from", str(ck),
                      "--output_dir", str(ck), "--dev_seen_manifest", str(seen)])
