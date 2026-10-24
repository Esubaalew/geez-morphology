"""CUDA launch conventions and real CPU/DDP checks of the shared fine-tuning runtime."""
import contextlib
import json
import os
from pathlib import Path
import types

import pytest

import train_tpu as TT


def test_torchrun_layout_and_t4_precision():
    assert TT.distributed_layout({}) == (1, 0, 0)
    assert TT.distributed_layout(dict(WORLD_SIZE="2", RANK="1", LOCAL_RANK="1")) == (2, 1, 1)
    with pytest.raises(ValueError, match="torchrun"):
        TT.distributed_layout(dict(WORLD_SIZE="2", RANK="2"))
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_bf16_supported=lambda: False))
    assert not TT.bf16_enabled("auto", "cuda", fake)
    with pytest.raises(ValueError, match="does not support"):
        TT.bf16_enabled("yes", "cuda", fake)
    assert TT.bf16_enabled("auto", "xla", fake)


def test_initial_weights_are_eligible_for_best_selection():
    state = dict(best=-1, history=[])
    m = dict(dev_score=0.8, dev_root=0.9, dev_exact=0.7)
    baseline = TT.establish_baseline(state, m)
    assert baseline == dict(m, step=0, baseline=True)
    assert state["best"] == 0.8 and state["best_step"] == 0
    assert state["history"] == [baseline]
    assert not 0.79 > state["best"]


def _ddp_worker(rank, world, init_path, out_path):
    import socket
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    interfaces = {name for _, name in socket.if_nameindex()}
    loopback = next((name for name in ("lo0", "lo") if name in interfaces), None)
    if loopback:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", loopback)
    dist.init_process_group("gloo", init_method="file://" + init_path, rank=rank, world_size=world)
    torch.manual_seed(41)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(16, 8)
            self.out = torch.nn.Linear(8, 16)

        def forward(self, input_ids, attention_mask, labels):
            h = self.emb(input_ids).mean(1, keepdim=True)
            return types.SimpleNamespace(logits=self.out(h).expand(-1, labels.shape[1], -1))

    model = Tiny()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    net = DistributedDataParallel(model)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    inp = torch.tensor([[3, 4], [5, 6]])
    # Different target counts across ranks and microbatches, including an empty microbatch.
    labels = [torch.tensor([[7, 8], [9, -100]]), torch.tensor([[10, -100], [-100, -100]])]
    if rank == 1:
        labels = [torch.full((2, 2), -100), torch.tensor([[11, 12], [13, 14]])]
    batches = [(inp, torch.ones_like(inp), lab) for lab in labels]

    def all_sum(t):
        value = t.clone()
        dist.all_reduce(value)
        return value

    _, flags = TT.train_step(net, opt, batches, torch=torch, device=torch.device("cpu"), world=world,
                             all_sum=all_sum, autocast=contextlib.nullcontext, max_norm=0)
    assert flags is None
    if rank == 0:
        reference = Tiny()
        reference.load_state_dict(initial)
        ref_opt = torch.optim.SGD(reference.parameters(), lr=0.1)
        all_labels = [torch.tensor([[7, 8], [9, -100]]), torch.tensor([[10, -100], [-100, -100]]),
                      torch.full((2, 2), -100), torch.tensor([[11, 12], [13, 14]])]
        ref_batches = [(inp, torch.ones_like(inp), lab) for lab in all_labels]
        TT.train_step(reference, ref_opt, ref_batches, torch=torch, device=torch.device("cpu"), world=1,
                      all_sum=lambda t: t, autocast=contextlib.nullcontext, max_norm=0)
        for p, q in zip(model.parameters(), reference.parameters()):
            assert torch.allclose(p, q, atol=1e-6)
        Path(out_path).write_text("ok")
    dist.barrier()
    dist.destroy_process_group()


def test_ddp_update_matches_global_token_mean_with_padding(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU gloo backend unavailable")
    import torch.multiprocessing as mp
    mp.spawn(_ddp_worker, args=(2, str(tmp_path / "init"), str(tmp_path / "result")), nprocs=2, join=True)
    assert (tmp_path / "result").read_text() == "ok"


def test_baseline_first_exports_initial_model_and_fresh_optimizer(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import T5Config, T5ForConditionalGeneration
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(TT, "STOP", {"signal": None})
    source = tmp_path / "a2"
    model = T5ForConditionalGeneration(T5Config(vocab_size=384, d_model=16, d_ff=32, num_layers=1,
                                               num_decoder_layers=1, num_heads=2, dropout_rate=0,
                                               pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))
    model.save_pretrained(source)
    # The trainer should use the checkpoint's tokenizer rather than a fresh remote tokenizer.
    class Tokenizer:
        def save_pretrained(self, path):
            Path(path, "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: Tokenizer())
    sample = dict(input_text="analyze: x", target_text="root: none | type: particle", task="analyze",
                  dev="gold_val", cls="particle")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(TT, "prepare", lambda args, write=True: ([sample], [sample], {}))
    args = TT.parse_args(["--output_dir", str(out), "--dataset_dir", str(tmp_path), "--device", "cpu",
                          "--init_from", str(source), "--baseline_first", "--max_steps", "1", "--lr", "0",
                          "--per_core_batch", "1", "--gradient_accumulation", "1", "--skip_final_report"])
    TT.run(0, args)
    info = json.loads((out / "run_info.json").read_text())
    baseline = json.loads((out / "baseline.json").read_text())
    assert baseline["step"] == 0 and baseline["baseline"]
    assert info["state"]["step"] == 1 and info["state"]["best_step"] == 0
    assert info["state"]["best"] == baseline["dev_score"]
    assert (out / "best/model.safetensors").is_file()
    ck = torch.load(out / "resume/state.pt", map_location="cpu", weights_only=False)
    assert ck["protocol"]["world"] == 1
    assert not any(k.startswith("module.") for k in ck["model"])
