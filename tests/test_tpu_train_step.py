"""The TPU trainer's world-size check, NaN handling, resume protocol, one-read finiteness check and dapt padding,
on the CPU, with several workers simulated by threads and a summing all-reduce."""
import contextlib
import os
import sys
import threading
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_tpu as TT  # noqa: E402

torch = pytest.importorskip("torch")


class TinyModel(torch.nn.Module):
    """Enough of a seq2seq model for train_step: logits of shape (rows, target len, vocab)."""

    def __init__(self, vocab=400):
        super().__init__()
        g = torch.Generator().manual_seed(0)          # per instance: workers start from identical weights
        self.emb = torch.nn.Embedding(vocab, 8)
        self.out = torch.nn.Linear(8, vocab)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
        self.poison = None          # a callable(input_ids) -> bool; True makes the loss NaN

    def forward(self, input_ids, attention_mask, labels):
        h = (self.emb(input_ids) * attention_mask.unsqueeze(-1)).mean(1, keepdim=True)
        logits = self.out(h).expand(-1, labels.shape[1], -1)
        if self.poison is not None and self.poison(input_ids):
            logits = logits * float("nan")
        return types.SimpleNamespace(logits=logits)


class FakeOpt:
    def __init__(self, model):
        self.inner = torch.optim.SGD(model.parameters(), lr=0.1)
        self.steps = 0
        self.param_groups = self.inner.param_groups

    def step(self):
        self.steps += 1
        self.inner.step()

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)


class Reducer:
    """A summing all-reduce across `world` threads; counts calls per worker."""

    def __init__(self, world):
        self.world, self.lock, self.barrier = world, threading.Lock(), threading.Barrier(world)
        self.buf, self.out, self.calls = [], None, collections_counter()

    def __call__(self, t):
        with self.lock:
            self.buf.append(t.detach().clone())
            self.calls[threading.current_thread().name] += 1
        if self.barrier.wait() == 0:
            self.out = torch.stack(self.buf).sum(0)
            self.buf = []
        self.barrier.wait()
        return self.out.clone()


def collections_counter():
    import collections
    return collections.Counter()


def micro(rows, seed, poison_value=None):
    g = torch.Generator().manual_seed(seed)
    inp = torch.randint(3, 259, (rows, 6), generator=g)
    if poison_value is not None:
        inp[0, 0] = poison_value
    lab = torch.randint(3, 259, (rows, 4), generator=g)
    lab[:, -1] = -100
    return inp, (inp != TT.PAD).long(), lab


def run_workers(world, per_worker_batches, max_norm=1.0, poison=None):
    """Run one optimizer step on `world` simulated workers; returns per-worker (loss, flags, opt steps, model)."""
    red = Reducer(world)
    results = {}

    def work(r):
        model = TinyModel()
        model.poison = poison
        opt = FakeOpt(model)
        loss, flags = TT.train_step(model, opt, per_worker_batches[r], torch=torch, device=torch.device("cpu"),
                                    world=world, all_sum=red, autocast=contextlib.nullcontext, max_norm=max_norm)
        results[r] = (loss, flags, opt.steps, model)

    threads = [threading.Thread(target=work, args=(r,), name=f"w{r}") for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, red


# ---------------------------------------------------------------- 1. world size

def test_world_size_defaults_to_detected_chips_and_fails_loudly():
    assert TT.parse_args(["--output_dir", "o"]).expected_world_size == -1
    assert TT.check_world_size(-1, 8, 8) == 8
    with pytest.raises(RuntimeError, match="expected 8 workers, got 1"):
        TT.check_world_size(-1, 1, 8)
    with pytest.raises(RuntimeError, match="cannot detect"):
        TT.check_world_size(-1, 4, None)
    assert TT.check_world_size(4, 4, 8) == 4          # explicit override
    with pytest.raises(RuntimeError):
        TT.check_world_size(4, 8, 8)
    assert TT.check_world_size(0, 3, 8) is None       # explicit opt-out
    assert TT.check_world_size(-1, 1, 1) == 1         # CPU / CUDA: one worker


def test_detect_chip_count_on_kaggle_v3_8():
    if "torch_xla" in sys.modules:
        pytest.skip("real XLA runtime present")
    assert TT.detect_chip_count({"TPU_ACCELERATOR_TYPE": "v3-8"}) == 8
    assert TT.detect_chip_count({"TPU_ACCELERATOR_TYPE": "v5litepod-4"}) == 4
    assert TT.detect_chip_count({}) is None


# ---------------------------------------------------------------- 2./4. NaN handling, one check per step

def test_clean_step_updates_once_with_one_host_read_per_worker():
    world, accum = 2, 3
    batches = [[micro(2, 10 * r + k) for k in range(accum)] for r in range(world)]
    res, red = run_workers(world, batches)
    for r in range(world):
        loss, flags, steps, _ = res[r]
        assert flags is None and steps == 1 and torch.isfinite(loss)
    # one all-reduce for the token count, one for the finiteness flags, regardless of accumulation
    assert set(red.calls.values()) == {2}


def test_nan_on_one_worker_stops_every_worker_before_the_update():
    world, accum = 2, 3
    batches = [[micro(2, 10 * r + k, poison_value=(7 if (r, k) == (1, 2) else None)) for k in range(accum)]
               for r in range(world)]
    poison = lambda ids: bool((ids[:, 0] == 7).any())
    res, _ = run_workers(world, batches, poison=poison)
    ref = TinyModel()
    for r in range(world):
        _, flags, steps, model = res[r]
        assert steps == 0 and flags == [0, 0, 1, 1]
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), ref.state_dict().values()))
    msg = TT.describe_nonfinite(flags, 12, 3, 45)
    assert "optimizer step 12" in msg and "micro-step(s) [2]" in msg and "batch index 45" in msg


def test_grad_norm_checked_without_clipping():
    model = TinyModel()
    opt = FakeOpt(model)
    batches = [micro(2, 1)]

    def nan_grad(_grad):
        return torch.full_like(_grad, float("inf"))
    model.out.weight.register_hook(nan_grad)
    loss, flags = TT.train_step(model, opt, batches, torch=torch, device=torch.device("cpu"), world=1,
                                all_sum=lambda t: t, autocast=contextlib.nullcontext, max_norm=0)
    assert torch.isfinite(loss) and flags == [0, 1] and opt.steps == 0


def test_grad_norm_zero_max_does_not_clip():
    model = TinyModel()
    for p in model.parameters():
        p.grad = torch.full_like(p, 100.0)
    before = [p.grad.clone() for p in model.parameters()]
    n = TT.grad_norm(model, 0, torch)
    assert torch.isfinite(n) and n > 1
    assert all(torch.equal(a, p.grad) for a, p in zip(before, model.parameters()))


def test_skip_nonfinite_batches_flag():
    assert TT.parse_args(["--output_dir", "o"]).skip_nonfinite_batches == 0
    assert TT.parse_args(["--output_dir", "o", "--skip_nonfinite_batches", "3"]).skip_nonfinite_batches == 3
    assert TT.skip_or_raise("m", 0, 2) == 1 and TT.skip_or_raise("m", 1, 2) == 2
    with pytest.raises(FloatingPointError, match="stopped before optimizer update"):
        TT.skip_or_raise("m", 2, 2)
    with pytest.raises(FloatingPointError):
        TT.skip_or_raise("m", 0, 0)


def test_warmup_help_names_optimizer_steps(capsys):
    with pytest.raises(SystemExit):
        TT.parse_args(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "Linear warmup length in optimizer steps" in out


# ---------------------------------------------------------------- 6. the loop's loss is token_loss

def test_step_loss_matches_global_token_mean_over_workers_and_microbatches():
    world, accum = 2, 2
    batches = [[micro(2, 100 + 10 * r + k) for k in range(accum)] for r in range(world)]
    batches[1][1][2][:] = -100          # one worker's microbatch has no targets
    res, _ = run_workers(world, batches)
    model = TinyModel()
    inp = torch.cat([m[0] for w in batches for m in w])
    att = torch.cat([m[1] for w in batches for m in w])
    lab = torch.cat([m[2] for w in batches for m in w])
    reference = TT.token_loss(model(inp, att, lab).logits, lab, torch)
    mean_loss = sum(res[r][0] for r in range(world)) / world
    assert torch.allclose(mean_loss, reference, atol=1e-5)


# ---------------------------------------------------------------- 3. resume protocol

def test_resume_refuses_a_changed_protocol():
    saved = {"gradient_accumulation": 4, "world": 8, "rows": 2048}
    TT.check_resume({"protocol": dict(saved)}, saved)
    for k, v in (("gradient_accumulation", 2), ("world", 1), ("rows", 1024)):
        with pytest.raises(RuntimeError, match=f"{k}: saved {saved[k]}, now {v}"):
            TT.check_resume({"protocol": dict(saved)}, dict(saved, **{k: v}))
    with pytest.raises(RuntimeError, match="no saved batch protocol"):
        TT.check_resume({"state": {}}, saved)


def test_saved_state_carries_the_protocol():
    src = open(TT.__file__, encoding="utf-8").read()
    assert '"protocol": protocol' in src and "check_resume(ck, protocol)" in src
    assert set(TT.RESUME_KEYS) == {"gradient_accumulation", "world", "rows"}


# ---------------------------------------------------------------- 5. dapt padding

def test_dapt_padding_rows_have_no_targets():
    items = [([10, 11, TT.EOS], [20, TT.EOS]), ([30, TT.EOS], [40, 41, TT.EOS])]
    inp, att, lab = TT.dapt_tensors(items, [0, 1, -1, -1], 8, 4, torch)
    assert (lab[2:] == -100).all() and (inp[2:] == TT.PAD).all() and (att[2:] == 0).all()
    assert lab[1, :3].tolist() == [40, 41, TT.EOS]
    # a worker that gets only padding contributes zero loss and zero gradient
    logits = torch.randn(2, 4, 400, requires_grad=True)
    loss = TT.token_loss(logits, lab[2:], torch)
    loss.backward()
    assert loss.item() == 0 and torch.equal(logits.grad, torch.zeros_like(logits))
