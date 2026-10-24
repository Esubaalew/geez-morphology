#!/usr/bin/env python3
"""Train ByT5 on Ge'ez morphology on a TPU VM (PyTorch/XLA, one process per TPU core, data parallel), on CUDA,
or on the CPU (for a dry run). CUDA supports torchrun with one DDP worker per GPU. Two stages:

  --stage finetune   the multitask mixture of training_mixture.py (analyze:, generate:, pos:)
  --stage dapt       span corruption on the Ge'ez corpora (build_dapt_corpus.py), then use --init_from with
                     its output folder for the fine-tune

Shapes are static: every batch has one of a few (input, target) lengths (INPUT_BUCKETS x TARGET_BUCKETS) and a fixed
number of rows, so XLA compiles each shape once. ByT5 ids are bytes + 3 (pad 0, eos 1), computed here directly.
Weights stay fp32; the forward pass runs under bf16 autocast (--bf16, default on XLA). Optimizer: Adafactor with a
fixed learning rate (--lr, default 1e-3), linear warmup, linear decay to 10% at the last step.

Model selection: every --eval_every steps the dev set (gold val + unified book val, training_mixture.dev_samples)
is scored with teacher forcing. A row whose every target byte is the argmax under teacher forcing is exactly the
greedy output, so the scores are the greedy exact-match and root accuracy, at fixed shapes:
  dev_exact   full analysis equal to the gold target
  dev_root    root field equal (the bytes before the first " |")
  dev_pos     the pos: task's "type: <class>" equal
and dev_score = (dev_root + dev_exact) / 2 picks best/. With --baseline_first, the starting weights participate
at step zero and remain best/ unless training improves that score. --dev_seen_manifest filters known historical
training exposure before dev sampling and scoring. At the end best/ is reloaded and dev_report.json gets the
greedy-decoded scores per dev slice and per class, with the class confusion of the pos: task.

Output (--output_dir, lean): best/ (bf16 safetensors + config, the model to use), resume/ (fp32 weights, optimizer,
position; the latest only), mixture_report.json, leak_audit.txt, eval_log.jsonl, dev_report.json, run_info.json.
Resume: rerun with the same --output_dir, or point --resume_from at a previous run's output folder. The run stops
cleanly (saving resume/) when --time_budget_hours is reached or on SIGTERM.
"""
import argparse
import collections
import contextlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dataset_loader  # noqa: E402
import training_mixture as tm  # noqa: E402

INPUT_BUCKETS = (32, 64, 128)
TARGET_BUCKETS = (16, 32, 128)
DAPT_INPUT, DAPT_TARGET = 256, 64
PAD, EOS = 0, 1
SENTINEL_FIRST, SENTINEL_LAST = 259, 383   # ByT5 <extra_id_0> .. <extra_id_124>
START = time.time()
STOP = {"signal": None}


# ----------------------------------------------------------------------------- bytes

def encode(text, max_len):
    ids = [b + 3 for b in text.encode("utf-8")]
    if len(ids) + 1 > max_len:
        raise ValueError(f"Text needs {len(ids) + 1} byte tokens, but its bucket holds {max_len}")
    return ids + [EOS]


def decode(ids):
    out = bytearray()
    for i in ids:
        if i == EOS:
            break
        if 3 <= i < 259:
            out.append(i - 3)
    return out.decode("utf-8", errors="replace")


def bucket(n, sizes):
    for s in sizes:
        if n <= s:
            return s
    # Extend the static shape vocabulary for long analyses instead of losing labels.
    return 1 << (n - 1).bit_length()


def make_batches(samples, rows, seed, epoch, drop_last=True):
    """Lists of (sample indices, input len, target len), each `rows` long, in a seeded order per epoch."""
    if rows <= 0:
        raise ValueError("Batch rows must be positive")
    groups = collections.defaultdict(list)
    for i, s in enumerate(samples):
        a = bucket(len(s["input_text"].encode()) + 1, INPUT_BUCKETS)
        b = bucket(len(s["target_text"].encode()) + 1, TARGET_BUCKETS)
        groups[(a, b)].append(i)
    rng = random.Random(seed * 1000 + epoch)
    batches = []
    for (a, b), idx in sorted(groups.items()):
        rng.shuffle(idx)
        for k in range(0, len(idx), rows):
            chunk = idx[k:k + rows]
            if len(chunk) < rows:
                if drop_last and len(idx) >= rows:
                    continue
                chunk = chunk + [-1] * (rows - len(chunk))     # -1: a padding row (weight 0)
            batches.append((chunk, a, b))
    rng.shuffle(batches)
    return batches


def tensors(samples, chunk, a, b, torch):
    inp = torch.full((len(chunk), a), PAD, dtype=torch.long)
    lab = torch.full((len(chunk), b), -100, dtype=torch.long)
    for r, i in enumerate(chunk):
        if i < 0:
            continue
        x, y = encode(samples[i]["input_text"], a), encode(samples[i]["target_text"], b)
        inp[r, :len(x)] = torch.tensor(x)
        lab[r, :len(y)] = torch.tensor(y)
    return inp, (inp != PAD).long(), lab


# ----------------------------------------------------------------------------- span corruption

def corrupt(ids, rng, noise=0.15, mean_span=20):
    n = len(ids)
    if not n:
        return [], []
    n_noise = max(1, round(n * noise))
    n_spans = max(1, round(n_noise / mean_span))
    starts = sorted(rng.sample(range(n), min(n_spans, n)))
    mask = [False] * n
    remaining = n_noise
    for s in starts:
        for j in range(s, min(n, s + max(1, int(rng.expovariate(1 / mean_span))))):
            if not remaining:
                break
            if not mask[j]:
                mask[j] = True
                remaining -= 1
    inp, tgt, sent, prev = [], [], SENTINEL_FIRST, False
    for i, m in enumerate(mask):
        if m and not prev:
            inp.append(sent); tgt.append(sent); sent = min(sent + 1, SENTINEL_LAST)
        (tgt if m else inp).append(ids[i])
        prev = m
    return inp, tgt


def dapt_samples(path, epoch, seed):
    """Fixed-length span-corruption pairs of the corpus, re-noised each epoch; ids, not text."""
    with open(path, encoding="utf-8") as f:
        data = [b + 3 for b in f.read().replace("\n", " ").encode("utf-8")]
    rng = random.Random(seed * 7919 + epoch)
    chunk = 230
    out = []
    for k in range(0, len(data), chunk):
        inp, tgt = corrupt(data[k:k + chunk], rng)
        if len(inp) >= DAPT_INPUT or len(tgt) >= DAPT_TARGET:
            raise ValueError("Span corruption exceeds the DAPT tensor shape")
        out.append((inp + [EOS], tgt + [EOS]))
    return out


def dapt_batches(n_items, rows, seed, epoch):
    """Keep a short corpus/remainder with zero-weight padding, including on many workers."""
    if rows <= 0:
        raise ValueError("Batch rows must be positive")
    order = list(range(n_items))
    random.Random(seed + epoch).shuffle(order)
    return [(order[k:k + rows] + [-1] * max(0, rows - len(order[k:k + rows])), DAPT_INPUT, DAPT_TARGET)
            for k in range(0, len(order), rows)]


def training_batches(train, dapt_count, rows, seed, epoch):
    batches = (dapt_batches(dapt_count, rows, seed, epoch) if dapt_count is not None
               else make_batches(train, rows, seed, epoch))
    if not batches:
        raise ValueError("No training samples: provide a non-empty corpus or training mixture")
    return batches


# ----------------------------------------------------------------------------- args

def add_data_args(p):
    p.add_argument("--sample_limit", type=int, default=0, help="Use N synthetic pairs (0 = all)")
    p.add_argument("--gold_upsample", type=int, default=4)
    p.add_argument("--gold_nominal_upsample", type=int, default=5)
    p.add_argument("--unified_upsample", type=int, default=3)
    p.add_argument("--rare_class_min", type=int, default=400, help="Rare classes are repeated up to this many rows")
    p.add_argument("--rare_class_max_factor", type=int, default=20)
    p.add_argument("--no_unified", action="store_true")
    p.add_argument("--dictionary_radicals", default="3,4", help="Root lengths of the dictionary pairs")
    p.add_argument("--biradical_dictionary", action="store_true",
                   help="Also train the 2-radical dictionary roots (off until the 2-radical table (scripts/export_biradical_roots.py) is checked)")
    p.add_argument("--extra_synthetic", default="checked", help="Comma list of extra_synthetic.FAMILIES, or checked for only the checked cells (default checked; pass "" for none)")
    p.add_argument("--no_generate_task", dest="generate_task", action="store_false")
    p.add_argument("--no_pos_task", dest="pos_task", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unified_file", default=None,
                   help="Unified rows file inside the dataset (default data/unified/geez_morphology_rows.jsonl), "
                        "e.g. data/unified/geez_morphology_rows_reviewed.jsonl")
    p.add_argument("--clitic_forms", action="store_true", help="Add data/clitics/clitic_forms.jsonl (train split)")
    p.add_argument("--clitic_upsample", type=int, default=1)
    p.add_argument("--extra_rootless", default=None,
                   help="Review table of names and primary nouns without a root (word, word_class, review_note)")
    p.add_argument("--rootless_upsample", type=int, default=1)
    p.add_argument("--synthetic_fraction", type=float, default=1.0, help="Share of the synthetic pairs to keep")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_dir", default=dataset_loader.default_dataset_dir(),
                   help="Checkout of the geez-dataset repo "
                        "(default: $GEEZ_DATASET_DIR, else ../geez-dataset or ../geez-dataset-v1)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--stage", choices=["finetune", "dapt"], default="finetune")
    p.add_argument("--dapt_corpus", default=None, help="Text from build_dapt_corpus.py (built on the fly if missing)")
    p.add_argument("--model_name", default="google/byt5-small")
    p.add_argument("--init_from", default=None, help="Start from this model folder (e.g. the dapt stage's best/)")
    p.add_argument("--resume_from", default=None, help="A previous run's output folder (its resume/ is used)")
    p.add_argument("--device", choices=["auto", "xla", "cuda", "cpu"], default="auto")
    p.add_argument("--xla_launch", choices=["multi", "single"], default="multi",
                   help="multi: one process per TPU chip; single: one process on one chip (fallback)")
    p.add_argument("--expected_world_size", type=int, default=-1,
                   help="Fail unless the runtime uses exactly this many workers. Default -1: the detected TPU chip "
                       "count (8 on a TPU v3-8), or torchrun WORLD_SIZE off TPU; 0 disables the check")
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Clip the gradient norm to this (0 = no clipping; the norm is still checked for NaN/inf)")
    p.add_argument("--skip_nonfinite_batches", type=int, default=0,
                   help="Skip up to N optimizer steps whose loss or gradient is NaN/inf (no update), then stop; "
                        "0 = stop at the first one")
    p.add_argument("--skip_final_report", action="store_true", help="Skip slow autoregressive report in a runtime pilot")
    p.add_argument("--baseline_first", action="store_true",
                   help="Evaluate and export the starting model before training; replace best/ only on improvement")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Trade extra computation for lower GPU memory use")
    p.add_argument("--bf16", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--per_core_batch", type=int, default=64)
    p.add_argument("--gradient_accumulation", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Linear warmup length in optimizer steps (one step = --gradient_accumulation microbatches)")
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--dev_cap", type=int, default=0, help="Seeded sample of N dev pairs (0 = all)")
    p.add_argument("--dev_seen_manifest", "--seen_manifest", default=None,
                   help="Historical training exposure manifest; remove known-seen forms/roots from dev")
    p.add_argument("--report_cap", type=int, default=0, help="Greedy-decode at most N dev rows at the end (0 = all)")
    p.add_argument("--time_budget_hours", type=float, default=8.0)
    p.add_argument("--train_limit", type=int, default=0, help="Dry runs: a seeded sample of N training samples")
    p.add_argument("--load_only", action="store_true", help="Build the mixture, write reports and the audit, exit")
    p.add_argument("--export_dir", default=None, help="Also write train_pairs.jsonl / dev_pairs.jsonl here")
    add_data_args(p)
    args = p.parse_args(argv)
    for name in ("per_core_batch", "gradient_accumulation", "epochs", "eval_every", "save_every", "log_every"):
        if getattr(args, name) <= 0:
            p.error(f"--{name} must be positive")
    if args.max_steps < 0:
        p.error("--max_steps must be nonnegative")
    return args


# ----------------------------------------------------------------------------- data + audit (rank 0 writes)

def prepare(args, write=True):
    if args.stage == "dapt":
        corpus = args.dapt_corpus or os.path.join(args.output_dir, "dapt_corpus.txt")
        if not os.path.exists(corpus):
            import build_dapt_corpus
            os.makedirs(args.output_dir, exist_ok=True)
            rep = build_dapt_corpus.build(args.dataset_dir, corpus)
            with open(os.path.join(args.output_dir, "dapt_report.json"), "w", encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=1)
        args.dapt_corpus = corpus
        return [], [], {}
    if getattr(args, "unified_file", None):      # one unified file for training, the held-out gate and dev
        tm.dataset_loader.UNIFIED_FILE = args.unified_file
    train, _, report = tm.build(args.dataset_dir, args)
    if args.train_limit and args.train_limit < len(train):      # dry runs: a seeded sample of the mixture
        train = random.Random(args.seed).sample(train, args.train_limit)
    dev_filter_stats = {}
    dev = tm.dev_samples(args.dataset_dir, args.dev_cap, pos_task=args.pos_task,
                         seen_manifest=args.dev_seen_manifest, stats=dev_filter_stats)
    report["dev_exposure_filter"] = dev_filter_stats
    if write:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "mixture_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        export = args.export_dir or os.path.join(args.output_dir, "_pairs")
        os.makedirs(export, exist_ok=True)
        tm.write_jsonl(os.path.join(export, "train_pairs.jsonl"), train)
        tm.write_jsonl(os.path.join(export, "dev_pairs.jsonl"), dev)
        audit_cmd = [sys.executable, os.path.join(HERE, "scripts", "audit_train_leakage.py"),
                                "--dataset_dir", args.dataset_dir, "--train", os.path.join(export, "train_pairs.jsonl")]
        if args.unified_file:
            audit_cmd += ["--unified_file", args.unified_file]
        audit = subprocess.run(audit_cmd, capture_output=True, text=True)
        with open(os.path.join(args.output_dir, "leak_audit.txt"), "w", encoding="utf-8") as f:
            f.write(audit.stdout + audit.stderr)
        print(audit.stdout.strip().splitlines()[-1] if audit.stdout.strip() else audit.stderr, flush=True)
        if audit.returncode != 0:
            raise SystemExit("leak audit failed; see leak_audit.txt")
        if not args.export_dir:
            for n in ("train_pairs.jsonl", "dev_pairs.jsonl"):
                os.remove(os.path.join(export, n))
            os.rmdir(export)
    print(f"train samples {len(train)}, dev rows {len(dev)}: " +
          ", ".join(f"{k} {v}" for k, v in report["pairs_by_source"].items()), flush=True)
    return train, dev, report


# ----------------------------------------------------------------------------- scoring

def root_end(labels_row):
    """Index of the first '|' byte of a target (the end of the root field), else the target length."""
    bar = ord("|") + 3
    for j, v in enumerate(labels_row):
        if v == bar:
            return j
    return len(labels_row)


def score_batch(logits, labels, torch):
    """Per-row (exact, root) under teacher forcing, as 0/1 tensors."""
    pred = logits.argmax(-1)
    valid = labels != -100
    ok = (pred == labels) | ~valid
    exact = ok.all(-1)
    bar = ord("|") + 3
    pos = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    is_bar = (labels == bar)
    first_bar = torch.where(is_bar.any(-1), is_bar.float().argmax(-1), torch.full_like(labels[:, 0], labels.shape[1]))
    in_root = pos < first_bar.unsqueeze(1)
    root = (ok | ~in_root).all(-1)
    return exact, root


def token_loss(logits, labels, torch, all_sum=lambda x: x, world=1, count=None):
    """Global token-mean CE, including workers with no non-padding targets. `count` is the global target-token
    count when it is already known (e.g. summed over all microbatches of an optimizer step)."""
    loss_sum = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1),
        ignore_index=-100, reduction="sum")
    if count is None:
        count = all_sum((labels != -100).sum().float())
    return loss_sum * world / count.clamp_min(1)


def dapt_tensors(items, chunk, a, b, torch):
    """Span-corruption rows of `items` (id lists); index -1 is a padding row (all PAD input, all -100 labels)."""
    inp = torch.full((len(chunk), a), PAD, dtype=torch.long)
    lab = torch.full((len(chunk), b), -100, dtype=torch.long)
    for r, i in enumerate(chunk):
        if i < 0:
            continue
        x, y = items[i]
        inp[r, :len(x)] = torch.tensor(x)
        lab[r, :len(y)] = torch.tensor(y)
    return inp, (inp != PAD).long(), lab


def detect_chip_count(env=None):
    """Number of local TPU chips, or None if unknown. A Kaggle TPU v3-8 has 8."""
    env = os.environ if env is None else env
    try:
        import torch_xla.runtime as xr
        for name in ("global_runtime_device_count", "global_device_count"):
            if hasattr(xr, name):
                n = int(getattr(xr, name)())
                if n > 0:
                    return n
    except Exception:
        pass
    acc = env.get("TPU_ACCELERATOR_TYPE", "")
    if acc.startswith("v3-8") or acc.startswith("v2-8") or (env.get("KAGGLE_KERNEL_RUN_TYPE") and env.get("TPU_NAME")):
        return 8
    m = re.match(r"v\d+\w*-(\d+)$", acc)
    if m:
        return int(m.group(1))
    return None


def check_world_size(expected, world, detected):
    """Return the required world size (None = not checked); raise if the runtime differs."""
    if expected == 0:
        return None
    if expected < 0:
        if detected is None:
            raise RuntimeError(f"cannot detect the TPU chip count (runtime has {world} workers); "
                               "pass --expected_world_size N, or 0 to skip the check")
        expected = detected
    if world != expected:
        raise RuntimeError(f"expected {expected} workers, got {world}; refusing a changed protocol "
                           "(set --expected_world_size to override)")
    return expected


RESUME_KEYS = ("gradient_accumulation", "world", "rows")


def check_resume(ck, protocol):
    """Refuse to resume when the saved batch protocol differs from this run's."""
    saved = ck.get("protocol")
    if saved is None:
        raise RuntimeError("resume/state.pt has no saved batch protocol (gradient_accumulation, world, rows); "
                           "it cannot be checked, so it is not resumed. Start a new --output_dir")
    diff = [f"{k}: saved {saved.get(k)}, now {protocol[k]}" for k in RESUME_KEYS if saved.get(k) != protocol[k]]
    if diff:
        raise RuntimeError("refusing to resume with a different batch protocol (" + "; ".join(diff) + "). "
                           "Rerun with the saved settings or start a new --output_dir")


def nonfinite_flags(micro_losses, grad_norm, torch):
    """One device vector per optimizer step: 1 per non-finite microbatch loss, then 1 if the grad norm is."""
    return torch.stack([(~torch.isfinite(l.detach())).to(torch.int32) for l in micro_losses] +
                       [(~torch.isfinite(grad_norm.detach())).to(torch.int32)])


def grad_norm(model, max_norm, torch):
    """Total grad norm (clipping to max_norm when > 0). Never raises on NaN/inf; the caller checks the result."""
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return torch.zeros(())
    return torch.nn.utils.clip_grad_norm_(params, max_norm if max_norm > 0 else float("inf"),
                                          error_if_nonfinite=False)


def train_step(model, opt, micro_batches, *, torch, device, world, all_sum, autocast, max_norm,
               sync=None, reduce_gradients=None):
    """Forward/backward all microbatches, then step once unless any worker saw a NaN/inf loss or grad norm.
    The finiteness check is one host read per optimizer step. Returns (summed loss, flags list or None);
    flags is None when the step was taken, else the all-reduced per-microbatch flags (last = grad norm)
    and the gradients are dropped without stepping."""
    local_count = sum((lab != -100).sum() for _, _, lab in micro_batches)
    global_count = all_sum(local_count.to(device).float())
    loss = torch.zeros((), device=device)
    micro_losses = []
    for inp, att, lab in micro_batches:
        inp, att, lab = inp.to(device), att.to(device), lab.to(device)
        # DDP averages gradients; token_loss multiplies by world to recover the global token mean.
        # Synchronize only the last microbatch of an accumulated update.
        last_micro = len(micro_losses) == len(micro_batches) - 1
        ddp_sync = model.no_sync() if hasattr(model, "no_sync") and not last_micro else contextlib.nullcontext()
        with ddp_sync:
            with autocast():
                logits = model(input_ids=inp, attention_mask=att, labels=lab).logits
                micro_loss = token_loss(logits, lab, torch, world=world, count=global_count)
            micro_loss.backward()
        micro_losses.append(micro_loss.detach())
        loss = loss + micro_loss.detach()
        if sync:
            sync()
    if reduce_gradients:
        reduce_gradients(opt)
    norm = grad_norm(model, max_norm, torch).to(device)
    flags = all_sum(nonfinite_flags(micro_losses, norm, torch)).cpu().tolist()
    if any(flags):
        opt.zero_grad(set_to_none=True)
        return loss, flags
    opt.step()
    if sync:
        sync()
    opt.zero_grad(set_to_none=True)
    return loss, None


def skip_or_raise(msg, skipped, limit):
    """Count one more skipped batch, or raise once `limit` skips are used up (limit 0: never skip)."""
    if skipped >= limit:
        raise FloatingPointError(msg + "; stopped before optimizer update")
    return skipped + 1


def describe_nonfinite(flags, step, epoch, batch):
    bad_micro = [i for i, f in enumerate(flags[:-1]) if f]
    what = []
    if bad_micro:
        what.append(f"loss at micro-step(s) {bad_micro} on {max(flags[:-1])} worker(s)")
    if flags[-1]:
        what.append(f"grad norm on {flags[-1]} worker(s)")
    return (f"non-finite {' and '.join(what)} at optimizer step {step} (epoch {epoch}, batch index {batch}, "
            f"micro-steps {len(flags) - 1})")


def greedy(model, inp, att, max_len, torch, sync=None):
    """Greedy decoding at a fixed decoder length. The step index is a device tensor, so on XLA every step runs the
    same compiled graph."""
    enc = model.get_encoder()(input_ids=inp, attention_mask=att)
    dec = torch.full((inp.shape[0], max_len), PAD, dtype=torch.long, device=inp.device)
    pos = torch.arange(max_len, device=inp.device).unsqueeze(0)
    for t in range(max_len - 1):
        tt = torch.tensor(t, device=inp.device)
        logits = model(encoder_outputs=enc, attention_mask=att, decoder_input_ids=dec).logits
        nxt = torch.index_select(logits, 1, tt.view(1)).squeeze(1).argmax(-1)
        dec = torch.where(pos == tt + 1, nxt.unsqueeze(1), dec)
        if sync:
            sync()
    return dec[:, 1:].cpu().tolist()


# ----------------------------------------------------------------------------- the run

def distributed_layout(env=None):
    """torchrun supplies all three values; ordinary CPU/CUDA runs use one worker."""
    env = os.environ if env is None else env
    world, rank, local = (int(env.get(k, d)) for k, d in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if world < 1 or not 0 <= rank < world or local < 0:
        raise ValueError("Invalid torchrun WORLD_SIZE/RANK/LOCAL_RANK")
    return world, rank, local


def bf16_enabled(requested, device, torch):
    """T4 GPUs do not support bf16: auto must keep them in fp32."""
    supported = device == "xla" or (device == "cuda" and torch.cuda.is_bf16_supported())
    if requested == "yes" and device == "cuda" and not supported:
        raise ValueError("This CUDA device does not support bf16; use --bf16 no")
    return requested == "yes" or (requested == "auto" and supported)


def establish_baseline(state, metrics):
    """Step zero participates in model selection, so a regression never replaces the initial model."""
    baseline = dict(metrics, step=0, baseline=True)
    state["baseline"] = baseline
    state["best"] = baseline["dev_score"]
    state["best_step"] = 0
    state["history"].append(baseline)
    return baseline


def run(rank, args):
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from transformers.optimization import Adafactor

    xla = args.device == "xla"
    dist = None
    if xla:
        import torch_xla
        import torch_xla.core.xla_model as xm
        try:
            import torch_xla.runtime as xr
            world, rank = xr.world_size(), xr.global_ordinal()
        except Exception:
            world, rank = xm.xrt_world_size(), xm.get_ordinal()
        device = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
        sync = xm.mark_step
    else:
        world, rank, local = distributed_layout()
        if args.device == "cuda":
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
        else:
            device = torch.device("cpu")
        if world > 1:
            import torch.distributed as dist
            if not dist.is_initialized():
                import datetime
                dist.init_process_group("nccl" if args.device == "cuda" else "gloo",
                                        timeout=datetime.timedelta(minutes=30))
        sync = None
    master = rank == 0
    check_world_size(args.expected_world_size, world, detect_chip_count() if xla else distributed_layout()[0])
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_bf16 = bf16_enabled(args.bf16, args.device, torch)

    def log(*a):
        if master:
            print(*a, flush=True)

    def all_sum(t):
        if xla and world > 1:
            return xm.all_reduce(xm.REDUCE_SUM, t)
        if dist is not None:
            t = t.clone()
            dist.all_reduce(t)
        return t

    def barrier(tag):
        if xla:
            xm.rendezvous(tag)
        elif dist is not None:
            dist.barrier()

    train, dev, report = prepare(args, write=master) if master else prepare(args, write=False)
    barrier("data")

    src = args.init_from or args.model_name
    model = AutoModelForSeq2SeqLM.from_pretrained(src, dtype=torch.float32)
    resume_dir = None
    for d in (os.path.join(args.output_dir, "resume"), args.resume_from and os.path.join(args.resume_from, "resume")):
        if d and os.path.exists(os.path.join(d, "state.pt")):
            resume_dir = d
            break
    state = {"step": 0, "epoch": 0, "batch": 0, "best": -1.0, "history": []}
    physical_rows = args.per_core_batch * world
    rows = physical_rows * args.gradient_accumulation
    protocol = {"gradient_accumulation": args.gradient_accumulation, "world": world, "rows": rows}
    saved_rng = None
    if resume_dir:
        ck = torch.load(os.path.join(resume_dir, "state.pt"), map_location="cpu", weights_only=False)
        check_resume(ck, protocol)
        model.load_state_dict(ck["model"])
        state = ck["state"]
        saved_rng = ck.get("rng")
        log(f"resumed from {resume_dir} at step {state['step']}")
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    opt = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    if resume_dir:
        opt.load_state_dict(ck["opt"])
        del ck
        previous_best = os.path.join(os.path.dirname(resume_dir), "best")
        local_best = os.path.join(args.output_dir, "best")
        if master and os.path.abspath(previous_best) != os.path.abspath(local_best) and not os.path.exists(local_best):
            if state["best"] >= 0 and not os.path.isdir(previous_best):
                raise FileNotFoundError(f"Resume state selects a best model, but {previous_best} is missing")
            if os.path.isdir(previous_best):
                import shutil
                shutil.copytree(previous_best, local_best)
        barrier("resume_best")
    net = model
    if dist is not None:
        from torch.nn.parallel import DistributedDataParallel
        net = DistributedDataParallel(model, device_ids=[local] if args.device == "cuda" else None,
                                      broadcast_buffers=False)
    # Restore each worker's dropout stream after model initialization and DDP setup.
    torch.manual_seed(args.seed + rank)
    if saved_rng:
        rng = saved_rng[rank]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if args.device == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state(rng["cuda"], device)
        if xla and rng.get("xla") is not None:
            xm.set_rng_state(rng["xla"])

    dapt_count = len(dapt_samples(args.dapt_corpus, 0, args.seed)) if args.stage == "dapt" else None
    steps_per_epoch = len(training_batches(train, dapt_count, rows, args.seed, 0))
    total = args.max_steps or int(math.ceil(steps_per_epoch * args.epochs))
    log(f"device {args.device} x{world}, bf16 {use_bf16}, global batch {rows}, steps/epoch {steps_per_epoch}, "
        f"total steps {total}")

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        frac = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
        return args.lr * max(0.1, 1 - 0.9 * frac)

    def autocast():
        if use_bf16:
            return torch.autocast("xla" if xla else device.type, dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    # ---- dev scoring (teacher forced), sharded over the cores
    dev_batches = make_batches(dev, physical_rows, 0, 0, drop_last=False) if dev else []

    def evaluate():
        model.eval()
        keys = ["n", "exact", "root", "pos_n", "pos"]
        tot = torch.zeros(len(keys), device=device)
        with torch.no_grad():
            for chunk, a, b in dev_batches:
                mine = chunk[rank * args.per_core_batch:(rank + 1) * args.per_core_batch]
                inp, att, lab = tensors(dev, mine, a, b, torch)
                is_pos = torch.tensor([i >= 0 and dev[i]["task"] == tm.POS_TASK for i in mine])
                is_an = torch.tensor([i >= 0 and dev[i]["task"] != tm.POS_TASK for i in mine])
                inp, att, lab = inp.to(device), att.to(device), lab.to(device)
                with autocast():
                    logits = model(input_ids=inp, attention_mask=att, labels=lab).logits
                exact, root = score_batch(logits, lab, torch)
                is_pos, is_an = is_pos.to(device), is_an.to(device)
                tot += torch.stack([is_an.sum(), (exact & is_an).sum(), (root & is_an).sum(),
                                    is_pos.sum(), (exact & is_pos).sum()]).float()
                if sync:
                    sync()
        tot = all_sum(tot).cpu().tolist()
        model.train()
        m = dict(zip(keys, tot))
        res = {"dev_exact": m["exact"] / max(1, m["n"]), "dev_root": m["root"] / max(1, m["n"]),
               "dev_pos": m["pos"] / max(1, m["pos_n"])}
        res["dev_score"] = (res["dev_exact"] + res["dev_root"]) / 2
        return res

    def save_resume():
        local_rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state(device) if args.device == "cuda" else None,
                     "xla": xm.get_rng_state() if xla else None}
        if dist is not None:
            rng = [None] * world
            dist.all_gather_object(rng, local_rng)
        elif xla and world > 1:
            # CPU RNG states are rank-specific; XLA's tensor RNG is captured as an integer.
            import pickle
            payloads = xm.rendezvous("resume_rng", pickle.dumps(local_rng))
            rng = [pickle.loads(payload) for payload in payloads]
        else:
            rng = [local_rng]
        cpu = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
        if xla:
            optsd = xm._maybe_convert_to_cpu(opt.state_dict()) if hasattr(xm, "_maybe_convert_to_cpu") else opt.state_dict()
        else:
            optsd = opt.state_dict()
        if master:
            d = os.path.join(args.output_dir, "resume")
            os.makedirs(d, exist_ok=True)
            torch.save({"model": cpu, "opt": optsd, "state": state, "protocol": protocol, "rng": rng},
                       os.path.join(d, "state.pt.tmp"))
            os.replace(os.path.join(d, "state.pt.tmp"), os.path.join(d, "state.pt"))
        barrier("resume")

    def save_best():
        cpu = {k: v.detach().to("cpu").to(torch.bfloat16).clone() for k, v in model.state_dict().items()}
        if master:
            d = os.path.join(args.output_dir, "best")
            from safetensors.torch import save_file
            os.makedirs(d, exist_ok=True)
            save_file({k: v.contiguous() for k, v in cpu.items()}, os.path.join(d, "model.safetensors.tmp"),
                      metadata={"format": "pt"})
            os.replace(os.path.join(d, "model.safetensors.tmp"), os.path.join(d, "model.safetensors"))
            model.config.save_pretrained(d)
            try:
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(src).save_pretrained(d)
            except Exception as e:     # the tokenizer has no weights; ByT5 ids are bytes + 3
                print(f"[!] tokenizer files not saved: {e}", flush=True)
        barrier("best")

    if args.baseline_first and not resume_dir:
        if args.stage != "finetune" or not dev:
            raise ValueError("--baseline_first requires a finetune run with a non-empty dev set")
        baseline = establish_baseline(state, evaluate())
        if master:
            with open(os.path.join(args.output_dir, "baseline.json"), "w") as f:
                json.dump(baseline, f, indent=1)
            with open(os.path.join(args.output_dir, "eval_log.jsonl"), "a") as f:
                f.write(json.dumps(baseline) + "\n")
        save_best()
        save_resume()
        log("baseline " + json.dumps(baseline))

    # ---- training loop
    model.train()
    budget = args.time_budget_hours * 3600
    loss_acc, n_acc, t_log = 0.0, 0, time.time()
    skipped = len(state.get("skipped", []))
    done = state["step"] >= total
    while not done:
        if args.stage == "dapt":
            items = dapt_samples(args.dapt_corpus, state["epoch"], args.seed)
        batches = training_batches(train, len(items) if args.stage == "dapt" else None,
                                   rows, args.seed, state["epoch"])
        for bi in range(state["batch"], len(batches)):
            chunk, a, b = batches[bi]
            micro_batches = []
            for accum in range(args.gradient_accumulation):
                start = accum * physical_rows + rank * args.per_core_batch
                mine = chunk[start:start + args.per_core_batch]
                if args.stage == "dapt":
                    inp, att, lab = dapt_tensors(items, mine, a, b, torch)
                else:
                    inp, att, lab = tensors(train, mine, a, b, torch)
                micro_batches.append((inp, att, lab))
            for g in opt.param_groups:
                g["lr"] = lr_at(state["step"])
            # All microbatches are normalized by one global target-token count (token_loss).
            loss, flags = train_step(net, opt, micro_batches, torch=torch, device=device, world=world,
                                     all_sum=all_sum, autocast=autocast, max_norm=args.max_grad_norm, sync=sync,
                                     reduce_gradients=xm.reduce_gradients if xla else None)
            if flags is not None:
                msg = describe_nonfinite(flags, state["step"] + 1, state["epoch"], bi)
                skipped = skip_or_raise(msg, skipped, args.skip_nonfinite_batches)
                state["skipped"] = state.get("skipped", []) + [[state["step"] + 1, state["epoch"], bi]]
                log(f"[!] {msg}; batch skipped without an update ({skipped}/{args.skip_nonfinite_batches})")
                state["batch"] = bi + 1
                continue
            state["step"] += 1
            state["batch"] = bi + 1
            if state["step"] % args.log_every == 0 or state["step"] == 1:
                lv = all_sum(loss.detach().float()).item() / world
                el = time.time() - START
                log(f"step {state['step']}/{total} epoch {state['epoch']} loss {lv:.4f} lr {lr_at(state['step']):.2e} "
                    f"{(time.time() - t_log) / max(1, args.log_every if state['step'] > 1 else 1):.3f}s/step "
                    f"elapsed {el / 3600:.2f}h")
                t_log = time.time()
            last = state["step"] >= total
            out_of_time = False
            if state["step"] % args.log_every == 0 or last:
                out_of_time = bool(all_sum(torch.tensor(int(time.time() - START > budget), device=device)).item() > 0)
            if args.stage == "finetune" and dev and (state["step"] % args.eval_every == 0 or last or out_of_time):
                m = evaluate()
                m["step"] = state["step"]
                state["history"].append(m)
                log("eval " + json.dumps(m))
                if master:
                    with open(os.path.join(args.output_dir, "eval_log.jsonl"), "a") as f:
                        f.write(json.dumps(m) + "\n")
                if m["dev_score"] > state["best"]:
                    state["best"] = m["dev_score"]
                    state["best_step"] = state["step"]
                    save_best()
                    log(f"best/ updated (dev_score {m['dev_score']:.4f})")
            stop_requested = bool(all_sum(torch.tensor(int(bool(STOP["signal"])), device=device)).item() > 0)
            if state["step"] % args.save_every == 0 or last or out_of_time or stop_requested:
                if args.stage == "dapt":
                    save_best()        # the dapt stage keeps its latest weights as best/
                save_resume()
            if last or out_of_time or stop_requested:
                done = True
                if out_of_time or stop_requested:
                    reason = "time budget" if out_of_time else STOP["signal"] or "signal received by another worker"
                    log(f"stopping at step {state['step']} ({reason}); "
                        "rerun with --resume_from to continue")
                break
        else:
            state["epoch"] += 1
            state["batch"] = 0
    if master:
        with open(os.path.join(args.output_dir, "run_info.json"), "w") as f:
            json.dump({"args": vars(args), "state": {k: v for k, v in state.items() if k != "history"},
                       "total_steps": total, "world": world, "bf16": use_bf16, "hours": (time.time() - START) / 3600,
                       "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                       "dataset_commit": git_head(args.dataset_dir), "model_commit": git_head(HERE)}, f, indent=1)
    has_best = bool(all_sum(torch.tensor(int(os.path.exists(os.path.join(args.output_dir, "best"))), device=device)).item() > 0)
    has_stop = bool(all_sum(torch.tensor(int(bool(STOP["signal"])), device=device)).item() > 0)
    if args.stage == "finetune" and dev and not has_stop and has_best and not args.skip_final_report:
        del net, opt, model
        import gc
        gc.collect()
        final_report(args, dev, torch, device, sync, rank=rank, world=world, master=master, xm=xm if xla else None,
                     distributed=dist)
    barrier("end")
    if dist is not None:
        dist.destroy_process_group()


def git_head(d):
    try:
        return subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def final_report(args, dev, torch, device, sync, rank=0, world=1, master=True, xm=None, distributed=None):
    """Greedy-decoded dev scores of best/ per dev slice and class, sharded across all ranks in lockstep."""
    from transformers import AutoModelForSeq2SeqLM
    import analyzer
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.output_dir, "best"), dtype=torch.float32)
    model.to(device).eval()
    rows = dev if not args.report_cap else dev[:0] + [dev[i] for i in sorted(
        random.Random(0).sample(range(len(dev)), min(args.report_cap, len(dev))))]
    bs = args.per_core_batch
    global_bs = bs * world
    local_decoded = []
    with torch.no_grad():
        for k in range(0, len(rows), global_bs):
            gchunk = list(range(k, min(k + global_bs, len(rows))))
            gchunk_pad = gchunk + [gchunk[-1]] * (global_bs - len(gchunk))
            mine_idx = gchunk_pad[rank * bs:(rank + 1) * bs]
            a = max(bucket(len(rows[i]["input_text"].encode()) + 1, INPUT_BUCKETS) for i in gchunk_pad)
            inp = torch.full((bs, a), PAD, dtype=torch.long)
            for r, i in enumerate(mine_idx):
                x = encode(rows[i]["input_text"], a)
                inp[r, :len(x)] = torch.tensor(x)
            outs = greedy(model, inp.to(device), (inp != PAD).long().to(device), 129, torch, sync)
            for pos_in_mine, o in enumerate(outs):
                global_pos = k + rank * bs + pos_in_mine
                if global_pos < len(rows):
                    local_decoded.append((global_pos, decode(o)))
    if xm is not None and world > 1:
        gathered = xm.rendezvous("final_report_gather", json.dumps(local_decoded, ensure_ascii=False).encode("utf-8"))
        all_decoded = [item for payload in gathered for item in json.loads(payload.decode("utf-8"))]
    elif distributed is not None and world > 1:
        gathered = [None] * world
        distributed.all_gather_object(gathered, local_decoded)
        all_decoded = [item for part in gathered for item in part]
    else:
        all_decoded = local_decoded
    if not master:
        return
    per = collections.defaultdict(collections.Counter)
    confusion = collections.defaultdict(collections.Counter)
    for idx, text in sorted(all_decoded, key=lambda x: x[0]):
        s = rows[idx]
        c = per[f"{s['dev']}/{s['task']}/{s['cls']}"]
        c["n"] += 1
        c["exact"] += text == s["target_text"]
        if s["task"] == tm.POS_TASK:
            confusion[s["cls"]][text.replace("type:", "").strip() or "?"] += 1
        else:
            c["root"] += text.split("|")[0].strip() == s["target_text"].split("|")[0].strip()
            parsed = analyzer.parse_neural_output(text)
            c["parsed"] += parsed is not None
    report = {"rows": len(rows), "by_slice_task_class": {k: dict(v) for k, v in sorted(per.items())},
              "pos_confusion": {k: dict(v) for k, v in sorted(confusion.items())}}
    tot = collections.Counter()
    for k, v in per.items():
        if "/analyze/" in k:
            tot.update(v)
    report["analyze_total"] = {"n": tot["n"], "exact": tot["exact"] / max(1, tot["n"]), "root": tot["root"] / max(1, tot["n"])}
    with open(os.path.join(args.output_dir, "dev_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("dev_report " + json.dumps(report["analyze_total"]), flush=True)


def _mp_fn(index, args):
    run(index, args)


# Kaggle TPU VMs preset TPU_PROCESS_ADDRESSES for a single process; with it set, every spawned
# worker sees one address instead of one per chip and TPU init fails with
# "Expected 8 worker addresses, got 1". Removing it lets PJRT build the local topology itself.
PRESET_TPU_ENV = ("TPU_PROCESS_ADDRESSES",)


def clean_tpu_env(env=None):
    env = os.environ if env is None else env
    removed = [k for k in PRESET_TPU_ENV if env.pop(k, None) is not None]
    env.setdefault("PJRT_DEVICE", "TPU")
    return removed


def _launch_xla(args, argv):
    removed = clean_tpu_env()
    if removed:
        print("removed preset env " + ",".join(removed), flush=True)
    import torch_xla
    single = args.xla_launch == "single"
    # A worker can die after training starts. Retrying silently on one chip
    # changes batch size and schedule and can overwrite a valid checkpoint.
    if hasattr(torch_xla, "launch"):
        torch_xla.launch(_mp_fn, args=(args,), debug_single_process=single)
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp_fn, args=(args,), nprocs=1 if single else None)
    return 0


def main(argv=None):
    args = parse_args(argv)
    args.dataset_dir = os.path.abspath(dataset_loader.require_dataset_dir(args.dataset_dir))
    args.output_dir = os.path.abspath(args.output_dir)
    if args.device == "auto":
        try:
            import torch_xla  # noqa: F401
            args.device = "xla"
        except ImportError:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

    def handler(signum, frame):
        STOP["signal"] = signal.Signals(signum).name
    signal.signal(signal.SIGTERM, handler)

    if args.load_only:
        prepare(args, write=True)
        return 0
    if args.device == "xla":
        return _launch_xla(args, argv)
    else:
        run(0, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
