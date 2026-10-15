#!/usr/bin/env python3
"""Train ByT5 on Ge'ez morphology on a TPU VM (PyTorch/XLA, one process per TPU core, data parallel), on CUDA,
or on the CPU (for a dry run). Two stages:

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
and dev_score = (dev_root + dev_exact) / 2 picks best/. At the end best/ is reloaded and dev_report.json gets the
greedy-decoded scores per dev slice and per class, with the class confusion of the pos: task.

Output (--output_dir, lean): best/ (bf16 safetensors + config, the model to use), resume/ (fp32 weights, optimizer,
position; the latest only), mixture_report.json, leak_audit.txt, eval_log.jsonl, dev_report.json, run_info.json.
Resume: rerun with the same --output_dir, or point --resume_from at a previous run's output folder. The run stops
cleanly (saving resume/) when --time_budget_hours is reached or on SIGTERM.
"""
import argparse
import collections
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

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
    ids = [b + 3 for b in text.encode("utf-8")][:max_len - 1]
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
    return sizes[-1]


def make_batches(samples, rows, seed, epoch, drop_last=True):
    """Lists of (sample indices, input len, target len), each `rows` long, in a seeded order per epoch."""
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
    n_noise = max(1, round(n * noise))
    n_spans = max(1, round(n_noise / mean_span))
    starts = sorted(rng.sample(range(n), min(n_spans, n)))
    mask = [False] * n
    for s in starts:
        for j in range(s, min(n, s + max(1, int(rng.expovariate(1 / mean_span))))):
            mask[j] = True
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
    for k in range(0, len(data) - chunk, chunk):
        inp, tgt = corrupt(data[k:k + chunk], rng)
        out.append((inp[:DAPT_INPUT - 1] + [EOS], tgt[:DAPT_TARGET - 1] + [EOS]))
    return out


# ----------------------------------------------------------------------------- args

def add_data_args(p):
    p.add_argument("--sample_limit", type=int, default=0, help="Use N synthetic pairs (0 = all)")
    p.add_argument("--gold_upsample", type=int, default=4)
    p.add_argument("--gold_nominal_upsample", type=int, default=5)
    p.add_argument("--unified_upsample", type=int, default=3)
    p.add_argument("--rare_class_min", type=int, default=400, help="Rare classes are repeated up to this many rows")
    p.add_argument("--rare_class_max_factor", type=int, default=20)
    p.add_argument("--no_unified", action="store_true")
    p.add_argument("--dictionary_radicals", default="2,3,4,5", help="Root lengths of the dictionary pairs")
    p.add_argument("--biradical_dictionary", action="store_true",
                   help="Also train the 2-radical dictionary roots (off until the 2-radical table (scripts/export_biradical_roots.py) is checked)")
    p.add_argument("--extra_synthetic", default="", help="Comma list of extra_synthetic.FAMILIES (default none)")
    p.add_argument("--no_generate_task", dest="generate_task", action="store_false")
    p.add_argument("--no_pos_task", dest="pos_task", action="store_false")
    p.add_argument("--seed", type=int, default=42)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_dir", default=os.environ.get("GEEZ_DATASET_DIR", "../geez-dataset"))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--stage", choices=["finetune", "dapt"], default="finetune")
    p.add_argument("--dapt_corpus", default=None, help="Text from build_dapt_corpus.py (built on the fly if missing)")
    p.add_argument("--model_name", default="google/byt5-small")
    p.add_argument("--init_from", default=None, help="Start from this model folder (e.g. the dapt stage's best/)")
    p.add_argument("--resume_from", default=None, help="A previous run's output folder (its resume/ is used)")
    p.add_argument("--device", choices=["auto", "xla", "cuda", "cpu"], default="auto")
    p.add_argument("--bf16", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--per_core_batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--dev_cap", type=int, default=0, help="Seeded sample of N dev pairs (0 = all)")
    p.add_argument("--report_cap", type=int, default=0, help="Greedy-decode at most N dev rows at the end (0 = all)")
    p.add_argument("--time_budget_hours", type=float, default=8.0)
    p.add_argument("--train_limit", type=int, default=0, help="Dry runs: a seeded sample of N training samples")
    p.add_argument("--load_only", action="store_true", help="Build the mixture, write reports and the audit, exit")
    p.add_argument("--export_dir", default=None, help="Also write train_pairs.jsonl / dev_pairs.jsonl here")
    add_data_args(p)
    return p.parse_args(argv)


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
    train, _, report = tm.build(args.dataset_dir, args)
    if args.train_limit and args.train_limit < len(train):      # dry runs: a seeded sample of the mixture
        train = random.Random(args.seed).sample(train, args.train_limit)
    dev = tm.dev_samples(args.dataset_dir, args.dev_cap, pos_task=args.pos_task)
    if write:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "mixture_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        export = args.export_dir or os.path.join(args.output_dir, "_pairs")
        os.makedirs(export, exist_ok=True)
        tm.write_jsonl(os.path.join(export, "train_pairs.jsonl"), train)
        tm.write_jsonl(os.path.join(export, "dev_pairs.jsonl"), dev)
        audit = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "audit_train_leakage.py"),
                                "--dataset_dir", args.dataset_dir, "--train", os.path.join(export, "train_pairs.jsonl")],
                               capture_output=True, text=True)
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

def run(rank, args):
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from transformers.optimization import Adafactor

    xla = args.device == "xla"
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
        world, rank = 1, 0
        device = torch.device("cuda" if args.device == "cuda" else "cpu")
        sync = None
    master = rank == 0
    use_bf16 = args.bf16 == "yes" or (args.bf16 == "auto" and args.device in ("xla", "cuda"))

    def log(*a):
        if master:
            print(*a, flush=True)

    def all_sum(t):
        if xla and world > 1:
            return xm.all_reduce(xm.REDUCE_SUM, t)
        return t

    def barrier(tag):
        if xla:
            xm.rendezvous(tag)

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
    if resume_dir:
        ck = torch.load(os.path.join(resume_dir, "state.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        state = ck["state"]
        log(f"resumed from {resume_dir} at step {state['step']}")
    model.to(device)
    opt = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    if resume_dir:
        opt.load_state_dict(ck["opt"])
        del ck

    rows = args.per_core_batch * world
    if args.stage == "dapt":
        n_items = len(dapt_samples(args.dapt_corpus, 0, args.seed))
    else:
        n_items = len(train)
    steps_per_epoch = max(1, n_items // rows)
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
    dev_batches = make_batches(dev, rows, 0, 0, drop_last=False) if dev else []

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
        cpu = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
        if xla:
            optsd = xm._maybe_convert_to_cpu(opt.state_dict()) if hasattr(xm, "_maybe_convert_to_cpu") else opt.state_dict()
        else:
            optsd = opt.state_dict()
        if master:
            d = os.path.join(args.output_dir, "resume")
            os.makedirs(d, exist_ok=True)
            torch.save({"model": cpu, "opt": optsd, "state": state}, os.path.join(d, "state.pt.tmp"))
            os.replace(os.path.join(d, "state.pt.tmp"), os.path.join(d, "state.pt"))
        barrier("resume")

    def save_best():
        cpu = {k: v.detach().to("cpu").to(torch.bfloat16) for k, v in model.state_dict().items()}
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
                AutoTokenizer.from_pretrained(args.model_name).save_pretrained(d)
            except Exception as e:     # the tokenizer has no weights; ByT5 ids are bytes + 3
                print(f"[!] tokenizer files not saved: {e}", flush=True)
        barrier("best")

    # ---- training loop
    model.train()
    budget = args.time_budget_hours * 3600
    loss_acc, n_acc, t_log = 0.0, 0, time.time()
    done = state["step"] >= total
    while not done:
        if args.stage == "dapt":
            items = dapt_samples(args.dapt_corpus, state["epoch"], args.seed)
            order = list(range(len(items)))
            random.Random(args.seed + state["epoch"]).shuffle(order)
            batches = [(order[k:k + rows], DAPT_INPUT, DAPT_TARGET) for k in range(0, len(order) - rows + 1, rows)]
        else:
            batches = make_batches(train, rows, args.seed, state["epoch"])
        for bi in range(state["batch"], len(batches)):
            chunk, a, b = batches[bi]
            mine = chunk[rank * args.per_core_batch:(rank + 1) * args.per_core_batch]
            if args.stage == "dapt":
                inp = torch.full((len(mine), a), PAD, dtype=torch.long)
                lab = torch.full((len(mine), b), -100, dtype=torch.long)
                for r, i in enumerate(mine):
                    x, y = items[i]
                    inp[r, :len(x)] = torch.tensor(x)
                    lab[r, :len(y)] = torch.tensor(y)
                att = (inp != PAD).long()
            else:
                inp, att, lab = tensors(train, mine, a, b, torch)
            inp, att, lab = inp.to(device), att.to(device), lab.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(state["step"])
            with autocast():
                loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
            loss.backward()
            if xla:
                xm.optimizer_step(opt)
                sync()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
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
            out_of_time = time.time() - START > budget
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
                    save_best()
                    log(f"best/ updated (dev_score {m['dev_score']:.4f})")
            if state["step"] % args.save_every == 0 or last or out_of_time or STOP["signal"]:
                if args.stage == "dapt":
                    save_best()        # the dapt stage keeps its latest weights as best/
                save_resume()
            if last or out_of_time or STOP["signal"]:
                done = True
                if out_of_time or STOP["signal"]:
                    log(f"stopping at step {state['step']} ({'time budget' if out_of_time else STOP['signal']}); "
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
    if args.stage == "finetune" and dev and master and not STOP["signal"] and os.path.exists(os.path.join(args.output_dir, "best")):
        final_report(args, dev, torch, device, sync)
    barrier("end")


def git_head(d):
    try:
        return subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def final_report(args, dev, torch, device, sync):
    """Greedy-decoded dev scores of best/ per dev slice and class, with the pos: task's class confusion."""
    from transformers import AutoModelForSeq2SeqLM
    import analyzer
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.output_dir, "best"), dtype=torch.float32)
    model.to(device).eval()
    rows = dev if not args.report_cap else dev[:0] + [dev[i] for i in sorted(
        random.Random(0).sample(range(len(dev)), min(args.report_cap, len(dev))))]
    per = collections.defaultdict(collections.Counter)
    confusion = collections.defaultdict(collections.Counter)
    bs = args.per_core_batch
    with torch.no_grad():
        for k in range(0, len(rows), bs):
            chunk = rows[k:k + bs]
            n = len(chunk)
            chunk = chunk + [chunk[-1]] * (bs - n)
            a = max(bucket(len(s["input_text"].encode()) + 1, INPUT_BUCKETS) for s in chunk)
            inp = torch.full((bs, a), PAD, dtype=torch.long)
            for r, s in enumerate(chunk):
                x = encode(s["input_text"], a)
                inp[r, :len(x)] = torch.tensor(x)
            outs = greedy(model, inp.to(device), (inp != PAD).long().to(device), 129, torch, sync)
            for s, o in zip(chunk[:n], outs[:n]):
                text = decode(o)
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


def main(argv=None):
    args = parse_args(argv)
    args.dataset_dir = os.path.abspath(args.dataset_dir)
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
        import torch_xla
        if hasattr(torch_xla, "launch"):
            torch_xla.launch(_mp_fn, args=(args,))
        else:
            import torch_xla.distributed.xla_multiprocessing as xmp
            xmp.spawn(_mp_fn, args=(args,))
    else:
        run(0, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
