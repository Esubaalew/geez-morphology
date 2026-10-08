#!/usr/bin/env python3
"""Fine-tune ByT5-small to analyze Classical Ge'ez verb forms.

Input text:  "analyze: <surface form>"
Target text: "root: <ሀ-ለ-መ> | tense: <t> | person: <p> | gender: <g> | number: <n> | stem: <s>"

Data (read from --dataset_dir, the geez-dataset checkout), as written:
  data/paradigms/synthetic_morphology_train.jsonl        synthetic pairs
      (only this file: the lines held out for the gold val/test roots, data/synthetic/, are never read)
  data/paradigms/merho_ges_gold_train.jsonl              Merho Ges gold-train paradigms
The gold pairs are read by gold_filters.load_gold_split: the rows listed in
data/paradigms/merho_ges_gold_flagged_rows.tsv and rows with no root do not load. Nothing is
corrected at load time. All gold-train pairs go to training. The eval split is 10% of the synthetic pairs
(seed 42), so eval loss never includes gold rows. The held-out gold test file is
not read here; use evaluate_gold.py for that.

The script picks the device (cuda, then mps, then cpu), trains in fp32 (ByT5
is unstable in fp16), saves a checkpoint every --save_steps and keeps the last
two. If --output_dir already holds a checkpoint-*, training resumes from the
latest one; --resume_from can point at a checkpoint (or a folder of
checkpoints) somewhere else, e.g. a previous Kaggle run's output. On SIGTERM or
SIGINT it saves a checkpoint at the end of the current step and exits.

Examples:
  python train_byt5_morphology.py                     # full run, resumes if possible
  python train_byt5_morphology.py --dataset_dir /kaggle/input/geez-dataset \\
      --output_dir /kaggle/working/checkpoints --resume_from /kaggle/input/prev-run
  python train_byt5_morphology.py --max_steps 5 --sample_limit 200 \\
      --max_eval_samples 32 --output_dir /tmp/byt5_dry_run   # dry run
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from geez_config import to_ethiopic_root  # one shared definition
import gold_filters  # reads the gold files; shared with evaluate_gold.py

MODEL_NAME = "google/byt5-small"
SEED = 42
EVAL_FRACTION = 0.1
SYNTHETIC_FILE = os.path.join("data", "paradigms", "synthetic_morphology_train.jsonl")
GOLD_TRAIN_FILE = os.path.join("data", "paradigms", gold_filters.gold_file_name("train"))
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def load_env_file(path: str) -> None:
    """Read KEY=VALUE lines from a local .env (git-ignored) without overriding the environment."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


# ----------------------------------------------------------------------------- data

def gold_train_file(dataset_dir: str) -> Optional[str]:
    path = os.path.join(dataset_dir, GOLD_TRAIN_FILE)
    return path if os.path.exists(path) else None


def load_data(dataset_dir: str, limit: int = 0, stats: Optional[Dict] = None) -> List[Dict]:
    """Synthetic pairs (optionally a uniform sample of `limit`) plus all gold-train pairs, shuffled with seed 42.
    `stats` (a Counter) receives the gold load counts."""
    import collections
    stats = stats if stats is not None else collections.Counter()
    train_file = os.path.join(dataset_dir, SYNTHETIC_FILE)
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found at: {train_file}")

    print(f"Reading synthetic pairs from {train_file}")
    with open(train_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    rng = random.Random(SEED)
    if limit and limit < len(all_lines):
        print(f"Sampling {limit} of {len(all_lines)} synthetic pairs")
        selected_lines = rng.sample(all_lines, limit)
    else:
        selected_lines = all_lines
        rng.shuffle(selected_lines)

    samples = []
    for line in selected_lines:
        d = json.loads(line)
        feat = d.get("features", {})
        surface = d.get("surface_form", "")
        eth_root = to_ethiopic_root(feat.get("root", ""))
        target = (f"root: {eth_root} | tense: {feat.get('tense', '')} | person: {feat.get('person', '')} | "
                  f"gender: {feat.get('gender', '')} | number: {feat.get('number', '')} | stem: {feat.get('stem', '')}")
        samples.append({"input_text": f"analyze: {surface}", "target_text": target, "is_gold": False})

    # Merho Ges gold-train pairs as written (gold_filters.load_gold_split); targets in
    # gold_filters.target_text (verbs and nominals).
    gold_file = gold_train_file(dataset_dir)
    if gold_file:
        pairs = gold_filters.load_gold_split(dataset_dir, "train", stats)
        for p in pairs:
            samples.append({"input_text": f"analyze: {p['form']}", "target_text": gold_filters.target_text(p),
                            "is_gold": True})
        gold_count = len(pairs)
        print(f"Read {gold_count} Merho Ges gold-train pairs from {gold_file}")
    else:
        print("[!] No gold-train file found; training on synthetic pairs only.")

    rng.shuffle(samples)
    return samples


def split_samples(samples: List[Dict]):
    """All gold pairs to train; eval = last 10% of the synthetic pairs. Train order shuffled with seed 42."""
    synthetic = [x for x in samples if not x["is_gold"]]
    gold = [x for x in samples if x["is_gold"]]
    cut = int(len(synthetic) * (1 - EVAL_FRACTION))
    train = synthetic[:cut] + gold
    evals = synthetic[cut:]
    random.Random(SEED).shuffle(train)
    return train, evals, len(synthetic), len(gold)


# ----------------------------------------------------------------------------- run helpers

def pick_device(requested: str = "auto") -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def latest_checkpoint(folder: Optional[str]) -> Optional[str]:
    """Highest-numbered checkpoint-N folder directly inside `folder` (needs trainer_state.json), else None."""
    if not folder or not os.path.isdir(folder):
        return None
    best, best_step = None, -1
    for name in os.listdir(folder):
        m = CHECKPOINT_RE.match(name)
        path = os.path.join(folder, name)
        if m and os.path.isdir(path) and os.path.exists(os.path.join(path, "trainer_state.json")):
            if int(m.group(1)) > best_step:
                best, best_step = path, int(m.group(1))
    return best


def resolve_resume(output_dir: str, resume_from: Optional[str]) -> Optional[str]:
    """Checkpoint to resume from: the latest in output_dir wins (the run's own progress),
    then --resume_from (a checkpoint-N folder or a folder holding them)."""
    own = latest_checkpoint(output_dir)
    if own:
        return own
    if resume_from:
        if CHECKPOINT_RE.match(os.path.basename(os.path.normpath(resume_from))) and \
                os.path.exists(os.path.join(resume_from, "trainer_state.json")):
            return resume_from
        found = latest_checkpoint(resume_from)
        if found:
            return found
        raise FileNotFoundError(f"--resume_from {resume_from}: no checkpoint-N folder with trainer_state.json")
    return None


def optimizer_steps(n_train: int, batch_size: int, grad_accum: int, n_devices: int, epochs: float, max_steps: int) -> int:
    if max_steps and max_steps > 0:
        return max_steps
    batches_per_epoch = math.ceil(n_train / (batch_size * max(1, n_devices)))
    return math.ceil(batches_per_epoch / grad_accum) * math.ceil(epochs)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(folder: str) -> Dict[str, Optional[str]]:
    def run(*cmd):
        try:
            return subprocess.run(["git", "-C", folder, *cmd], capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(dirty) if commit else None}


class StopFlag:
    """Set by SIGTERM/SIGINT; the callback saves a checkpoint and stops at the end of the step."""
    signal_name: Optional[str] = None


def install_signal_handlers() -> None:
    def handler(signum, frame):
        name = signal.Signals(signum).name
        if StopFlag.signal_name:      # second signal: give up immediately
            print(f"\n[!] Second {name}: exiting without saving.", flush=True)
            os._exit(128 + signum)
        StopFlag.signal_name = name
        print(f"\n[!] {name} received: saving a checkpoint at the end of this step, then exiting.", flush=True)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ----------------------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune ByT5-small on Ge'ez verb morphology (resumable).")
    p.add_argument("--dataset_dir", default=gold_filters.default_dataset_dir(),
                   help="Checkout of the geez-dataset repo (default: $GEEZ_DATASET_DIR, else ../geez-dataset)")
    p.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "checkpoints_full"),
                   help="Where checkpoints, trainer_state.json and run_info.json go")
    p.add_argument("--resume_from", default=None,
                   help="checkpoint-N folder, or a folder of them, to resume from when output_dir has none")
    p.add_argument("--sample_limit", type=int, default=0, help="Use N synthetic pairs (0 = all, the default)")
    p.add_argument("--full", action="store_true", help="Same as --sample_limit 0 (kept for old commands)")
    p.add_argument("--batch_size", type=int, default=16, help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps; effective batch = batch_size x grad_accum x devices")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=3e-4, help="Peak AdamW learning rate")
    p.add_argument("--warmup_ratio", type=float, default=0.03, help="Fraction of optimizer steps for linear warmup")
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=0, help="Stop after N optimizer steps (dry runs)")
    p.add_argument("--max_eval_samples", type=int, default=0, help="Cap the eval split (0 = all)")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--load_only", action="store_true", help="Load and split the data, print counts, exit")
    p.add_argument("--export_dir", default=None,
                   help="Write the loaded train and eval pairs (train.jsonl, eval.jsonl) to this folder")
    p.add_argument("--no_test_inference", action="store_true", help="Skip the sample predictions after training")
    args = p.parse_args(argv)
    if args.full:
        args.sample_limit = 0
    if args.grad_accum < 1 or args.batch_size < 1:
        p.error("--batch_size and --grad_accum must be >= 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    load_env_file(os.path.join(SCRIPT_DIR, ".env"))
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    import collections
    load_stats = collections.Counter()
    samples = load_data(args.dataset_dir, limit=args.sample_limit, stats=load_stats)
    train_samples, eval_samples, n_syn, n_gold = split_samples(samples)
    if args.max_eval_samples and args.max_eval_samples < len(eval_samples):
        eval_samples = eval_samples[:args.max_eval_samples]
    print("Gold-train load counts:")
    for k, v in sorted(load_stats.items()):
        print(f"  {k}: {v}")
    print(f"Synthetic pairs: {n_syn}. Gold-train pairs: {n_gold}.")
    print(f"Train: {len(train_samples)} pairs ({n_gold} gold). Eval: {len(eval_samples)} synthetic pairs.")
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        for name, rows in (("train", train_samples), ("eval", eval_samples)):
            with open(os.path.join(args.export_dir, name + ".jsonl"), "w", encoding="utf-8") as f:
                for x in rows:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"Exported train/eval pairs to {args.export_dir}")
    if args.load_only:
        return 0

    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                                  Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback)
    except ImportError:
        print("[!] Needs torch and transformers:  pip install torch transformers accelerate")
        return 1

    class GeezDataset(Dataset):
        def __init__(self, data, tokenizer, max_len=128):
            self.data, self.tokenizer, self.max_len = data, tokenizer, max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            enc = self.tokenizer(item["input_text"], max_length=self.max_len, truncation=True)
            enc["labels"] = self.tokenizer(text_target=item["target_text"], max_length=self.max_len,
                                           truncation=True)["input_ids"]
            return enc

    class SaveOnSignal(TrainerCallback):
        def on_step_end(self, targs, state, control, **kwargs):
            if StopFlag.signal_name:
                control.should_save = True
                control.should_training_stop = True
            return control

    device = pick_device(args.device)
    n_devices = torch.cuda.device_count() if device == "cuda" else 1
    print(f"Device: {device}" + (f" x{n_devices} ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
          + ". Precision: fp32.")
    print(f"Effective batch: {args.batch_size} x {args.grad_accum} accum x {n_devices} device(s) = "
          f"{args.batch_size * args.grad_accum * n_devices}")

    resume = resolve_resume(args.output_dir, args.resume_from)
    print(f"Resuming from: {resume}" if resume else "Starting from the pretrained model (no checkpoint found).")

    total_steps = optimizer_steps(len(train_samples), args.batch_size, args.grad_accum, n_devices,
                                  args.epochs, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Optimizer steps: {total_steps}. Warmup: {warmup_steps}. Save every {args.save_steps}.")

    os.makedirs(args.output_dir, exist_ok=True)
    flagged = os.path.join(args.dataset_dir, gold_filters.PARADIGMS_DIR, gold_filters.FLAGGED_ROWS_FILE)
    data_files = {"synthetic": os.path.join(args.dataset_dir, SYNTHETIC_FILE),
                  "gold_train": gold_train_file(args.dataset_dir),
                  "gold_flagged_rows": flagged if os.path.exists(flagged) else None}
    run_info = {
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model_repo": git_commit(SCRIPT_DIR),
        "dataset_repo": git_commit(args.dataset_dir),
        "data_sha256": {k: (sha256_file(v) if v else None) for k, v in data_files.items()},
        "data_files": data_files,
        "counts": {"synthetic": n_syn, "gold_train": n_gold, "train": len(train_samples), "eval": len(eval_samples)},
        "load_counts": dict(sorted(load_stats.items())),
        "device": device, "n_devices": n_devices, "precision": "fp32",
        "optimizer_steps": total_steps, "warmup_steps": warmup_steps,
        "resumed_from": resume,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "transformers": __import__("transformers").__version__},
        "args": vars(args),
    }
    if resume:
        prev = os.path.join(os.path.dirname(resume), "run_info.json")
        if os.path.exists(prev):
            with open(prev, encoding="utf-8") as f:
                prev_sha = json.load(f).get("data_sha256")
            if prev_sha and prev_sha != run_info["data_sha256"]:
                print("[!] Warning: the training data differ from the run that wrote this checkpoint.")
    with open(os.path.join(args.output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=False,
        fp16=False,
        bf16=False,
        use_cpu=(device == "cpu"),
        seed=SEED,
        dataloader_num_workers=args.num_workers,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=GeezDataset(train_samples, tokenizer),
        eval_dataset=GeezDataset(eval_samples, tokenizer),
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100),
        callbacks=[SaveOnSignal()],
    )

    install_signal_handlers()
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()   # output_dir/trainer_state.json

    if StopFlag.signal_name:
        print(f"Stopped by {StopFlag.signal_name} at step {trainer.state.global_step}; "
              f"checkpoint saved in {args.output_dir}. Rerun the same command to resume.")
        return 0

    trainer.save_model(args.output_dir)   # final weights + tokenizer at the top of output_dir
    run_info["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    run_info["global_step"] = trainer.state.global_step
    with open(os.path.join(args.output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    print(f"Training finished at step {trainer.state.global_step}. Output: {args.output_dir}")

    if not args.no_test_inference:
        model.eval()
        for word in ["ይቀትሉ", "ተቀተለት", "ይባርኩ", "ይቀድሱ", "ቀተለ"]:
            inp = tokenizer(f"analyze: {word}", return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_length=128, num_beams=2, repetition_penalty=1.5)
            print(f"  {word} -> {tokenizer.decode(out[0], skip_special_tokens=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
