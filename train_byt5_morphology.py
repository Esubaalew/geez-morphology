#!/usr/bin/env python3
"""Fine-tune ByT5-small to analyze Classical Ge'ez word forms: verbs, and the nouns and adjectives
of the gold tables and of the Kidane Wold dictionary.

Input text:  "analyze: <surface form>" for the full analysis, "root: <surface form>" for the root alone
             (dataset_loader.input_text; the "task" field of a pair picks the prefix and the target).
             --single_prefix gives every input the "analyze: " prefix.
Target text: "root: <ሀ-ለ-መ> | tense: <t> | person: <p> | gender: <g> | number: <n> | stem: <s>" for a verb
             (verbs with an object suffix add " | object: <o>"), and
             "root: <ሀ-ለ-መ> | type: <noun|adjective> | gender: <g> | number: <n>" for a noun or adjective
             (gender only where the gold row has one); see dataset_loader.target_text.
             A root-only pair (task root_only) has the root alone: "root: <root>", or "root: none" for a word
             without root.

Data (read from --dataset_dir, the geez-dataset checkout), as written:
  data/paradigms/synthetic_morphology_train.jsonl        synthetic pairs
      (only this file: the lines held out for the gold val/test roots, data/synthetic/, are never read)
  data/paradigms/merho_ges_gold_train.jsonl              Merho Ges gold-train paradigms
  data/lexicon/classical_dictionary_kidane_wold.jsonl    Kidane Wold nouns and adjectives, headword to root
  data/unified/geez_morphology_rows.jsonl               book rows of the unified dataset, split "train", status empty
      (--no_unified leaves them out); they replace book_forms.jsonl unless --legacy_book_forms is given
  data/lexicon/book_forms.jsonl                          forms printed in the other books (ርባሐ ስም, Kidane Wold's
      grammar and dictionary), split "train" only (--no_book_forms leaves them out)
and, from this repo, data/closed_class_words_without_root.tsv (words without root, target "root: none";
--no_words_without_root leaves them out).
The gold pairs are read by dataset_loader.load_gold_split: the rows listed in
data/paradigms/merho_ges_gold_flagged_rows.tsv and rows with no root do not load. Nothing is
corrected at load time. All gold-train pairs go to training, each verb pair --gold_upsample times (default 4)
and each noun or adjective pair --gold_nominal_upsample times (default 5); 1 means once. The eval split is 10%
of the synthetic pairs (seed 42), so eval loss never includes gold rows. The held-out gold test file is
not read here; use evaluate_gold.py for that.
The dictionary pairs are read by dataset_loader.load_dictionary_pairs (its docstring lists what is left out);
dataset_loader.split_dictionary_pairs holds out whole roots for 10% of them (seed 42) as a second eval
slice, dictionary_nominal (logged as eval_dictionary_nominal_loss next to the synthetic eval_loss), and the
rest go to training. After the split, a pair whose form is a gold val/test form up to spelling
(geez_script.fold_spelling) is dropped from either side (dataset_loader.dictionary_train_eval).
--no_dictionary_pairs leaves the dictionary out.
--root_accuracy_rows N also generates the root of up to N gold val pairs at each evaluation and logs the share
that match the gold root exactly (eval_root_accuracy); --metric_for_best_model root_accuracy then keeps and
loads the best checkpoint by it. When training finishes, the final model's root-only replies on the
dictionary_nominal slice are scored the same way (final_dictionary_nominal_root_accuracy in run_info.json).

The script picks the device (cuda, then mps, then cpu), trains in fp32 (ByT5
is unstable in fp16), saves a checkpoint every --save_steps and keeps the last
two. Output folder (--output_dir, default byt5_morphology_model/):
  checkpoint-<step>/   resumable checkpoints (model, optimizer, trainer_state.json)
  run_info.json        git commits, data sha256, pair counts and arguments of the run
  config.json, model.safetensors, tokenizer files
                       the final model, written when training finishes; evaluate_gold.py
                       and the analyzer load the newest of this and checkpoint-<step>
If --output_dir already holds a checkpoint-*, training resumes from the
latest one; --resume_from can point at a checkpoint (or a folder of
checkpoints) somewhere else, e.g. a previous Kaggle run's output. On SIGTERM or
SIGINT it saves a checkpoint at the end of the current step and exits.

Examples:
  python train_byt5_morphology.py                     # full run, resumes if possible
  python train_byt5_morphology.py --dataset_dir /kaggle/input/geez-dataset \\
      --output_dir /kaggle/working/byt5_morphology_model --resume_from /kaggle/input/prev-run
  python train_byt5_morphology.py --max_steps 5 --sample_limit 200 \\
      --max_eval_samples 32 --output_dir /tmp/byt5_dry_run   # dry run
"""

import argparse
import collections
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
import time
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import geez_script
from geez_script import to_ethiopic_root  # one shared definition
import dataset_loader  # reads the gold files; shared with evaluate_gold.py

MODEL_NAME = "google/byt5-small"
SEED = 42
EVAL_FRACTION = 0.1
SYNTHETIC_FILE = os.path.join("data", "paradigms", "synthetic_morphology_train.jsonl")
GOLD_TRAIN_FILE = os.path.join("data", "paradigms", dataset_loader.gold_file_name("train"))
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "byt5_morphology_model")
EXPORT_FILES = {"train": "train_pairs.jsonl", "eval": "eval_pairs_synthetic.jsonl",
                "eval_dictionary": "eval_pairs_dictionary_nominal.jsonl"}
DICTIONARY_EVAL_NAME = "dictionary_nominal"
# Where a training sample comes from (its "source" field), in the order the counts are printed.
SOURCES = ("synthetic", "gold_verb", "gold_nominal", "dictionary", "word_without_root", "book_forms", "unified_book")
DEFAULT_GOLD_UPSAMPLE = 4
DEFAULT_GOLD_NOMINAL_UPSAMPLE = 5
ROOT_ACCURACY_MAX_NEW_TOKENS = 32   # bytes; "root: " and a 4-radical root take 21
# Decode shared by checkpoint selection (eval_root_accuracy) and evaluate_gold.py scoring.
# Beam-2 is the default so the checkpoint kept for root accuracy is decoded the same way
# as the published gold metric. Pass --num_beams 1 for greedy.
DEFAULT_NUM_BEAMS = 2
DEFAULT_GENERATE_MAX_LENGTH = 128


# ----------------------------------------------------------------------------- data

def gold_train_file(dataset_dir: str) -> Optional[str]:
    path = os.path.join(dataset_dir, GOLD_TRAIN_FILE)
    return path if os.path.exists(path) else None


def load_data(dataset_dir: str, limit: int = 0, stats: Optional[Dict] = None) -> List[Dict]:
    """Synthetic pairs (optionally a uniform sample of `limit`) plus all gold-train pairs, shuffled with seed 42.
    `stats` (a Counter) receives the gold load counts."""
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
        if feat.get("object"):
            target += f" | object: {feat['object']}"
        samples.append({"input_text": dataset_loader.input_text(surface), "target_text": target, "is_gold": False,
                        "task": dataset_loader.ANALYZE_TASK, "source": "synthetic"})

    # Merho Ges gold-train pairs as written (dataset_loader.load_gold_split); targets in
    # dataset_loader.target_text (verbs and nominals).
    gold_file = gold_train_file(dataset_dir)
    if gold_file:
        pairs = dataset_loader.load_gold_split(dataset_dir, "train", stats)
        for p in pairs:
            source = "gold_verb" if p["type"] == "verb" else "gold_nominal"   # names, rootless words: nominal
            samples.append({"input_text": dataset_loader.input_text(p["form"]),
                            "target_text": dataset_loader.target_text(p), "is_gold": True,
                            "task": dataset_loader.task_of(p), "source": source})
        gold_count = len(pairs)
        print(f"Read {gold_count} Merho Ges gold-train pairs from {gold_file}")
    else:
        print("[!] No gold-train file found; training on synthetic pairs only.")

    rng.shuffle(samples)
    return samples


ROOT_ONLY_TASKS = {"on": False}   # set by --root_only_tasks: the old "root: " prefix and root-alone target


def root_only_sample(pair: Dict, source: str, single_prefix: bool = False) -> Dict:
    """A training sample of a dictionary pair, word without root or unified row. By default it has the
    full-analysis prefix and target the analyzer reads at inference ("analyze: <form>" -> "root: ... | type: ...",
    "root: none | type: <class>" for a word without root); with --root_only_tasks, the root-only task."""
    pair = dict(pair)
    if not ROOT_ONLY_TASKS["on"] and pair.get("root") == dataset_loader.NO_ROOT and "type" not in pair:
        import training_mixture
        pair["root"], pair["type"] = None, training_mixture.CLOSED_CLASS_TYPE.get(pair.get("kind"), "particle")
    if not ROOT_ONLY_TASKS["on"] and pair.get("type"):
        pair["task"] = dataset_loader.ANALYZE_TASK
    task = dataset_loader.task_of(pair)
    out = {"input_text": dataset_loader.input_text(pair["form"], task, single_prefix),
           "target_text": dataset_loader.target_text(pair), "is_gold": False, "source": source, "task": task}
    if "record_id" in pair:
        out["record_id"] = pair["record_id"]
    return out


def load_dictionary_samples(dataset_dir: str, stats: Optional[Dict] = None, single_prefix: bool = False):
    """(train, eval) samples of the Kidane Wold noun and adjective pairs (dataset_loader.load_dictionary_pairs),
    split by root with dataset_loader.split_dictionary_pairs, then without the pairs whose form is a gold val/test
    form up to spelling (dataset_loader.dictionary_train_eval). Both are empty when the dictionary file is missing."""
    path = os.path.join(dataset_dir, dataset_loader.KIDANE_WOLD_FILE)
    if not os.path.exists(path):
        print(f"[!] No dictionary file at {path}; no dictionary pairs.")
        return [], []
    stats = stats if stats is not None else collections.Counter()
    pairs = dataset_loader.load_dictionary_pairs(dataset_dir, stats)
    train, evals = dataset_loader.dictionary_train_eval(dataset_dir, pairs, stats)
    print(f"Read {len(pairs)} dictionary noun and adjective pairs from {path}")
    if stats[dataset_loader.GOLD_FORM_UP_TO_SPELLING_EVAL]:
        print(f"[!] {stats[dataset_loader.GOLD_FORM_UP_TO_SPELLING_EVAL]} dictionary_nominal eval pair(s) dropped: "
              "the form is a gold val/test form up to spelling.")

    return ([root_only_sample(p, "dictionary", single_prefix) for p in train],
            [root_only_sample(p, "dictionary", single_prefix) for p in evals])


def load_book_form_samples(dataset_dir: str, stats: Optional[Dict] = None, single_prefix: bool = False):
    """Training samples of the forms printed in the other books (dataset_loader.load_book_forms, split "train":
    roots and forms outside the gold val/test files)."""
    pairs = dataset_loader.load_book_forms(dataset_dir, "train", stats)
    print(f"Read {len(pairs)} book-form pairs from {dataset_loader.BOOK_FORMS_FILE}")
    return [{"input_text": dataset_loader.input_text(p["form"], single_prefix=single_prefix),
             "target_text": dataset_loader.target_text(p), "is_gold": True, "source": "book_forms",
             "task": dataset_loader.ANALYZE_TASK} for p in pairs]


def load_unified_samples(dataset_dir: str, stats: Optional[Dict] = None, single_prefix: bool = False):
    """Training samples of the book rows of data/unified/geez_morphology_rows.jsonl (dataset_loader.load_unified,
    split "train", rows with an empty status). Rootless classes (function words and the like) have target
    'root: none | type: ...'. [] when the file is missing."""
    pairs = dataset_loader.load_unified(dataset_dir, "train", ("book",), stats)
    print(f"Read {len(pairs)} unified book rows (split train) from {dataset_loader.UNIFIED_FILE}")
    return [root_only_sample(p, "unified_book", single_prefix) | {"is_gold": True} for p in pairs]


def load_words_without_root_samples(dataset_dir: str, stats: Optional[Dict] = None, single_prefix: bool = False):
    """Training samples (target "root: none") of the words without root (dataset_loader.load_words_without_root)."""
    pairs = dataset_loader.load_words_without_root(dataset_dir, stats)
    print(f"Read {len(pairs)} words without root from {dataset_loader.WORDS_WITHOUT_ROOT_FILE}")
    return [root_only_sample(p, "word_without_root", single_prefix) for p in pairs]


def split_samples(samples: List[Dict], gold_upsample: int = 1, gold_nominal_upsample: int = 1):
    """All gold pairs to train; eval = last 10% of the synthetic pairs. In train, each gold verb pair appears
    gold_upsample times and each gold noun or adjective pair gold_nominal_upsample times (eval is never
    upsampled). Train order shuffled with seed 42. Returns (train, eval, synthetic count, gold count), the counts
    before upsampling."""
    synthetic = [x for x in samples if not x["is_gold"]]
    gold = [x for x in samples if x["is_gold"]]
    cut = int(len(synthetic) * (1 - EVAL_FRACTION))
    repeats = {"gold_verb": gold_upsample, "gold_nominal": gold_nominal_upsample}
    train = synthetic[:cut] + [x for x in gold for _ in range(repeats.get(x.get("source"), gold_upsample))]
    evals = synthetic[cut:]
    random.Random(SEED).shuffle(train)
    return train, evals, len(synthetic), len(gold)


def counts_by_source(samples: List[Dict]) -> Dict[str, int]:
    found = collections.Counter(x.get("source") for x in samples)
    return {k: found[k] for k in SOURCES if found[k]}


def filter_heldout_samples(samples, heldout_pairs):
    """Protect held-out evaluation forms across every source, including gold homographs.

    The legacy trainer allows spelling variants of roots, but never the same exact root or
    a held-out form (including its spelling variants). Keep that policy consistent across sources.
    """
    forms = {geez_script.fold_spelling(form) for form, _ in heldout_pairs}
    roots = {root for _, root in heldout_pairs if root}
    return [s for s in samples
            if geez_script.fold_spelling(s["input_text"].split(":", 1)[-1].strip()) not in forms
            and root_in_output(s["target_text"]) not in roots]


def export_row(sample: Dict) -> Dict:
    """The line --export_dir writes for a sample. A full-analysis sample keeps its input_text, target_text and
    is_gold; a root-only sample keeps all its fields."""
    if sample.get("task", dataset_loader.ANALYZE_TASK) == dataset_loader.ANALYZE_TASK:
        return {k: sample[k] for k in ("input_text", "target_text", "is_gold")}
    return sample


def root_in_output(text: str) -> str:
    """The root of a model output or target ("root: <root> | ..." or "root: <root>"), "" when there is none."""
    m = re.match(r"\s*root:\s*([^|]*)", text)
    return m.group(1).strip() if m else ""


def dev_eval_samples(dataset_dir: str, gold_only: bool = False) -> List[Dict]:
    """Full-analysis eval samples from gold val plus (unless gold_only) unified book val rows."""
    if not os.path.exists(dataset_loader.gold_file_path(dataset_dir, "val")):
        return []
    pairs = dataset_loader.load_gold_split(dataset_dir, "val")
    if not gold_only:
        pairs = pairs + dataset_loader.load_unified(dataset_dir, "val", ("book",))
    return [{"input_text": dataset_loader.input_text(p["form"]),
             "target_text": dataset_loader.target_text(p), "is_gold": True,
             "task": dataset_loader.ANALYZE_TASK, "source": "dev_val"} for p in pairs]


def root_accuracy_items(dataset_dir: str, n: int, gold_only: bool = False) -> List[tuple]:
    """(input text, gold root) of up to n dev pairs (a seeded sample in file order when there are more), with the
    full-analysis prefix, for eval_root_accuracy. The dev pairs are the gold val pairs plus, unless gold_only, the
    book rows of the unified file in split val (status empty); a word without root has the root "none". The dev
    set does not depend on --no_unified or --seed, so runs of an ablation select checkpoints on the same words."""
    if n <= 0 or not os.path.exists(dataset_loader.gold_file_path(dataset_dir, "val")):
        return []
    pairs = dataset_loader.load_gold_split(dataset_dir, "val")
    if not gold_only:
        pairs = pairs + dataset_loader.load_unified(dataset_dir, "val", ("book",))
    if n < len(pairs):
        keep = sorted(random.Random(SEED).sample(range(len(pairs)), n))
        pairs = [pairs[i] for i in keep]
    return [(dataset_loader.input_text(p["form"]), p["root"] or "none") for p in pairs]


def sample_items(samples: List[Dict]) -> List[tuple]:
    """(input text, gold root) of training-format samples, e.g. the dictionary_nominal slice."""
    return [(s["input_text"], root_in_output(s["target_text"])) for s in samples]


def root_accuracy(predict, items: List[tuple], batch_size: int) -> float:
    """Share of `items` ((input text, gold root)) whose generated root (root_in_output) equals the gold root
    exactly. `predict` maps a list of input texts to the list of generated texts."""
    hits = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        outputs = predict([text for text, _ in batch])
        hits += sum(root_in_output(o) == root for o, (_, root) in zip(outputs, batch))
    return hits / len(items) if items else 0.0


# ----------------------------------------------------------------------------- run helpers

def pick_device(requested: str = "auto", torch_module=None) -> str:
    if torch_module is None:
        import torch
    else:
        torch = torch_module
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def decode_generate_kwargs(num_beams=None, max_length=None, max_new_tokens=None):
    """model.generate kwargs shared by checkpoint selection and evaluate_gold scoring."""
    kwargs = {
        "num_beams": DEFAULT_NUM_BEAMS if num_beams is None else num_beams,
        "do_sample": False,
    }
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    else:
        kwargs["max_length"] = DEFAULT_GENERATE_MAX_LENGTH if max_length is None else max_length
    return kwargs


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


def _trainer_global_step(folder: str) -> Optional[int]:
    path = os.path.join(folder, "trainer_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            step = json.load(f).get("global_step")
        return int(step) if step is not None else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _model_mtime(folder: str) -> float:
    latest = 0.0
    for name in ("model.safetensors", "pytorch_model.bin", "config.json"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            latest = max(latest, os.path.getmtime(path))
    return latest or (os.path.getmtime(folder) if os.path.isdir(folder) else 0.0)


def model_to_load(model_dir: str) -> str:
    """The model to load from a training output folder: the newest among the final model
    at the top of model_dir (config.json, written when a run finished) and its
    checkpoint-<step> folders. A higher step wins when it is known (the checkpoint-N
    name, or trainer_state.json global_step at the top); otherwise the newer
    modification time wins, so a stale top-level model does not beat a newer checkpoint
    after a resume/extend. When steps and times tie, the top-level model is kept (a
    finished run's save_total_limit checkpoints can be older than the final step).
    Used by evaluate_gold.py and the analyzer."""
    if not os.path.isdir(model_dir):
        return model_dir
    # The TPU trainer exports its selected model in best/, without a model at the run root.
    best = os.path.join(model_dir, "best")
    if os.path.exists(os.path.join(best, "config.json")) and not os.path.exists(os.path.join(model_dir, "config.json")) \
            and not any(CHECKPOINT_RE.match(name) and os.path.isdir(os.path.join(model_dir, name))
                        for name in os.listdir(model_dir)):
        return best
    top = model_dir if os.path.exists(os.path.join(model_dir, "config.json")) else None
    ckpts = []
    for name in os.listdir(model_dir):
        if CHECKPOINT_RE.match(name) and os.path.isdir(os.path.join(model_dir, name)):
            ckpts.append(os.path.join(model_dir, name))
    latest = max(ckpts, key=lambda p: int(CHECKPOINT_RE.match(os.path.basename(p)).group(1))) if ckpts else None
    if top and latest:
        top_step = _trainer_global_step(top)
        latest_step = int(CHECKPOINT_RE.match(os.path.basename(latest)).group(1))
        if top_step is not None:
            return latest if latest_step > top_step else top
        if _model_mtime(latest) > _model_mtime(top):
            return latest
        return top
    return latest or top or model_dir


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
    p.add_argument("--dataset_dir", default=dataset_loader.default_dataset_dir(),
                   help="Checkout of the geez-dataset repo (default: $GEEZ_DATASET_DIR, "
                        "else ../geez-dataset or ../geez-dataset-v1)")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                   help="Where the checkpoints, run_info.json and the final model go (default: byt5_morphology_model/)")
    p.add_argument("--resume_from", default=None,
                   help="checkpoint-N folder, or a folder of them, to resume from when output_dir has none")
    p.add_argument("--init_from", default=None,
                   help="Model or run folder whose weights start a new fine-tune (e.g. the selected A2 checkpoint); "
                        "a resume checkpoint takes precedence")
    p.add_argument("--sample_limit", type=int, default=0, help="Use N synthetic pairs (0 = all, the default)")
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
    p.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS,
                   help="Beam width for checkpoint-selection decode (eval_root_accuracy); "
                        f"evaluate_gold.py uses the same default ({DEFAULT_NUM_BEAMS})")
    p.add_argument("--load_only", action="store_true", help="Load and split the data, print counts, exit")
    p.add_argument("--export_dir", default=None,
                   help="Write the loaded pairs to this folder (train_pairs.jsonl, eval_pairs_synthetic.jsonl, "
                        "eval_pairs_dictionary_nominal.jsonl)")
    p.add_argument("--no_dictionary_pairs", action="store_true",
                   help="Leave out the Kidane Wold noun and adjective pairs (training and the dictionary_nominal eval)")
    p.add_argument("--no_words_without_root", action="store_true",
                   help="Leave out the words without root (data/closed_class_words_without_root.tsv, target 'root: none')")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Seed of the training run: dropout, data order and the shuffle of the training mixture "
                        "(default 42). The splits and the dev set do not depend on it")
    p.add_argument("--dev_gold_only", action="store_true",
                   help="Select checkpoints on the gold val pairs only (default: gold val plus the unified book val rows)")
    p.add_argument("--no_unified", action="store_true",
                   help="Leave out the unified book rows (data/unified/geez_morphology_rows.jsonl, split train)")
    p.add_argument("--legacy_book_forms", action="store_true",
                   help="Also read data/lexicon/book_forms.jsonl and the words without root when the unified file "
                        "is used (by default the unified rows replace them)")
    p.add_argument("--no_book_forms", action="store_true",
                   help="Leave out the forms printed in the other books (data/lexicon/book_forms.jsonl, split train)")
    p.add_argument("--single_prefix", action="store_true",
                   help="Give every input the prefix 'analyze: ' (root-only pairs too) instead of one prefix per task")
    p.add_argument("--gold_upsample", type=int, default=DEFAULT_GOLD_UPSAMPLE,
                   help="Times each gold-train verb pair appears in training (1 = once, no upsampling)")
    p.add_argument("--gold_nominal_upsample", type=int, default=DEFAULT_GOLD_NOMINAL_UPSAMPLE,
                   help="Times each gold-train noun or adjective pair appears in training (1 = once)")
    p.add_argument("--unified_upsample", type=int, default=1,
                   help="Times each unified book train row appears in training (1 = once)")
    p.add_argument("--root_accuracy_rows", type=int, default=0,
                   help="At each evaluation, generate the root of up to N gold val pairs and log the exact-match "
                        "share as eval_root_accuracy (0 = off)")
    p.add_argument("--eval_steps", type=int, default=0,
                   help="Evaluate every N steps (0 = once per epoch; with --metric_for_best_model, every --save_steps)")
    p.add_argument("--metric_for_best_model", default=None,
                   help="Keep and load at the end the checkpoint with the best value of this eval metric, e.g. "
                        "root_accuracy (needs --root_accuracy_rows), loss or dictionary_nominal_loss")
    p.add_argument("--root_only_tasks", action="store_true",
                   help="Train dictionary pairs and words without root as the old root-only task ('root: ' prefix, "
                        "root-alone target) instead of the full analysis")
    p.add_argument("--no_test_inference", action="store_true", help="Skip the sample predictions after training")
    args = p.parse_args(argv)
    if args.grad_accum < 1 or args.batch_size < 1:
        p.error("--batch_size and --grad_accum must be >= 1")
    if args.gold_upsample < 1 or args.gold_nominal_upsample < 1 or args.unified_upsample < 1:
        p.error("--gold_upsample, --gold_nominal_upsample and --unified_upsample must be >= 1")
    if args.metric_for_best_model and args.metric_for_best_model.removeprefix("eval_") == "root_accuracy" \
            and args.root_accuracy_rows < 1:
        p.error("--metric_for_best_model root_accuracy needs --root_accuracy_rows")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    ROOT_ONLY_TASKS["on"] = args.root_only_tasks
    args.dataset_dir = os.path.abspath(dataset_loader.require_dataset_dir(args.dataset_dir))
    args.output_dir = os.path.abspath(args.output_dir)

    load_stats = collections.Counter()
    samples = load_data(args.dataset_dir, limit=args.sample_limit, stats=load_stats)
    train_samples, eval_samples, n_syn, n_gold = split_samples(samples, args.gold_upsample, args.gold_nominal_upsample)
    dictionary_stats = collections.Counter()
    dictionary_train, dictionary_eval = [], []
    if not args.no_dictionary_pairs:
        dictionary_train, dictionary_eval = load_dictionary_samples(args.dataset_dir, dictionary_stats,
                                                                    args.single_prefix)
    without_root_stats = collections.Counter()
    without_root = [] if args.no_words_without_root else \
        load_words_without_root_samples(args.dataset_dir, without_root_stats, args.single_prefix)
    book_stats = collections.Counter()
    unified_stats = collections.Counter()
    unified = [] if args.no_unified else load_unified_samples(args.dataset_dir, unified_stats, args.single_prefix)
    # a word without root that the unified file holds out (split val or test) is never trained on
    heldout_forms = {p["form"] for sp in ("val", "test") for p in dataset_loader.load_unified(args.dataset_dir, sp, ("book",))}
    if heldout_forms and without_root:
        n_before = len(without_root)
        without_root = [s for s in without_root if s["input_text"].split(":", 1)[-1].strip() not in heldout_forms]
        if n_before != len(without_root):
            print(f"{n_before - len(without_root)} word(s) without root dropped: held out in the unified file.")
    if unified and not args.legacy_book_forms:
        # the unified rows replace book_forms.jsonl; words without root and dictionary pairs whose form is already
        # in the unified rows are dropped so the full-analysis unified target is not duplicated by a root-only target,
        # and synthetic pairs whose surface form is a held-out unified val/test book form are dropped
        unified_forms = {s["input_text"].split(":", 1)[-1].strip() for s in unified}
        all_unified_forms = unified_forms | heldout_forms
        if heldout_forms:
            train_samples = [s for s in train_samples
                             if s["is_gold"] or s["input_text"].split(":", 1)[-1].strip() not in heldout_forms]
            eval_samples = [s for s in eval_samples
                            if s["input_text"].split(":", 1)[-1].strip() not in heldout_forms]
        n_before = len(without_root)
        without_root = [s for s in without_root if s["input_text"].split(":", 1)[-1].strip() not in unified_forms]
        n_dict_tr_before, n_dict_ev_before = len(dictionary_train), len(dictionary_eval)
        dictionary_train = [s for s in dictionary_train if s["input_text"].split(":", 1)[-1].strip() not in all_unified_forms]
        dictionary_eval = [s for s in dictionary_eval if s["input_text"].split(":", 1)[-1].strip() not in all_unified_forms]
        print(f"Unified rows replace book_forms.jsonl; dropped {n_before - len(without_root)} word(s) without root, "
              f"{n_dict_tr_before - len(dictionary_train)} dictionary train pair(s), and "
              f"{n_dict_ev_before - len(dictionary_eval)} dictionary eval pair(s) (same form in the unified rows).")
        book_forms = []
    else:
        book_forms = [] if args.no_book_forms else \
            load_book_form_samples(args.dataset_dir, book_stats, args.single_prefix)
    unified_upsampled = [x for x in unified for _ in range(args.unified_upsample)]
    if dictionary_train or without_root or book_forms or unified_upsampled:
        train_samples = train_samples + dictionary_train + without_root + book_forms + unified_upsampled
        random.Random(args.seed).shuffle(train_samples)
    from scripts.audit_train_leakage import heldout
    held_pairs = [p for pairs in heldout(args.dataset_dir).values() for p in pairs]
    n_before = len(train_samples)
    train_samples = filter_heldout_samples(train_samples, held_pairs)
    eval_samples = filter_heldout_samples(eval_samples, held_pairs)
    dictionary_eval = filter_heldout_samples(dictionary_eval, held_pairs)
    if n_before != len(train_samples):
        print(f"Dropped {n_before - len(train_samples)} training samples matching held-out forms or roots.")
    if args.max_eval_samples and args.max_eval_samples < len(eval_samples):
        eval_samples = eval_samples[:args.max_eval_samples]
    if args.max_eval_samples and args.max_eval_samples < len(dictionary_eval):
        dictionary_eval = dictionary_eval[:args.max_eval_samples]
    print("Gold-train load counts:")
    for k, v in sorted(load_stats.items()):
        print(f"  {k}: {v}")
    if dictionary_stats:
        print("Dictionary load counts:")
        for k, v in sorted(dictionary_stats.items()):
            print(f"  {k}: {v}")
    if book_stats:
        print("Book-form load counts:")
        for k, v in sorted(book_stats.items()):
            print(f"  {k}: {v}")
    if without_root_stats:
        print("Words-without-root load counts:")
        for k, v in sorted(without_root_stats.items()):
            print(f"  {k}: {v}")
    train_by_source = counts_by_source(train_samples)
    print(f"Synthetic pairs: {n_syn}. Gold-train pairs: {n_gold}. "
          f"Dictionary pairs: {len(dictionary_train)} train, {len(dictionary_eval)} eval. "
          f"Words without root: {len(without_root)}. Book forms: {len(book_forms)}. Unified book rows: {len(unified)}.")
    print(f"Gold upsampling: verbs x{args.gold_upsample}, nouns and adjectives x{args.gold_nominal_upsample}. "
          f"Prefixes: {'analyze: for every task' if args.single_prefix else 'one per task'}.")
    print(f"Train: {len(train_samples)} pairs (" + ", ".join(f"{k} {v}" for k, v in train_by_source.items()) + "). "
          f"Eval: {len(eval_samples)} synthetic pairs; {DICTIONARY_EVAL_NAME}: {len(dictionary_eval)} pairs.")
    dev_eval = dev_eval_samples(args.dataset_dir, args.dev_gold_only)
    if args.max_eval_samples and args.max_eval_samples < len(dev_eval):
        dev_eval = random.Random(args.seed).sample(dev_eval, args.max_eval_samples)
    root_items = root_accuracy_items(args.dataset_dir, args.root_accuracy_rows, args.dev_gold_only)
    if args.root_accuracy_rows:
        print(f"Root accuracy at each evaluation: {len(root_items)} gold val pairs.")
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        for name, rows in (("train", train_samples), ("eval", eval_samples), ("eval_dictionary", dictionary_eval)):
            with open(os.path.join(args.export_dir, EXPORT_FILES[name]), "w", encoding="utf-8") as f:
                for x in rows:
                    f.write(json.dumps(export_row(x), ensure_ascii=False) + "\n")
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
    init_model = model_to_load(args.init_from or MODEL_NAME)
    print(f"Resuming from: {resume}" if resume else f"Starting from: {init_model}")

    total_steps = optimizer_steps(len(train_samples), args.batch_size, args.grad_accum, n_devices,
                                  args.epochs, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Optimizer steps: {total_steps}. Warmup: {warmup_steps}. Save every {args.save_steps}.")

    os.makedirs(args.output_dir, exist_ok=True)
    flagged = os.path.join(args.dataset_dir, dataset_loader.PARADIGMS_DIR, dataset_loader.FLAGGED_ROWS_FILE)
    data_files = {"synthetic": os.path.join(args.dataset_dir, SYNTHETIC_FILE),
                  "gold_train": gold_train_file(args.dataset_dir),
                  "gold_flagged_rows": flagged if os.path.exists(flagged) else None}
    if not args.no_dictionary_pairs:
        dictionary = os.path.join(args.dataset_dir, dataset_loader.KIDANE_WOLD_FILE)
        data_files["dictionary"] = dictionary if os.path.exists(dictionary) else None
    if unified:
        data_files["unified"] = os.path.join(args.dataset_dir, dataset_loader.UNIFIED_FILE)
    if not args.no_words_without_root:
        data_files["words_without_root"] = dataset_loader.WORDS_WITHOUT_ROOT_FILE
    run_info = {
        "seed": args.seed,
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model_repo": git_commit(SCRIPT_DIR),
        "dataset_repo": git_commit(args.dataset_dir),
        "data_sha256": {k: (sha256_file(v) if v else None) for k, v in data_files.items()},
        "data_files": data_files,
        "counts": {"synthetic": n_syn, "gold_train": n_gold, "dictionary_train": len(dictionary_train),
                   "words_without_root": len(without_root), "unified_book": len(unified), "train": len(train_samples),
                   "train_by_source": train_by_source, "eval": len(eval_samples),
                   "eval_" + DICTIONARY_EVAL_NAME: len(dictionary_eval), "root_accuracy_pairs": len(root_items)},
        "load_counts": dict(sorted(load_stats.items())),
        "dictionary_load_counts": dict(sorted(dictionary_stats.items())),
        "unified_load_counts": dict(sorted(unified_stats.items())),
        "words_without_root_load_counts": dict(sorted(without_root_stats.items())),
        "device": device, "n_devices": n_devices, "precision": "fp32",
        "optimizer_steps": total_steps, "warmup_steps": warmup_steps,
        "resumed_from": resume,
        "initialized_from": init_model,
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
    model = AutoModelForSeq2SeqLM.from_pretrained(init_model)

    eval_every = args.eval_steps or (args.save_steps if args.metric_for_best_model else 0)
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
        eval_strategy="steps" if eval_every else "epoch",
        eval_steps=eval_every or None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=bool(args.metric_for_best_model),
        predict_with_generate=False,
        fp16=False,
        bf16=False,
        use_cpu=(device == "cpu"),
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.num_workers,
        report_to="none",
    )

    def predict(texts):
        model = trainer.model
        was_training = model.training
        model.eval()
        enc = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, **decode_generate_kwargs(
                num_beams=args.num_beams, max_new_tokens=ROOT_ACCURACY_MAX_NEW_TOKENS))
        model.train(was_training)
        return tokenizer.batch_decode(out, skip_special_tokens=True)

    class TrainerWithDictionaryEval(Seq2SeqTrainer):
        """Each evaluation of the synthetic split (eval_loss) also scores the dictionary_nominal slice
        (eval_dictionary_nominal_loss) and, with --root_accuracy_rows, the generated roots of gold val pairs
        (eval_root_accuracy, and eval_root_accuracy_seconds for the time it took)."""
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **kwargs)
            if eval_dataset is None and metric_key_prefix == "eval" and dev_eval:
                metrics.update(super().evaluate(GeezDataset(dev_eval, tokenizer), ignore_keys,
                                                "eval_dev", **kwargs))
            if eval_dataset is None and metric_key_prefix == "eval" and dictionary_eval:
                metrics.update(super().evaluate(GeezDataset(dictionary_eval, tokenizer), ignore_keys,
                                                f"eval_{DICTIONARY_EVAL_NAME}", **kwargs))
            if eval_dataset is None and metric_key_prefix == "eval" and root_items:
                start = time.time()
                extra = {"eval_root_accuracy": root_accuracy(predict, root_items, args.batch_size)}
                extra["eval_root_accuracy_seconds"] = round(time.time() - start, 2)
                self.log(extra)
                metrics.update(extra)
            return metrics

    trainer = TrainerWithDictionaryEval(
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
    if dictionary_eval:
        # Root-only replies of the final model on the dictionary_nominal slice, exact root match.
        run_info["final_" + DICTIONARY_EVAL_NAME + "_root_accuracy"] = acc = \
            root_accuracy(predict, sample_items(dictionary_eval), args.batch_size)
        print(f"Root accuracy on {DICTIONARY_EVAL_NAME} ({len(dictionary_eval)} pairs): {acc:.4f}")
    with open(os.path.join(args.output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    print(f"Training finished at step {trainer.state.global_step}. Output: {args.output_dir}")

    if not args.no_test_inference:
        model.eval()
        for word in ["ይቀትሉ", "ተቀተለት", "ይባርኩ", "ይቀድሱ", "ቀተለ"]:
            inp = tokenizer(dataset_loader.input_text(word), return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, **decode_generate_kwargs(num_beams=args.num_beams),
                                     repetition_penalty=1.5)
            print(f"  {word} -> {tokenizer.decode(out[0], skip_special_tokens=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
