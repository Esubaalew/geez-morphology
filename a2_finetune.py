#!/usr/bin/env python3
"""A short, baseline-protected fine-tune from the selected A2 model weights."""
import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--init_from", required=True, help="A2 best/ weights; a new optimizer is created")
    p.add_argument("--output_dir", required=True, help="A fresh run directory, separate from A2")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--global_batch", type=int, default=256)
    p.add_argument("--micro_batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--time_budget_hours", type=float, default=5.0)
    p.add_argument("--dev_cap", type=int, default=0)
    p.add_argument("--report_cap", type=int, default=1000)
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--load_only", action="store_true")
    p.add_argument("--skip_final_report", action="store_true")
    p.add_argument("--dev_seen_manifest", default=str(HERE / "data/a2_training_exposure.json"))
    a = p.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if a.global_batch <= 0 or a.micro_batch <= 0 or a.global_batch % (a.micro_batch * world):
        p.error("global_batch must be positive and divisible by micro_batch × worker count")
    if a.lr <= 0 or a.max_steps <= 0 or a.eval_every <= 0 or a.time_budget_hours <= 0:
        p.error("learning rate, steps, evaluation interval and time budget must be positive")
    init = Path(a.init_from).resolve()
    if not (init / "config.json").is_file() or not (init / "model.safetensors").is_file():
        p.error("init_from must contain A2 config.json and model.safetensors")
    out = Path(a.output_dir).resolve()
    if out == init or out in init.parents or init in out.parents:
        p.error("output_dir must be separate from the source checkpoint")
    if not Path(a.dev_seen_manifest).is_file():
        p.error("the historical training-exposure manifest is missing")
    return a


def trainer_args(a):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    accum = a.global_batch // (a.micro_batch * world)
    args = ["--dataset_dir", a.dataset_dir, "--init_from", a.init_from,
            "--output_dir", a.output_dir, "--device", a.device,
            "--expected_world_size", str(world), "--per_core_batch", str(a.micro_batch),
            "--gradient_accumulation", str(accum), "--lr", str(a.lr), "--max_steps", str(a.max_steps),
            "--warmup_steps", "50", "--eval_every", str(a.eval_every), "--save_every", str(a.eval_every),
            "--log_every", "10", "--time_budget_hours", str(a.time_budget_hours),
            "--bf16", "no", "--gradient_checkpointing", "--baseline_first",
            "--unified_file", "data/unified/geez_morphology_rows_reviewed.jsonl", "--clitic_forms",
            "--synthetic_fraction", "0.15", "--rootless_upsample", "4",
            "--dev_seen_manifest", a.dev_seen_manifest, "--dev_cap", str(a.dev_cap),
            "--report_cap", str(a.report_cap), "--train_limit", str(a.train_limit)]
    if a.load_only:
        args.append("--load_only")
    if a.skip_final_report:
        args.append("--skip_final_report")
    return args


def main(argv=None):
    import train_tpu
    return train_tpu.main(trainer_args(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
