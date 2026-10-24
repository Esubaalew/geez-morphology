"""Kaggle entry point: verify the private local bundle, then fine-tune its A2 weights."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path("/kaggle/input")
    hits = []
    for p in root.rglob("bundle.json"):
        try:
            if json.loads(p.read_text()).get("format") == "geez-a2-finetune-v1":
                hits.append(p)
        except (ValueError, OSError):
            pass
    if len(hits) != 1:
        raise SystemExit(f"Expected exactly one A2 fine-tune bundle, found {len(hits)}")
    meta_path = hits[0]
    meta = json.loads(meta_path.read_text())
    inp = meta_path.parent
    work = Path("/kaggle/working")
    source = work / "inputs"
    source.mkdir(exist_ok=True)
    for name, expected in meta["archives"].items():
        path = inp / name
        if sha256(path) != expected:
            raise SystemExit(f"Archive hash mismatch: {name}")
        with tarfile.open(path) as archive:
            archive.extractall(source, filter="data")
    checkpoint = source / "a2_best"
    for name, expected in meta["checkpoint_sha256"].items():
        if sha256(checkpoint / name) != expected:
            raise SystemExit(f"A2 checkpoint hash mismatch: {name}")
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        version = ""
    if version != meta["transformers_version"]:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "transformers==" + meta["transformers_version"], "safetensors"], check=True)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No GPU attached. Select GPU T4 ×2 (or P100) before running.")
    workers = torch.cuda.device_count()
    code = source / "model"
    dataset = source / "geez-dataset"
    out = work / "out"
    # The immutable input weights and the new optimizer/output directory are always separate.
    cmd = [str(code / "a2_finetune.py"), "--dataset_dir", str(dataset),
           "--init_from", str(checkpoint), "--output_dir", str(out), "--device", "cuda"]
    for key, value in meta["profile"].items():
        cmd += ["--" + key, str(value)]
    print("A2 bundle:", json.dumps(meta, ensure_ascii=False), flush=True)
    print(f"GPUs: {workers}; precision fp32; source weights verified", flush=True)
    check_out = work / "load_check"
    check_cmd = list(cmd)
    check_cmd[check_cmd.index("--output_dir") + 1] = str(check_out)
    subprocess.run([sys.executable] + check_cmd + ["--load_only"], check=True, cwd=code)
    report = json.loads((check_out / "mixture_report.json").read_text())
    if report["train_samples"] != meta["expected_train_samples"]:
        raise SystemExit("Training mixture count differs from the locally verified bundle")
    shutil.copyfile(meta_path, work / "training_bundle.json")
    launch = ([sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=" + str(workers)] if workers > 1 else [sys.executable])
    subprocess.run(launch + cmd, check=True, cwd=code)


if __name__ == "__main__":
    main()
