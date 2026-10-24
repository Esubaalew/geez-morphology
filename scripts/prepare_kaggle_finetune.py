#!/usr/bin/env python3
"""Package the local fixed repositories and A2 weights for a private Kaggle run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(repo, dest, prefix):
    """Include local fixes and new nonignored files, without Git metadata or ignored sources/secrets."""
    names = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z", "--cached",
                                     "--others", "--exclude-standard"]).decode().split("\0")
    with tarfile.open(dest, "w:gz", compresslevel=1) as archive:
        for name in sorted(set(n for n in names if n)):
            path = repo / name
            if not path.is_file():
                continue
            if path.is_symlink() or name.startswith(".git/") or path.name.startswith(".env"):
                raise ValueError(f"Refusing unexpected link or secret file: {name}")
            archive.add(path, arcname=prefix + "/" + name, recursive=False)
    return {"head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip(),
            "local_changes": bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"])),
            "archive_sha256": sha256(dest)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--username", required=True)
    p.add_argument("--dataset_dir", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--slug", default="geez-a2-corrected-finetune")
    a = p.parse_args(argv)
    bundle = a.output / "dataset"
    kernel = a.output / "kernel"
    if a.output.exists() and any(a.output.iterdir()):
        p.error("output must be empty; choose a new folder for each bundle")
    bundle.mkdir(parents=True, exist_ok=True)
    kernel.mkdir()
    required = ("config.json", "model.safetensors", "tokenizer_config.json", "added_tokens.json")
    hashes = {name: sha256(a.checkpoint / name) for name in required}
    if hashes["model.safetensors"] != "9c27e9330032fed85e7512dd559357c1e01e39a65887e59be7a13c9cb30a30f3":
        p.error("checkpoint weights do not match the selected A2 best model")
    profile = dict(global_batch=256, micro_batch=8, lr=1e-4, max_steps=1000,
                   eval_every=100, time_budget_hours=5.0, report_cap=1000)
    import a2_finetune
    import train_tpu
    with tempfile.TemporaryDirectory(prefix="geez-a2-load-") as tmp:
        opts = a2_finetune.parse_args(["--dataset_dir", str(a.dataset_dir), "--init_from", str(a.checkpoint),
                                      "--output_dir", tmp, "--load_only"])
        train, dev, report = train_tpu.prepare(train_tpu.parse_args(a2_finetune.trainer_args(opts)), write=False)
        counts = len(train), len(dev)
    sources = {"model": snapshot(ROOT, bundle / "model.tar.gz", "model"),
               "dataset": snapshot(a.dataset_dir.resolve(), bundle / "geez-dataset.tar.gz", "geez-dataset")}
    with tarfile.open(bundle / "a2_best.tar", "w") as archive:
        for name in required:
            archive.add(a.checkpoint / name, arcname="a2_best/" + name, recursive=False)
    manifest = dict(format="geez-a2-finetune-v1", username=a.username, sources=sources,
                    checkpoint_sha256=hashes, transformers_version="5.16.1", profile=profile,
                    expected_train_samples=counts[0], expected_dev_samples=counts[1],
                    dev_exposure_filter=report.get("dev_exposure_filter", {}),
                    archives={name: sha256(bundle / name) for name in
                              ("model.tar.gz", "geez-dataset.tar.gz", "a2_best.tar")})
    (bundle / "bundle.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    metadata = dict(title=a.slug, id=a.username + "/" + a.slug, licenses=[{"name": "other"}],
                    isPrivate=True, description="Private A2 fine-tune inputs. Code: Apache-2.0; source data: "
                    "see bundled LICENSE-DATA and attribution. Model-weight licensing is separate; "
                    "this upload does not relicense weights.")
    (bundle / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(ROOT / "scripts/kaggle_a2_kernel.py", kernel / "train.py")
    (kernel / "kernel-metadata.json").write_text(json.dumps(dict(
        id=a.username + "/" + a.slug, title=a.slug, code_file="train.py", language="python",
        kernel_type="script", is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
        dataset_sources=[a.username + "/" + a.slug], competition_sources=[], kernel_sources=[], model_sources=[]
    ), indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nkaggle datasets create -p {bundle} --keep-tabular")
    print(f"kaggle kernels push -p {kernel} --accelerator NvidiaTeslaT4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
