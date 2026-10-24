#!/usr/bin/env python3
"""Rebuild a checkpoint-lineage exposure manifest from its exact historical Git snapshots.

Uses only local Git objects and standard-library Python. Executes the archived historical
mixture builders, verifies each source/task/class count, then checks the union of forms/roots.
The historical code and manifest must be trusted. No checkpoint weights or credentials are read.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import dataset_loader  # noqa: E402


WORKER = """
import json, os, sys
model, data, config_path, out = sys.argv[1:]
sys.path.insert(0, model)
import dataset_loader as dl, training_mixture as tm, train_tpu as tt
config = json.load(open(config_path, encoding='utf-8'))
options = list(config['data_options'])
if '--extra_rootless' in options:
    index = options.index('--extra_rootless') + 1
    options[index] = os.path.join(data, options[index])
args = tt.parse_args(['--dataset_dir', data, '--output_dir', out] + options)
if getattr(args, 'unified_file', None):
    dl.UNIFIED_FILE = args.unified_file
train, _, report = tm.build(data, args)
for key, expected in config['expected_report'].items():
    if report[key] != expected:
        raise ValueError('Historical mixture mismatch: ' + config['name'] + '/' + key)
result = dict(forms=sorted({s['form'] for s in train}),
              roots=sorted({s['root'] for s in train if s.get('root') and s['root'] != dl.NO_ROOT}))
with open(out, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)
print(config['name'], 'verified samples:', len(train), flush=True)
"""


def snapshot(repo, revision, destination):
    destination.mkdir()
    archive = destination.parent / (destination.name + ".tar")
    with archive.open("wb") as f:
        subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", revision], stdout=f, check=True)
    with tarfile.open(archive) as f:
        f.extractall(destination, filter="data")
    archive.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=HERE / "data/a2_training_exposure.json")
    p.add_argument("--model_repo", type=Path, default=HERE)
    p.add_argument("--dataset_repo", type=Path, default=Path(dataset_loader.default_dataset_dir()))
    args = p.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    forms, roots = set(), set()
    with tempfile.TemporaryDirectory(prefix="geez-exposure-") as folder:
        root = Path(folder)
        for index, config in enumerate(manifest["source_runs"]):
            run = root / str(index)
            run.mkdir()
            model, data = run / "model", run / "dataset"
            snapshot(args.model_repo, config["model_commit"], model)
            snapshot(args.dataset_repo, config["data_commit"], data)
            options, out = run / "config.json", run / "exposure.json"
            options.write_text(json.dumps(config), encoding="utf-8")
            subprocess.run([sys.executable, "-c", WORKER, str(model), str(data), str(options), str(out)], check=True)
            rebuilt = json.loads(out.read_text(encoding="utf-8"))
            forms.update(rebuilt["forms"])
            roots.update(rebuilt["roots"])
    if sorted(forms) != manifest["forms"] or sorted(roots) != manifest["roots"]:
        raise SystemExit("FAIL: saved exposure forms/roots differ from historical mixtures")
    print(f"PASS: {len(forms)} forms / {len(roots)} roots match the complete historical lineage mixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
