# geez-morphology

Morphological analysis and generation for Classical Ge'ez verbs: a root-and-pattern rule engine, a hybrid analyzer, a synthetic data generator, and ByT5 fine-tuning. The data lives in the companion repo [geez-dataset](https://github.com/Esubaalew/geez-dataset). The scripts look for it in `--dataset_dir`, else in `$GEEZ_DATASET_DIR`, else in `../geez-dataset` next to this repo.

## Layout

```
geez-morphology/
├── engine.py                     root-and-pattern interdigitation (language-agnostic)
├── geez_script.py                Ge'ez script, vowels, transliteration, to_ethiopic_root()
├── patterns_triliteral.json      triliteral templates
├── patterns_quadriliteral.json   quadriliteral templates
├── verb_class_paradigms.py       checked paradigms: Type B, Type C, Type D, final-w, final-y, initial-w
├── verb_classes.py               class lookup and paradigm rows for generator and analyzer
├── kidane_wold_triples.py        parses Kidane Wold verb triples; rewrites three dataset lexicon files
├── analyzer.py                   HybridAnalyzer: class paradigms -> fast path -> templates -> ByT5 fallback
├── cli.py                        analyze / lemmatize / generate / lookup
├── generate_synthetic_data.py    writes the synthetic training data
├── dataset_loader.py             reads the gold files of a geez-dataset checkout (trainer and evaluation)
├── split_heldout_synthetic.py    moves synthetic lines with a gold val/test root out of the training file
├── train_byt5_morphology.py      ByT5 fine-tuning (resumable)
├── evaluate_gold.py              scores a system on the Merho Ges gold val and test splits
├── data/
│   ├── heldout_test_roots.txt           gold test roots left out of the synthetic data
│   ├── synthetic_roots_allowed_v4.txt   the only roots the generator conjugates
│   └── heldout_gold_forms.tsv           gold val/test forms the generator never writes
├── tests/                        unit and regression tests
└── docs/
    └── grammar_rules.md          the language decisions behind the data, with sources
```

Not in git: `.venv/`, `byt5_morphology_model/`, `checkpoints*/`, logs.

## Setup

Tested with Python 3.13.5.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # pytest, for the tests
pip install -r requirements-train.txt    # torch, transformers, accelerate: training and the ByT5 fallback
```

The rule engine, the analyzer (rules only), the CLI and the generator need only the Python standard library.

### Get the dataset

Clone [geez-dataset](https://github.com/Esubaalew/geez-dataset) next to this repo (the default place), or give its path with `--dataset_dir` or `$GEEZ_DATASET_DIR`:

```bash
git clone https://github.com/Esubaalew/geez-dataset ../geez-dataset
(cd ../geez-dataset && python scripts/check_dataset.py)   # prints the pairs that load
```

The analyzer reads the dictionary `data/geez_amharic_dictionary.jsonl` of the dataset checkout. If that file is not there it stops with `Dictionary not found: ...`; it does not run without a dictionary.

## Use

```bash
python cli.py --dataset_dir ../geez-dataset analyze ይቀትል
python cli.py generate q-t-l --verb_type type_a --tense present
python cli.py --dataset_dir ../geez-dataset lookup ቀተለ
```

`generate` uses the triliteral templates. With `--verb_type` it uses the forms the generator uses for that class: the Type A templates for `type_a` and `hollow`, the quadriliteral templates for `quadriliteral`, and the checked class tables for `type_b`, `type_c`, `type_d`, `final_w`, `final_y` and `initial_w_drop`. `--tense` takes `past` (perfect), `present` (imperfect), `subjunctive` and `imperative`.

## Generate the data

```bash
python generate_synthetic_data.py --dataset_dir ../geez-dataset
```

By default the generator writes into the dataset folder: it writes `<dataset>/output/` (not in git), copies the two synthetic files over the ones in `<dataset>/data/paradigms/`, and moves the lines whose root is a gold val or test root to `<dataset>/data/synthetic/`. To leave the dataset's `data/` folder as it is, add `--no_mirror`; add `--output_dir` to write somewhere other than `<dataset>/output/`:

```bash
python generate_synthetic_data.py --dataset_dir ../geez-dataset --output_dir /tmp/synthetic --no_mirror
```

Line counts are in the dataset README.

The generator leaves out the roots listed in `data/heldout_test_roots.txt`: the 185 roots of the gold test file at dataset commit c7bc587 (`data/paradigms/merho_ges_gold_test.jsonl`, the file as extracted), one per line. The synthetic files in the dataset were generated without these roots. The list is committed here, so the generator does not need the dataset's git history (a shallow or fresh clone works). sha256 of the file: `8a6b64053cddd0d54301acc4142e3aa9e2b25cc88bb6334ce5a05dd061b7c88d` (checked by `tests/test_heldout_roots_file.py`). `--test_file <gold test file>` leaves out the roots of another test file instead.

Two more files in `data/` limit the output, and the generator stops if either is missing:

- `data/synthetic_roots_allowed_v4.txt`: the roots it may conjugate, one per line (1,271 roots). A root is on the list when it has a verb entry in the dictionary, a root in the gold files, or a Kidane Wold verb triple, or when the maintainers restore it. The list is the 2,148 roots of this generator's output on geez-dataset 3da9041 (with the list switched off) that have such evidence, minus the roots that `the exception table` (sha256 `ce8f608b...`) marks DROP or EXCLUDE_NO_VERB. Its eight RESTORE rows (m-h2-l, m-s-l, S-w-t, b-y-S, f-r-S, h2-r-g, m-s-k, w-k-a2) stay on the list. The file header gives the full hashes.
- `data/heldout_gold_forms.tsv`: every gold val and test form at dataset commit 6183c24 (2,939 rows, 2,832 distinct forms, rows with no root included), keyed by split, record_id, row and spelling. A synthetic line whose form equals one of them is not written; the root's other lines stay as they were. The run log lists the dropped lines by root (`BLOCKED_GOLD_FORMS_JSON`).

`python split_heldout_synthetic.py --full_dir <generator output> --dataset_dir <dataset> --check` compares the split with the dataset files and fails when any synthetic line in the dataset has a gold val or test form, computed from the current gold files.

## Test

```bash
python -m pytest tests/ --dataset_dir /path/to/geez-dataset
```

Without `--dataset_dir` the tests use `$GEEZ_DATASET_DIR`, else `../geez-dataset`. Without a dataset checkout the tests that need one are skipped. `tests/test_dataset_checkout.py` checks that the loader gives the same pair counts as the dataset's `scripts/check_dataset.py`, that no flagged row loads, that a root is in one gold split only, and that no synthetic training line has a gold val or test root.

## Rebuild the Kidane Wold triples

```bash
python kidane_wold_triples.py --dataset_dir ../geez-dataset
```

This reads `data/lexicon/classical_dictionary_kidane_wold.jsonl` of the dataset and rewrites three files next to it: `kidane_wold_verb_triples.jsonl`, `kidane_wold_verb_triples_rejected.jsonl` and `kidane_wold_root_fixes.jsonl`. Rows of the current root fixes file with `new_root` null or with `root_notes` are carried over. The generator and the analyzer read these files, so run it only when the dictionary changes.

## Train and evaluate

```bash
python train_byt5_morphology.py --dataset_dir ../geez-dataset --load_only     # counts only, no training
python train_byt5_morphology.py --dataset_dir ../geez-dataset                 # train
python evaluate_gold.py --dataset_dir ../geez-dataset --system byt5 \
    --out_json gold_eval_byt5.json --predictions_out gold_eval_byt5_predictions.jsonl
```

What the trainer reads, as written (nothing is corrected at load time):

- `data/paradigms/synthetic_morphology_train.jsonl`, all lines. 10% of them (seed 42) are the eval split for the loss; gold rows never go there.
- `data/paradigms/merho_ges_gold_train.jsonl`, through `dataset_loader.load_gold_split`: one pair per form of each table row. Rows listed in `data/paradigms/merho_ges_gold_flagged_rows.tsv` and rows with no root do not load. A malformed root or a form that is not one Ge'ez word stops the run; `check_dataset.py` would report it too. On dataset commit 0d0ceaa this gives 8,237 train pairs (val 1,173, test 1,558).
- The files under `data/synthetic/` are never read.

Training: ByT5-small, fp32, lr 3e-4, 3% warmup, 3 epochs, batch 16. Device: cuda, then mps, then cpu. Everything goes to `--output_dir` (default `byt5_morphology_model/`):

```
byt5_morphology_model/
├── checkpoint-<step>/     saved every 1000 steps, the last two kept; a rerun resumes from the latest
├── run_info.json          git commits of both repos, data sha256, pair counts, arguments
└── config.json, model.safetensors, tokenizer files
                           the final model, written when training finishes
```

On SIGTERM or Ctrl-C the trainer saves a checkpoint and stops; the same command resumes. `--resume_from` points at another run's checkpoints, and `--batch_size 8 --grad_accum 2` keeps the effective batch at 16. A dry run on CPU: `python train_byt5_morphology.py --dataset_dir ../geez-dataset --device cpu --max_steps 3 --sample_limit 64 --max_eval_samples 8 --batch_size 4 --output_dir /tmp/byt5_dry_run`.

`evaluate_gold.py` scores every pair of the gold val and test files (`--split val test`), read by the same loader. It loads the final model from `--model_dir` (default `byt5_morphology_model/`), or the latest `checkpoint-<step>` when the run did not finish. `HybridAnalyzer` (and `--system hybrid`) finds its ByT5 fallback the same way; if there is no model there it stops with `ByT5 model not found in ...`. `--no_neural` (`use_neural=False`) runs the rules only and needs no model. It ends with the root accuracy per split:

```
Root accuracy (exact match):
  val    43.1% (505/1173)
  test   39.9% (622/1558)
```

A root counts as correct only when it equals the gold root letter for letter (`ቅ-ት-ል` for `ቀ-ተ-ለ` is wrong). The numbers above are the rules-only analyzer (`--system hybrid --no_neural`) on dataset 0d0ceaa; it does not predict nouns and adjectives, so their type and exact match are always wrong. The full table also gives type, stem, tense, person, gender, number, object and exact match, by type, verb class, tense and tier. `--predictions_out` writes one line per pair with the gold labels, the prediction and the fields it got right.

## Rules

Language decisions and their sources: [`docs/grammar_rules.md`](docs/grammar_rules.md).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0; see [`LICENSE`](LICENSE). The geez-dataset repo is licensed separately (its data files under CC BY-SA 4.0).
