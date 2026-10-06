# geez-morphology

Morphological analysis and generation for Classical Ge'ez verbs: a root-and-pattern rule engine, a hybrid analyzer, a synthetic data generator, and ByT5 fine-tuning. The data lives in the companion repo **geez-dataset**, checked out next to this one as `../geez-dataset-v1`.

## Layout

```
geez-morphology/
├── engine.py                     root-and-pattern interdigitation (language-agnostic)
├── pattern_learner.py            learns templates from (root, form) data
├── geez_config.py                Ge'ez script, vowels, transliteration, to_ethiopic_root()
├── patterns.json                 triliteral templates
├── patterns_quadriliteral.json   quadriliteral templates
├── patterns_rejected.json        templates removed in review (kept for reference)
├── verb_class_paradigms.py       checked paradigms: Type B, Type C, final-weak, initial-weak
├── verb_classes.py               class lookup and paradigm rows for generator and analyzer
├── kw_verb_triples.py            parses Kidane Wold perfect/imperfect/subjunctive triples
├── analyzer.py                   HybridAnalyzer: fast path -> pattern matcher -> ByT5 fallback
├── cli.py                        analyze / lemmatize / generate / lookup
├── generate_synthetic_fixed.py   writes the synthetic training data
├── train_byt5_morphology.py      ByT5 fine-tuning (resumable)
├── evaluate_gold.py              scores a system on the Merho Ges gold test set
├── tests/                        unit and regression tests
└── docs/
    ├── grammar_rules.md          the language decisions behind the data, with sources
    └── audit/                    notes (data_and_model_notes_2026-10-05.md),
                                  checks/, eval/, logs/, patches/,
                                  gold-label suspects and relabel proposals
```

Not in git: `.venv/`, `checkpoints*/`, logs, `.env`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers accelerate pytest
```

The rule engine, analyzer (rules only), CLI and generator need only the Python standard library; torch and transformers are for training and the ByT5 fallback, pytest for the tests.

## Use

```bash
python cli.py analyze ይቀትል
python cli.py generate q-t-l --verb_type strong_A --tense present
python cli.py lookup ቀተለ
```

## Generate the data

```bash
python generate_synthetic_fixed.py
```

Writes `../geez-dataset-v1/output/` (not in git) and copies the synthetic files to `../geez-dataset-v1/data/paradigms/`. Current output: 216,977 pairs from 2,007 roots (full train file sha256 `e384ccf2dc276c95f2c65fde564a40eb31388ecb2c84a72faa34d050139abcb7`, guttural-final past rule, Merho Ges class tags from the corrected gold train file, per-root random streams); after the held-out split (`synthetic_heldout.py`) the training file has 210,348 lines (sha256 `f9b52cd4…`) and 6,629 lines are held out in `data/synthetic/`. That file was generated with the held-out roots of the gold test file as of dataset commit 544271a (`generate_extended_dataset(..., test_file=<544271a merho_ges_gold_test.jsonl>)`): the corrected test file holds out other roots, so the plain command above would change the lines of those roots (every random draw is per root, so no other root's lines change). Class and tense counts are in the dataset README.

## Test

```bash
python -m pytest tests/
```

## Train

```bash
python train_byt5_morphology.py
```

- Reads `data/paradigms/synthetic_morphology_train.jsonl` and the Merho Ges gold-train tables (3,322 pairs). All gold pairs go to training; eval is 10% of the synthetic pairs (seed 42).
- ByT5-small, fp32, lr 3e-4, 3% warmup, 3 epochs, batch 16. Device: cuda, then mps, then cpu.
- Saves a checkpoint every 1000 steps (keeps 2) in `--output_dir` (default `checkpoints_full/`), resumes from the latest one automatically, and saves before exiting on SIGTERM. `run_info.json` records the git commits, data sha256 and arguments.
- Other machines: `--dataset_dir /path/to/geez-dataset --output_dir /path/out`; `--resume_from` points at an earlier run's checkpoints; `--batch_size 8 --grad_accum 2` keeps the effective batch at 16.
- Dry run: `python train_byt5_morphology.py --max_steps 5 --sample_limit 200 --max_eval_samples 32 --output_dir /tmp/dry_run`

## Evaluate

```bash
python evaluate_gold.py --system hybrid --no_neural                      # rules only
python evaluate_gold.py --system byt5 --model_dir checkpoints_full       # trained model
```

Current rules-only result on the Merho Ges test split: exact match 193/377 (51.2%), root 248/394. With the proposed gold relabels excluded (`--exclude_proposed`): 193/360. Model results will follow after training.

## Rules and notes

- Language decisions and their sources: [`docs/grammar_rules.md`](docs/grammar_rules.md)
- The 2026-10-05 notes, sheets, patches and logs (formerly `notes/`) are not included since 2026-10-06; the relabel proposals the code reads are in `../geez-dataset-v1/data/corrections/`.

## License

Apache License 2.0.
