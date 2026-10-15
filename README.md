# geez-morphology

Morphological analysis, root extraction, paradigm generation, and ByT5 sequence-to-sequence fine-tuning for Classical Ge'ez (ግዕዝ).

The companion dataset lives in [geez-dataset](https://github.com/Esubaalew/geez-dataset). All scripts locate the dataset via `--dataset_dir`, else `$GEEZ_DATASET_DIR`, else `../geez-dataset`.

## Architecture and Repository Layout

```
geez-morphology/
├── engine.py                     Root-and-pattern interdigitation engine
├── geez_script.py                Ethiopic script tables, vowel orders, transliteration, spelling folding
├── patterns_triliteral.json      Triliteral verbal templates
├── patterns_quadriliteral.json   Quadriliteral verbal templates
├── verb_class_paradigms.py       Checked paradigms: Type B, Type C, Type D, final-w, final-y, initial-w
├── verb_classes.py               Verb class lookup and paradigm rows for generator and analyzer
├── kidane_wold_triples.py        Extracts perfect / imperfect / subjunctive verb triples from Kidane Wold
├── analyzer.py                   HybridAnalyzer: checked paradigms -> affix/stem rules -> templates -> ByT5
├── cli.py                        CLI for analyze, lemmatize, generate, and lookup
├── dataset_loader.py             Loads gold, unified, dictionary, and rootless datasets into training/eval pairs
├── generate_synthetic_data.py    Generates synthetic verbal paradigms for attested roots
├── split_heldout_synthetic.py    Partitions synthetic paradigms into train vs. held-out val/test splits
├── train_byt5_morphology.py      ByT5 fine-tuning and checkpoint selection
├── evaluate_gold.py              Evaluates rules, neural, or hybrid analyzers on gold and unified splits
├── predict_roots.py              Batch root prediction (word<TAB>root)
├── scripts/
│   ├── audit_train_leakage.py    Verifies zero root or pair overlap between training pairs and val/test
│   └── summarize_runs.py         Aggregates multi-seed metrics (mean, SD, 95% Wilson CI, paired McNemar test)
├── data/
│   ├── synthetic_roots_allowed_v5.txt   Attested verb roots conjugated by the synthetic generator (1,234 roots)
│   ├── heldout_test_roots.txt           Gold test roots excluded during generator runs (185 roots)
│   ├── heldout_gold_forms.tsv           Gold val/test surface forms blocked from synthetic training (2,907 entries)
│   └── closed_class_words_without_root.tsv   Closed-class particles, pronouns, names, and loanwords (`root: none`)
├── tests/                        Unit and dataset-checkout test suite
└── docs/
    └── grammar_rules.md          Grammatical rules and primary sources
```

## Setup

Tested with Python 3.13. The rule engine, rules-only analyzer, CLI, and synthetic generator use only the Python standard library.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # pytest
pip install -r requirements-train.txt    # torch, transformers, accelerate (for ByT5 training and neural inference)
```

Clone `geez-dataset` next to this repository (or set `$GEEZ_DATASET_DIR` / `--dataset_dir`):

```bash
git clone https://github.com/Esubaalew/geez-dataset ../geez-dataset
(cd ../geez-dataset && python3 scripts/check_dataset.py && python3 scripts/check_unified.py .)
```

## Command-Line Interface

```bash
python3 cli.py --dataset_dir ../geez-dataset analyze ይቀትል
python3 cli.py --dataset_dir ../geez-dataset lemmatize ወተቀበሎሙ
python3 cli.py generate q-t-l --verb_type type_a --tense present
python3 cli.py --dataset_dir ../geez-dataset lookup ቀተለ
```

`generate` conjugates triliteral and quadriliteral roots across `--verb_type` (`type_a`, `type_b`, `type_c`, `type_d`, `hollow`, `final_w`, `final_y`, `initial_w_drop`, `quadriliteral`) and `--tense` (`past`, `present`, `subjunctive`, `imperative`).

## Input and Target Representation

Each sample has a task (`analyze` for full morphological analysis, `root_only` for root extraction alone) that determines its input prefix and target string (`dataset_loader.input_text`, `dataset_loader.target_text`; `--single_prefix` uses `analyze: ` for all tasks):

| Word Class / Task | Input | Target Format |
|---|---|---|
| `verb` | `analyze: ይቀትል` | `root: ቀ-ተ-ለ \| tense: present \| person: 3 \| gender: m \| number: s \| stem: basic` (`+ \| object: 3ms` / `\| negated: yes`) |
| `noun` | `analyze: ቅትለት` | `root: ቀ-ተ-ለ \| type: noun \| number: s` (`+ \| possessor: 3ms`) |
| `adjective` | `analyze: ቅቱላን` | `root: ቀ-ተ-ለ \| type: adjective \| gender: m \| number: p` (`+ \| possessor: 3ms` / `\| negated: yes`) |
| `infinitive`, `gerund`, `verbal_noun`, `nominal` | `analyze: ቀቲሎ` | `root: ቀ-ተ-ለ \| type: gerund \| person: 3 \| gender: m \| number: s \| stem: basic` (fields in order: `tense`, `person`, `gender`, `number`, `stem`, `object`, `possessor`, `negated`) |
| `pronoun`, `demonstrative`, `preposition`, `particle`, `numeral`, `adverb` | `analyze: ውእቱ` | `root: none \| type: pronoun \| person: 3 \| gender: m \| number: s` |
| `root_only` (dictionary nominal or rootless word) | `root: መቅደስ` | `root: ቀ-ደ-ሰ` (`root: none` when rootless) |

`analyzer.parse_neural_output` parses all target formats back into structured feature dictionaries and rejects malformed outputs (such as a rootless class with a root or a rooted class with `none`).

## Training Data Mixture and Split Hygiene

By default, `train_byt5_morphology.py` builds **156,541** training pairs and two loss-evaluation slices (`12,122` synthetic pairs and `74` dictionary nominal pairs) from `--dataset_dir`:

| Source | Raw Loaded | Train Pairs | Eval Slice | Notes |
|---|---:|---:|---:|---|
| Synthetic verbs (`synthetic_morphology_train.jsonl`) | 121,227 | 109,101 | 12,122 | 90/10 split (seed 42); 4 lines matching held-out unified book forms removed |
| Gold verbs (`merho_ges_gold_train.jsonl`) | 1,768 | 7,072 | — | Upsampled $\times 4$ (`--gold_upsample 4`) |
| Gold nouns & adjectives (`merho_ges_gold_train.jsonl`) | 6,306 | 31,530 | — | Upsampled $\times 5$ (`--gold_nominal_upsample 5`) |
| Unified book rows (`geez_morphology_rows.jsonl`, `split="train"`) | 8,119 | 8,119 | — | All 13 word classes from the 6 non-gold books (`--no_unified` disables) |
| Kidane Wold dictionary nominals (`classical_dictionary_kidane_wold.jsonl`) | 1,084 | 687 | 74 | 90/10 root-disjoint split; forms already in gold or unified rows removed (`--no_dictionary_pairs` disables) |
| Closed-class rootless words (`closed_class_words_without_root.tsv`) | 57 | 32 | — | Words not already in gold or unified rows (`--no_words_without_root` disables) |
| **Total** | — | **156,541** | **12,196** | Verified leak-free against `val` and `test` by `scripts/audit_train_leakage.py` |

### Leak-Prevention Guarantees

1. **Root-disjoint splits**: No root in `merho_ges_gold_{val,test}.jsonl` or `data/unified/book_root_splits.tsv` (`val`/`test`) appears in synthetic training (`synthetic_morphology_train.jsonl`), gold training, unified book training, or dictionary training.
2. **Surface-form deduplication and blocking**:
   - Every gold `val`/`test` surface form (`data/heldout_gold_forms.tsv`: `2,907` rows, `2,853` distinct forms) and every near-duplicate differing only by labiovelar or homophone letters (`geez_script.fold_spelling`) is excluded from `synthetic_morphology_train.jsonl`, dictionary pairs, and rootless words.
   - When training with unified book rows (the default), any dictionary pair or closed-class rootless word whose surface form is already present in `data/unified/geez_morphology_rows.jsonl` is dropped so full-analysis unified targets are never duplicated by root-only targets, and any synthetic pair matching a held-out unified `val`/`test` form is removed.

## Training and Evaluation

### 1. Verify Data Loading and Leakage Audit

```bash
python3 train_byt5_morphology.py --dataset_dir ../geez-dataset --load_only --export_dir /tmp/exp
python3 scripts/audit_train_leakage.py --dataset_dir ../geez-dataset --train /tmp/exp/train_pairs.jsonl
```

### 2. Train ByT5 (`google/byt5-small`)

Hyperparameters: `fp32`, learning rate `3e-4`, `3%` linear warmup, `3` epochs, effective batch size `16` (`--batch_size 8 --grad_accum 2` on smaller GPUs). Checkpoints are saved every `1,000` steps to `--output_dir` (default `byt5_morphology_model/`) and automatically resumed on restart.

```bash
python3 train_byt5_morphology.py --dataset_dir ../geez-dataset \
    --root_accuracy_rows 1000 --metric_for_best_model root_accuracy \
    --seed 1 --output_dir runs/books_seed1
```

For the core-only ablation (gold + synthetic + dictionary + rootless words, without book rows):

```bash
python3 train_byt5_morphology.py --dataset_dir ../geez-dataset \
    --no_unified --no_book_forms \
    --root_accuracy_rows 1000 --metric_for_best_model root_accuracy \
    --seed 1 --output_dir runs/core_seed1
```

Both arms select checkpoints on the same dev set (`gold val` + `unified book val`; pass `--dev_gold_only` for gold `val` only).

### 3. Evaluate on Gold and Unified Splits

```bash
# Evaluate trained ByT5 model on gold val and test
python3 evaluate_gold.py --dataset_dir ../geez-dataset --system byt5 \
    --model_dir runs/books_seed1 --split val test \
    --out_json runs/books_seed1/eval_gold.json \
    --predictions_out runs/books_seed1/preds_gold.jsonl

# Evaluate on held-out unified book test split
python3 evaluate_gold.py --dataset_dir ../geez-dataset --system byt5 \
    --model_dir runs/books_seed1 --split unified_test \
    --out_json runs/books_seed1/eval_unified_test.json

# Evaluate rules-only analyzer (no neural model required)
python3 evaluate_gold.py --dataset_dir ../geez-dataset --system hybrid --no_neural --split val test
```

Rules-only baseline root accuracy (`--system hybrid --no_neural`, which covers verbs only):
- `val`: **50.9%** (`592 / 1,162`, 95% CI `48.1–53.8`)
- `test`: **45.9%** (`708 / 1,541`, 95% CI `43.5–48.4`)

`evaluate_gold.py` reports exact-match accuracy for `root`, `type`, `stem`, `tense`, `person`, `gender`, `number`, `object`, `possessor`, `negated`, and full-row `exact`, broken down by word class, verb class, tense, source book, and `rootvar` (`disjoint` vs. `variant_in_train` under homophone/labiovelar folding), along with the single-prediction exact-match ceiling on ambiguous surface forms.

### 4. Summarize Multi-Seed Runs and Regenerate Synthetic Data

```bash
# Aggregate mean, SD, and 95% Wilson CI across seeds, plus paired McNemar test between arms
python3 scripts/summarize_runs.py runs/books_seed1 runs/books_seed2 runs/books_seed3
python3 scripts/summarize_runs.py --compare runs/books_seed* -- runs/core_seed*

# Regenerate synthetic paradigms and verify split hygiene
python3 generate_synthetic_data.py --dataset_dir ../geez-dataset
python3 split_heldout_synthetic.py --full_dir ../geez-dataset/output --dataset_dir ../geez-dataset --check
```

## Tests

```bash
GEEZ_DATASET_DIR=../geez-dataset python3 -m pytest tests/ --dataset_dir ../geez-dataset
```

## License and Citation

Code in this repository is licensed under the [Apache License 2.0](LICENSE). Citation metadata: [CITATION.cff](CITATION.cff).


## Training on a TPU (train_tpu.py)

`train_tpu.py` trains the same model with PyTorch/XLA, one process per TPU core (data parallel), bf16 autocast,
Adafactor at a fixed learning rate, and a few static (input, target) byte lengths so XLA compiles each shape once.
Its training mixture (`training_mixture.py`) reads every source of the dataset with one target format and one leak
gate, and trains three tasks on the same pairs: `analyze: <form>` (the full analysis, `root: none | type: ...` for a
word without root), `generate: <analysis>` (the form) and `pos: <form>` (`type: <class>`). Checkpoints are selected on
the dev set (gold val plus the unified book val rows) by root and exact-match accuracy.

    python3 train_tpu.py --dataset_dir ../geez-dataset --output_dir out --load_only        # counts + leak audit
    python3 build_dapt_corpus.py --dataset_dir ../geez-dataset --out dapt_corpus.txt
    python3 train_tpu.py --dataset_dir ../geez-dataset --output_dir out/dapt --stage dapt
    python3 train_tpu.py --dataset_dir ../geez-dataset --output_dir out/ft --init_from out/dapt/best
    python3 train_tpu.py --dataset_dir ../geez-dataset --output_dir /tmp/dry --device cpu \
        --train_limit 400 --dev_cap 24 --report_cap 12 --per_core_batch 8 --max_steps 6 --eval_every 3   # dry run

Off by default: the extra synthetic families of `extra_synthetic.py` (`--extra_synthetic derived_c,hollow,...`) and
the 2-radical dictionary roots (`--biradical_dictionary`); `python3 extra_synthetic.py --table t.tsv` and
`scripts/export_biradical_roots.py` write their tables for checking.
