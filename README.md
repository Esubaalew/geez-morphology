# geez-morphology

Morphological analysis, root extraction and paradigm generation for Classical Ge'ez (ግዕዝ). A rule engine generates and analyses verb forms; a ByT5 model (`google/byt5-small`) is fine-tuned to return the root and features of a word, and serves as fallback in the hybrid analyzer.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # tests
pip install -r requirements-train.txt    # torch, transformers, accelerate (training and neural inference)
```

The rule engine, rules-only analyzer, CLI and generator use only the Python standard library.

## Data

The code reads [geez-dataset](https://github.com/Esubaalew/geez-dataset), found via `--dataset_dir`, else `$GEEZ_DATASET_DIR`, else the first existing sibling `../geez-dataset` or `../geez-dataset-v1`. If neither sibling exists, scripts exit and name both tried paths:

```bash
git clone https://github.com/Esubaalew/geez-dataset ../geez-dataset
```

It uses the gold files in `data/paradigms/`, the synthetic training forms, the unified book rows in `data/unified/`, the Kidane Wold dictionary, and `data/closed_class_words_without_root.tsv` from this repository. Only rows with an empty `status` are loaded. `data/heldout_gold_forms.tsv` and `data/synthetic_roots_allowed_v5.txt` control which forms and roots the synthetic generator may produce.

Input and target example:

```
analyze: ይቀትል  ->  root: ቀ-ተ-ለ | tense: present | person: 3 | gender: m | number: s | stem: basic
```

## Command line

```bash
python3 cli.py --dataset_dir ../geez-dataset analyze ይቀትል
python3 cli.py --dataset_dir ../geez-dataset lemmatize ወተቀበሎሙ
python3 cli.py generate q-t-l --verb_type type_a --tense present
python3 cli.py --dataset_dir ../geez-dataset lookup ቀተለ
```

## Train

```bash
python3 train_byt5_morphology.py --dataset_dir ../geez-dataset --load_only --export_dir /tmp/exp   # counts only
python3 scripts/audit_train_leakage.py --dataset_dir ../geez-dataset --train /tmp/exp/train_pairs.jsonl
python3 train_byt5_morphology.py --dataset_dir ../geez-dataset --output_dir runs/seed1 --seed 1
```

Main flags (defaults in brackets):

| Flag | Meaning |
|---|---|
| `--output_dir` | checkpoint directory [`byt5_morphology_model/`] |
| `--epochs` | training epochs [3] |
| `--lr` | peak AdamW learning rate [3e-4] |
| `--warmup_ratio` | linear warmup fraction [0.03] |
| `--batch_size`, `--grad_accum` | per-device batch [16], accumulation [1] |
| `--save_steps` | checkpoint interval [1000] |
| `--seed` | random seed [42] |
| `--gold_upsample`, `--gold_nominal_upsample` | gold verb [4] and noun/adjective [5] upsampling |
| `--no_unified`, `--no_book_forms` | leave out the unified book rows |
| `--no_dictionary_pairs`, `--no_words_without_root` | leave out dictionary and rootless pairs |
| `--metric_for_best_model`, `--root_accuracy_rows` | checkpoint selection |
| `--load_only` | load data, print counts, exit |

`train_tpu.py` trains the same model with PyTorch/XLA on TPU; run it with `--load_only` first to see counts.

## Evaluate

```bash
python3 evaluate_gold.py --dataset_dir ../geez-dataset --system byt5 --model_dir runs/seed1 \
    --split val test --out_json runs/seed1/eval.json
python3 evaluate_gold.py --dataset_dir ../geez-dataset --system hybrid --no_neural --split val test
```

`--split` takes `val`, `test`, `books`, `unified_val`, `unified_test`. The script reports exact-match accuracy for root, each feature and the full row.

## Tests

```bash
python3 -m pytest tests/ --dataset_dir ../geez-dataset
```

## License

Apache License 2.0 ([LICENSE](LICENSE)). Citation: [CITATION.cff](CITATION.cff).
