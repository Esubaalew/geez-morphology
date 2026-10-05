# Ge'ez Computational Morphology Engine & Neural ByT5 Analyzer

A modular, high-performance root-and-pattern interdigitation engine, inverse morphological analyzer, and neural ByT5 sequence-to-sequence model configured for Classical Ge'ez.

```
       Consonantal Root                    Template Specification
      ("q", "t", "l")                      "C0a-C1a-C2a"
             │                                   │
             └───────────────┬───────────────────┘
                             ▼
                 [ Interdigitation Engine ]
                             │
                             ▼
                   Abstract Realization
             [("q", "a"), ("t", "a"), ("l", "a")]
                             │
                             ▼
                 [ Glide Contraction Pass ]
                  (aw -> o, ay -> e, etc.)
                             │
                             ▼
                    [ Ge'ez Renderer ]
             (Ethiopic Unicode arithmetic)
                             │
                             ▼
                    Surface Form: ቀተለ
```

---

## Features

* **Decoupled Architecture:** Separates the universal Semitic root-and-pattern mechanism (`engine.py`) from language-specific parameters (consonant/vowel inventories, orthographic rendering, and verb classes in `geez_config.py`).
* **Classical Glide Contraction Pass:** Automatically resolves weak hollow verbs ($aw \to o, ay \to e, əw \to u$) so roots like `q-w-m` generate **`ቆመ`**, **`ይቁም`**, and `m-w-t` generates **`ሞተ`**, **`ይሙት`**.
* **Unified Ethiopic Script Targets:** Standardizes all root representations in native Ethiopic script (`ቀ-ተ-ለ`, `በ-ረ-ከ`, `ሰ-ሠ-ነ`), eliminating transliteration ambiguities.
* **Bidirectional Processing:** Supports both deterministic surface generation (`engine.py`, `cli.py generate`) and inverse morphological analysis and text lemmatization (`analyzer.py`, `cli.py analyze`).
* **Ethiopic Unicode Geometry:** Leverages arithmetic offsets in the Unicode Ethiopic block (`U+1200–U+137F`) for $O(1)$ syllable synthesis without heavy mapping dictionaries.
* **Apple Silicon Hardware Acceleration:** Dedicated training pipeline (`train_mac.py`) utilizing PyTorch Metal Performance Shaders (MPS) on Apple Silicon Macs.
* **Gold Benchmark Evaluator:** Standalone evaluation suite (`evaluate_gold.py`) measuring exact-match accuracy against a 15% held-out test split of authentic church paradigms from *Merho Ges*.

---

## Repository Structure

| File | Purpose |
| :--- | :--- |
| `engine.py` | Universal Semitic root-and-pattern interdigitation core with phonological glide contractions. |
| `analyzer.py` | Inverse morphological analyzer, text lemmatizer, and lexicon lookup engine. |
| `cli.py` | Unified command-line interface (`analyze`, `generate`, `lemmatize`, `lookup`). |
| `pattern_learner.py` | Inductive template learner extracting pattern specs from surface pairs. |
| `train_mac.py` | Optimized ByT5 fine-tuning pipeline for Apple Silicon (MPS) with uniform sampling. |
| `train_byt5.py` | CUDA/Colab ByT5 sequence-to-sequence training pipeline with unified Ethiopic roots. |
| `evaluate_gold.py` | Formal benchmark evaluator against the held-out *Merho Ges* gold test set. |
| `geez_config.py` | Ge'ez phonetic inventory, order mappings, and Ethiopic character renderer. |
| `geez_config.json` | Declarative configuration export of Ge'ez vocalic and consonantal tiers. |
| `patterns.json` | **103 verified morphological patterns** (basic, passive, causative, and intensive stems). |
| `EVALUATION.md` | Quantitative benchmark results, consistency checks, and error analyses. |
| `TECHNICAL_REPORT.md` | Complete architectural specification of the hybrid rule/neural system. |

---

## Quickstart

### Prerequisites
Python 3.8+ (Core engine and CLI run purely on Python standard library).

```bash
git clone https://github.com/esubaalew/geez-morphology-model.git
cd geez-morphology-model
```

### 1. Command-Line Interface (CLI)

```bash
# Analyze an inflected surface form
python3 cli.py analyze ይቀትሉ

# Lemmatize a Ge'ez sentence
python3 cli.py lemmatize "ወይቀትል ፡ ሰብአ"

# Generate forms from a consonantal root with verb_type filtering
python3 cli.py generate q-t-l --verb_type strong_A --tense past --person 3 --number s --gender m

# Look up lemma in Kidane Wold Kifle lexicon
python3 cli.py lookup ቀተለ
```

### 2. Neural ByT5 Training on Apple Silicon (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets accelerate

# Train ByT5 on 20k uniform samples + Merho Ges gold training pairs:
python3 train_mac.py --sample_limit 20000 --batch_size 16 --epochs 3

# Evaluate against the 15% held-out gold test set:
python3 evaluate_gold.py --limit 100
```

---

## Benchmarks & Evaluation

* **Pattern Consistency:** 100% across all 103 compiled patterns.
* **Training Loss:** ByT5 converged to `eval_loss: 0.00821` (`train_loss: 0.02013`) after 3 epochs on the unified Ethiopic dataset.
* **Morphological Agreement:** Verified on unseen *Merho Ges* church test verbs:
  * Number Accuracy: **89.0%**
  * Gender Accuracy: **82.0%**
  * Person Accuracy: **75.0%**
  * Stem Classification: **61.0%**

---

## License

Apache License 2.0. Developed for Classical Ge'ez NLP and Semitic Computational Linguistics research.
