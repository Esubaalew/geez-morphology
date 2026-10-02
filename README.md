# Ge'ez Computational Morphology Engine

A modular, high-performance root-and-pattern interdigitation engine, morphological analyzer, and pattern induction learner for Semitic languages, configured for Classical Ge'ez.

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
                    [ Ge'ez Renderer ]
             (Ethiopic Unicode arithmetic)
                             │
                             ▼
                    Surface Form: ቀተለ
```

---

## Features

* **Decoupled Architecture:** Separates the universal Semitic root-and-pattern mechanism from language-specific parameters (consonant/vowel inventories, orthographic rendering, and verb classes).
* **Bidirectional Processing:** Supports both deterministic surface generation (`engine.py`) and inverse morphological analysis and text lemmatization (`analyzer.py`).
* **Ethiopic Unicode Geometry:** Leverages arithmetic offsets in the Unicode Ethiopic block (`U+1200–U+137F`) for $O(1)$ syllable synthesis without heavy mapping dictionaries.
* **Unsupervised Pattern Induction:** Discovers morphological templates directly from annotated paradigm tables using an order-preserving alignment algorithm (`pattern_learner.py`).
* **Integrated Lexicon Grounding:** Seamlessly links morphological analyses to definitions, citations, and part-of-speech data in `geez-dataset-v1`.
* **ByT5 Neural Pipeline:** Includes a byte-level sequence-to-sequence fine-tuning pipeline (`train_byt5.py`) avoiding subword BPE tokenization traps.

---

## Repository Structure

| File | Purpose |
| :--- | :--- |
| `engine.py` | Universal, language-agnostic root-and-pattern interdigitation core. |
| `analyzer.py` | Inverse morphological analyzer, text lemmatizer, and lexicon lookup engine. |
| `cli.py` | Unified command-line interface (`analyze`, `generate`, `lemmatize`, `lookup`). |
| `pattern_learner.py` | Inductive template learner extracting pattern specs from surface pairs. |
| `train_byt5.py` | ByT5 character/byte-level sequence-to-sequence fine-tuning pipeline. |
| `geez_config.py` | Ge'ez phonetic inventory, order mappings, and Ethiopic character renderer. |
| `geez_config.json` | Declarative configuration export of Ge'ez vocalic and consonantal tiers. |
| `patterns.json` | 64 verified morphological patterns induced from classical grammar matrices. |
| `EVALUATION.md` | Quantitative benchmark results, consistency checks, and class generalization tests. |
| `COMBINATORIAL_ANALYSIS.md` | Mathematical derivation of the ~4.98 million form inflectional capacity. |
| `GENERALIZATION.md` | Cross-linguistic architectural comparison across Semitic languages. |

---

## Quickstart

### Prerequisites
Python 3.8+ (Core engine and CLI run purely on Python standard library).

```bash
git clone https://github.com/esubaalew/geez-morphology.git
cd geez-morphology
```

### 1. Command-Line Interface (CLI)

```bash
# Analyze an inflected surface form
python3 cli.py analyze ይቀትሉ

# Lemmatize a Ge'ez sentence
python3 cli.py lemmatize "ወይቀትል ፡ ሰብአ"

# Generate forms from a consonantal root
python3 cli.py generate q-t-l --tense present --person 3 --number p --gender m

# Look up lemma in Kidane Wold Kifle lexicon
python3 cli.py lookup ቀተለ
```

### 2. Python API Usage

```python
from analyzer import GeezMorphologicalAnalyzer

analyzer = GeezMorphologicalAnalyzer()

# Morphological analysis with dictionary definition lookup
analyses = analyzer.analyze("ይቀትሉ")
for a in analyses:
    print(f"Lemma: {a.canonical_lemma} | Root: {a.root_string} | {a.tense} {a.person}{a.gender}{a.number}")
    if a.lexicon_entry:
        print(f"Definition: {a.lexicon_entry['amharic_definition'][:80]}...")

# Sentence lemmatization with clitic peeling
tokens = analyzer.lemmatize_text("ወይቀትል ፡ ሰብአ")
for t in tokens:
    lemma = t.analyses[0].canonical_lemma if t.analyses else "(unmatched)"
    print(f"Token: {t.token} -> Lemma: {lemma} (Proclitics: {t.proclitics})")
```

### 3. Neural ByT5 Training

To train a subword-free, byte-level neural sequence-to-sequence model on the synthetic morphology corpus:

```bash
pip install torch transformers datasets accelerate
python3 train_byt5.py --dataset_dir ../geez-dataset-v1 --epochs 3 --batch_size 16
```

---

## Benchmarks & Evaluation

* **Pattern Consistency:** 100% (64 distinct feature combinations induced exactly 64 conflict-free patterns).
* **Weak Verb Handling:** 8 non-aligned observations safely flagged as phonological elisions rather than inducing flawed templates.
* **Surface Form Capacity:** Generates up to ~4.98 million theoretical inflectional forms across 2,267 verified dictionary roots.

For full experimental details, see [`EVALUATION.md`](EVALUATION.md) and [`COMBINATORIAL_ANALYSIS.md`](COMBINATORIAL_ANALYSIS.md).

---

## License

Apache License 2.0. Developed for Classical Ge'ez NLP and Semitic Computational Linguistics research.
