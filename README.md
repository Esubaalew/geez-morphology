# Ge'ez Computational Morphology Engine

A modular, high-performance root-and-pattern interdigitation engine and morphological pattern induction learner for Semitic languages, configured for Classical Ge'ez.

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
* **Deterministic Precision:** Zero hallucinations; 100% deterministic realization across standard inflectional paradigms.
* **Ethiopic Unicode Geometry:** Leverages arithmetic offsets in the Unicode Ethiopic block (`U+1200–U+137F`) for $O(1)$ syllable synthesis without heavy mapping dictionaries.
* **Unsupervised Pattern Induction:** Discovers morphological templates directly from annotated paradigm tables using an order-preserving alignment algorithm (`pattern_learner.py`).
* **Fail-Safe Elision Detection:** Automatically intercepts and flags weak-radical elision and irregular shifts without guessing or corrupting template slots.

---

## Repository Structure

| File | Purpose |
| :--- | :--- |
| `engine.py` | Universal, language-agnostic root-and-pattern interdigitation core. |
| `pattern_learner.py` | Inductive template learner extracting pattern specs from surface pairs. |
| `geez_config.py` | Ge'ez phonetic inventory, order mappings, and Ethiopic character renderer. |
| `geez_config.json` | Declarative configuration export of Ge'ez vocalic and consonantal tiers. |
| `patterns.json` | 64 verified morphological patterns induced from classical grammar matrices. |
| `EVALUATION.md` | Quantitative benchmark results, consistency checks, and class generalization tests. |
| `COMBINATORIAL_ANALYSIS.md` | Mathematical derivation of the ~5 million form inflectional capacity. |
| `GENERALIZATION.md` | Cross-linguistic architectural comparison across Semitic languages. |

---

## Quickstart

### Prerequisites
Python 3.8+ (No external package dependencies required; uses standard library).

```bash
git clone https://github.com/esubaalew/geez-morphology.git
cd geez-morphology
```

### Basic Generation
```python
from engine import InterdigitationEngine, slots_from_spec
from geez_config import render_unit

# Initialize engine with Ge'ez Unicode renderer
engine = InterdigitationEngine(renderer=render_unit)

# Define root and abstract pattern
root = ("q", "t", "l")
pattern = slots_from_spec("C0a-C1a-C2a")

# Generate surface form
surface_form = engine.generate(root, pattern)
print(surface_form)
# Output: ቀተለ
```

### Template Induction from Surface Data
```python
from pattern_learner import PatternLearner
from geez_config import decompose_char

learner = PatternLearner(decomposer=decompose_char)

# Observe annotated pairs
learner.observe(
    root=("q", "t", "l"), 
    surface="ይቀትል", 
    features={"tense": "present", "person": "3ms", "stem": "basic"}
)

pattern = learner.consensus_pattern({"tense": "present", "person": "3ms", "stem": "basic"})
print(pattern.spec)
# Output: [ይ]-C0a-C1I-C2I
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
