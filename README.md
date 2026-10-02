# A Computational Model of Semitic Root-and-Pattern Morphology
## Demonstrated on Ge'ez

### Abstract

We present a computational model of Semitic root-and-pattern morphology
that separates the **universal interdigitation mechanism** from
**language-specific parameters**. The core engine maps ordered root
consonants into abstract pattern templates; a pattern learner induces
templates from annotated data rather than hardcoding them; Ge'ez is
implemented as a configuration layer (vowel inventory, consonant-to-script
mapping via Unicode arithmetic, affix inventories). Evaluated on a cleaned
corpus of Ge'ez verb paradigms, the learner induces 64 patterns with 100%
internal consistency and correctly flags weak-verb phenomena. A
compositional generalization test reveals the Ge'ez Type A/B verb class
distinction as a systematic vowel-melody difference on a shared template.
Combinatorial analysis shows ~5 million generable verb forms from 2,267
attested roots. We discuss what is universal versus language-specific and
specify requirements for a truly universal Semitic morphology engine.

---

## 1. Introduction

Semitic languages build words by **interdigitating** root consonants into
pattern templates. A Ge'ez triliteral root like q-t-l (ቀተለ, "to kill")
combines with vowel patterns and affixes to produce:

- ቀተለ (qatala, "he killed") — past 3ms
- ይቀትል (yaqattal, "he kills") — present 3ms
- ይቅትል (yaqtal, "let him kill") — jussive 3ms
- ቅትል (qtal, "kill!") — imperative 2ms
- አቅተለ (aqtala, "he caused to kill") — causative past 3ms
- ተቀተለ (taqatala, "he was killed") — passive past 3ms

The root consonants (q, t, l) appear in order in every form. What changes
is the vocalic melody and the affixal material. This is the root-and-pattern
system, shared across Ge'ez, Arabic, Hebrew, Amharic, Aramaic, and all
Semitic languages.

**Thesis**: The interdigitation mechanism is universal. What varies by
language is (a) the consonant and vowel inventories, (b) the specific
pattern templates, (c) the script rendering function, and (d) verb class
conditioning. A computational model should reflect this separation.

## 2. Formalism

### 2.1 Roots

A root is an **ordered tuple** of abstract consonant symbols:

```
root = ("q", "t", "l")
```

Triliteral (3), quadriliteral (4), or any length. Order is significant.

### 2.2 Patterns

A pattern is a **list of slots**. Each slot is:

- `("R", i, v)` — root consonant `i` (0-indexed) with vowel `v`
- `("F", text)` — fixed material (affix or pattern-internal segment)

Example — Ge'ez past 3ms:
```
pattern = [("R", 0, "a"), ("R", 1, "a"), ("R", 2, "a")]
```

Example — Ge'ez present 3ms:
```
pattern = [("F", "ይ"), ("R", 0, "a"), ("R", 1, "I"), ("R", 2, "I")]
```

Compact notation: `C0a-C1a-C2a` and `[ይ]-C0a-C1I-C2I`.

### 2.3 Interdigitation

```
interdigitate(root, pattern) -> abstract form
```

Walk the pattern; for each `("R", i, v)` slot, emit `(root[i], v)`;
for each `("F", text)` slot, emit the text unchanged. This operation
is **language-agnostic**.

### 2.4 Rendering

```
render(abstract_form, config) -> surface string
```

The language configuration maps `(consonant, vowel)` pairs to script
characters. For Ge'ez, this uses Unicode vowel-order arithmetic on the
Ethiopic block (the usual approach):

```
character = base_codepoint(consonant) + vowel_order(vowel)
```

For Arabic, the renderer would output consonant + diacritic sequences.
The interdigitation logic is unchanged.

## 3. Architecture

```
engine.py            UNIVERSAL - InterdigitationEngine
pattern_learner.py   UNIVERSAL - PatternLearner (takes decomposer fn)
geez_config.py       GE'EZ-SPECIFIC - vowel/consonant inventories, renderer
```

## 4. Pattern Learning

Rather than hardcoding Ge'ez verb tables, patterns are **induced** from
annotated (root, surface_form, features) triples via order-preserving
alignment. See EVALUATION.md for results: 64/64 feature combinations
induced exactly one pattern (100% consistency); 8 alignment failures,
all genuine weak-verb elision (correctly flagged, not guessed).

## 5. Key Findings

### 5.1 The Type A/B Distinction (Discovered, Not Assumed)

A compositional generalization test (train on q-t-l, test on b-r-k)
achieved 0% accuracy — but every error was systematic: the template
transfers perfectly, only the C0 vowel differs (1st order vs 4th order).
This is the Ge'ez strong verb class distinction, discovered by the
experiment rather than stipulated.

### 5.2 Combinatorial Productivity

- 2,267 distinct verb roots, 220 subject-inflected forms per root,
  2,200 with object suffixes = **~5 million theoretically generable
  verb forms**. Attested in dictionaries: <0.1%. See
  COMBINATORIAL_ANALYSIS.md.

### 5.3 Universal vs Language-Specific

See GENERALIZATION.md. Universal: interdigitation, ordered roots, slot
templates, affixation. Language-specific: vowel inventory, pattern
shapes, script rendering, verb class conditioning, phonological rules.

## 6. Files

| File | Description |
|------|-------------|
| `engine.py` | Language-agnostic interdigitation engine |
| `pattern_learner.py` | Pattern induction from data |
| `geez_config.py` | Ge'ez configuration (renderer, inventories) |
| `geez_config.json` | Ge'ez configuration (JSON export) |
| `patterns.json` | 64 induced patterns with examples |
| `EVALUATION.md` | Quantitative evaluation |
| `COMBINATORIAL_ANALYSIS.md` | Productivity quantification |
| `GENERALIZATION.md` | Universal vs language-specific analysis |
| `README.md` | This file |

## 7. Limitations

1. Weak verbs: learner flags elision but does not model it.
2. Quadriliterals: engine supports n-consonant roots, untested on data.
3. Nominal morphology: 95 noun records not used for induction.
4. Cross-language: designed language-agnostic, tested only on Ge'ez.
5. Phonological rules: no post-interdigitation rewrite rules.

## 8. Future Work

1. Phonological rewrite rules for weak verbs.
2. Run learner on Arabic and Hebrew data.
3. Use engine as synthetic data generator for neural training.
4. Extend to nominal patterns.
5. Pattern inheritance (causative as transformation).

## 9. Data Sources

- Morphology: geez-dataset-v1/morphology.jsonl (208 records)
- Dictionary: geez-dataset-v1/dictionary.jsonl (13,895 entries)
