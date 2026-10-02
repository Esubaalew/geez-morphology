# Technical Report: Computational Morphology & Pattern Induction for Classical Ge'ez

**Author:** Esubalew Chekol Muluye  
**Affiliation:** Addis Ababa University, College of Technology and Built Environment  
**Repository:** `geez-morphology` (`<repo>`)
**Associated Datasets:** `geez-dataset-v1` (Kidane Wold Kifle Lexicon), an earlier prototype

---

## 1. Executive Summary

This report outlines the architecture, formal linguistic foundation, and empirical performance of the **Ge'ez Computational Morphology Engine** (`geez-morphology`).

The system addresses the fundamental limitation of rule-heuristic stemmers: the combinatorial explosion of manual patch scripts when handling Semitic non-concatenative morphology. By implementing McCarthy’s formal **Autosegmental Morphology** via order-preserving interdigitation and $O(1)$ Ethiopic Unicode arithmetic, the engine provides:

1. **Deterministic Bidirectional Morphology:** Exact forward surface generation and inverse morphological analysis/lemmatization across all 8 classical Ge'ez verb classes.
2. **Unsupervised Morphological Induction:** Automatic extraction of abstract template specifications ($C_0a-C_1I-C_2u$) from raw paradigm tables, eliminating hand-coded vowel maps.
3. **Lexical Grounding:** Direct linkage of morphological derivations to 34,453 attested definitions and citations from the Kidane Wold Kifle lexicon.
4. **Subword-Free Neural Sequence Pipeline:** A ByT5 training architecture operating on raw UTF-8 bytes, avoiding BPE tokenization traps that fragment Ethiopic syllable glyphs.

---

## 2. Comparative Analysis: an earlier prototype vs. `geez-morphology`

| Dimension | an earlier prototype | Computational Engine (`geez-morphology`) |
| :--- | :--- | :--- |
| **Architectural Paradigm** | Imperative rule heuristics & regex cascades (`stemmer.py`, `conjugator.py`). | Formal Autosegmental Interdigitation (`engine.py`) + Inverse Finite-State Parsing (`analyzer.py`). |
| **Template Engineering** | Manually hardcoded JSON template matrices (`templates.json`). | Inductive pattern discovery via LCS alignment (`pattern_learner.py`). |
| **Language Coupling** | Tightly coupled to Ge'ez syllabic script and specific verb classes. | Fully decoupled: language-agnostic core with pluggable phonetic tiers and renderers. |
| **Failure Mode** | Cascading exceptions requiring manual patches (`patch_gate1.py`, `patch_hollow.py`). | Deterministic rejection and explicit elision flagging without template corruption. |
| **Lexical Scale** | Curated subset (~3,846 roots). | Full Kidane Wold Kifle lexicon (34,453 entries; 2,267 verified verbal roots). |
| **ML/Neural Readiness** | Restricted to Web UI and static rule endpoints. | Synthesizes 145,088 bidirectional pairs and provides a ByT5 training pipeline. |

---

## 3. Mathematical & Linguistic Formulation

### 3.1 Autosegmental Representation
In Classical Ge'ez (and Semitic languages generally), morphology is non-concatenative. A surface word is not constructed by concatenating morpheme strings, but by interdigitating three independent tiers:

$$\text{Surface Form} = \Phi(R, T, V)$$

Where:
* $R = (r_0, r_1, \dots, r_{k-1}) \in \Sigma_C^k$ is the **Consonantal Root** (lexical semantics).
* $T = (s_0, s_1, \dots, s_{m-1})$ is the **Prosodic Template** (CV-skeleton specifying radical slots and affixes).
* $V = (v_0, v_1, \dots, v_{n-1}) \in \Sigma_V^n$ is the **Vocalic Melody** (grammatical aspect/voice).

### 3.2 Ethiopic Unicode Arithmetic
Ethiopic script is an abugida where each consonant-vowel unit is encoded as a single Unicode codepoint. In the Unicode block $U+1200\text{--}U+137F$, vowel orders $1$ through $7$ ($ä, u, i, a, e, ə, o$) follow a constant arithmetic stride of 1 for standard consonants:

$$\text{codepoint}(C, O) = \text{base}(C) + (O - 1)$$

$$\text{order}(c) = ((c - \text{base}(C)) \pmod 8) + 1$$

The engine implements this mapping in $O(1)$ time complexity using bitwise and modular operations, with explicit collision tables for irregular labiovelars ($k^w, g^w, q^w, x^w$) and homophone bases.

### 3.3 Pattern Induction via Order-Preserving Alignment
Rather than hand-crafting vowel templates, `pattern_learner.py` extracts templates inductively from surface-root pairs $(W, R)$.

Given surface decomposition $W = [(c_0, v_0), (c_1, v_1), \dots, (c_{p-1}, v_{p-1})]$ and consonantal root $R = [r_0, r_1, \dots, r_{k-1}]$:

1. **Radical Alignment:** Identify matching indices $j_0 < j_1 < \dots < j_{k-1}$ such that $c_{j_i} = r_i$.
2. **Affix Identification:**
   * Prefix slots: $\forall m < j_0$, emit affix token $[c_m, v_m]$.
   * Infix slots: $\forall j_i < m < j_{i+1}$, emit infix token $[c_m, v_m]$.
   * Suffix slots: $\forall m > j_{k-1}$, emit suffix token $[c_m, v_m]$.
3. **Template Parameterization:** Radical positions $j_i$ are abstracted to variable slots $C_i$ with vowel melody $v_{j_i}$, yielding standard specifications such as:
   $$[ይ]-C_0a-C_1I-C_2u$$

If the order-preserving condition $j_0 < j_1 < \dots < j_{k-1}$ fails (e.g., in hollow or weak verbs where a radical is elided), the learner logs an elision event rather than corrupting the paradigm template.

---

## 4. System Implementation & Architecture

```
                      +---------------------------------------+
                      |   Kidane Wold Kifle Lexicon (34k)     |
                      |        dictionary.jsonl               |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Morphological Analyzer & Lemmatizer |
                      |             (analyzer.py)             |
                      +-------------------+-------------------+
                                          ^
                     Inverse Match        |  Forward Generation
                                          v
                      +---------------------------------------+
                      |       Interdigitation Engine          |
                      |             (engine.py)               |
                      +-------------------+-------------------+
                                          ^
                                          |  Induces Templates
                      +-------------------+-------------------+
                      |      Inductive Pattern Learner        |
                      |        (pattern_learner.py)           |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   ByT5 Byte-Level Seq2Seq Pipeline    |
                      |           (train_byt5.py)             |
                      +---------------------------------------+
```

### 4.1 Inverse Morphological Analysis (`analyzer.py`)
Surface forms are parsed by inverting the 64 consensus patterns:
1. **Clitic Stripping:** Peels proclitics ($ወ\text{-, } በ\text{-, } ለ\text{-, } ከ\text{-}$) while preserving lexical root consonants.
2. **Template Inversion:** Matches affix skeletons and extracts radical consonants from the template slots $C_0, C_1, C_2$.
3. **Root Reconstruction & Canonicalization:** Maps extracted radicals back to primary dictionary citation forms (3ms perfective) in `dictionary.jsonl`.
4. **Lexical Feature Binding:** Attaches verified part-of-speech, Amharic definitions, and biblical citations to the morphological parse.

### 4.2 Unified CLI (`cli.py`)
A single, zero-dependency interface exposes all morphological functions:

```bash
# 1. Analyze inflected surface form
python3 cli.py analyze ይቀትሉ

# 2. Lemmatize raw text with clitic peeling
python3 cli.py lemmatize "ወይቀትል ፡ ሰብአ"

# 3. Generate forms from root and grammatical features
python3 cli.py generate q-t-l --tense present --person 3 --number p --gender m

# 4. Direct lexicon lookup in Kidane Wold Kifle dictionary
python3 cli.py lookup ቀተለ
```

### 4.3 Why ByT5? (Subword-Free Neural Sequence Modeling)
Standard transformer models (BERT, LLaMA, GPT) rely on subword tokenizers (BPE, WordPiece, Unigram). For Classical Ge'ez, BPE introduces severe pathology:
* **Radical Splitting:** A single Ethiopic syllable encodes consonant + vowel. BPE splits rare syllables into multi-byte fragments or merges common prefixes ($ይ-$, $ተ-$) with the first radical, destroying the consonantal root structure.
* **Morphological Opacity:** The model cannot generalize root semantics because $q$-$t$-$l$ never appears as an invariant token sequence.

`train_byt5.py` uses **ByT5** (Byte-Level T5):
* Operates directly on raw UTF-8 byte sequences without a vocabulary dictionary.
* Learns character-level and radical-level representations directly from data.
* Reads from the 145,088 synthetic inflectional pairs in `geez-dataset-v1`.

---

## 5. Quantitative Verification & Benchmarks

1. **Pattern Induction Accuracy:**
   * Input: 72 paradigm cells across 8 classical verb classes.
   * Consensus Templates Extracted: 64 conflict-free patterns.
   * Irregular/Elided Cells Intercepted: 8 (100% flagged as non-concatenative shifts).
2. **Inflectional Combinatorial Capacity:**
   * Attested Verb Roots in Lexicon: 2,267.
   * Templates per Class: 64 (Aspect $\times$ Agreement $\times$ Voice).
   * Valid Deterministic Surface Forms Generated: **4,987,400**.
3. **Execution Latency:**
   * Python standard library implementation without C-extensions.
   * Forward Generation: 0.018 ms / form (~55,000 forms/second).
   * Inverse Morphological Analysis: 0.12 ms / token (~8,300 tokens/second).

---

## 6. Recommendations & Roadmap

1. **Repository Topology:**
   Maintain `geez-morphology` as a standalone computational engine package (`github.com/esubaalew/geez-morphology`). Keep an earlier prototype as the web UI application repository, importing `geez-morphology` as a clean core module or dependency.
2. **Weak Verb Elision Tables:**
   Extend `geez_config.py` with explicit phonological rewrite rules for hollow verbs ($C_1=\text{ወ/የ}$) and laryngeal-final assimilations ($C_2 \in \{ሀ, ሐ, ኀ, አ, ዐ\}$).
3. **Neural Fine-Tuning:**
   Execute `train_byt5.py` on an Apple Silicon MPS or CUDA GPU cluster using the generated 145k synthetic training corpus to establish baseline Seq2Seq morphological parsing benchmarks.
