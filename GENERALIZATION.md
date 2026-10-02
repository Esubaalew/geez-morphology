# Architectural Generalization: Universal Engine & Language-Specific Parameters

## 1. Architectural Principles

Semitic root-and-pattern morphology operates through non-concatenative interdigitation: ordered root consonants are interleaved into abstract template slots with specified vocalic melodies and affixal material. 

This engine decouples:
1. **Universal Core:** The language-independent mechanism that manipulates ordered sequences of root consonants, slot positions, and affix passes.
2. **Language-Specific Parameters:** Inventories of consonants and vowels, script renderers, verb-class conditioning, and morphophonological rewrite rules.

---

## 2. Universal Layer Components

### 2.1 The Consonantal Root as an Ordered Tuple
A root is represented as an immutable, ordered sequence of abstract consonant tokens:
$$\mathbf{R} = (c_0, c_1, \dots, c_{n-1})$$
Relative ordering is strictly preserved across all inflections ($q\text{-}t\text{-}l \ne t\text{-}q\text{-}l$). The engine accommodates triliteral ($n=3$), quadriliteral ($n=4$), and multiliteral roots without architectural modification.

### 2.2 Template Slots
A pattern $\mathbf{P}$ is formalized as an ordered list of typed slots:
* **Root Slots:** $(\text{R}, i, v)$ where $i \in [0, n-1]$ maps to consonant $c_i$, and $v$ specifies an abstract vocalic designation.
* **Affix / Fixed Slots:** $(\text{F}, s)$ where $s$ is an invariable prefix, infix, or suffix string.

```
Example: Past 3ms Template (Ge'ez):
  P_past3ms = [("R", 0, "a"), ("R", 1, "a"), ("R", 2, "a")]

Example: Imperfective 3ms Template (Ge'ez):
  P_impf3ms = [("F", "ይ"), ("R", 0, "a"), ("R", 1, "I"), ("R", 2, "I")]
```

### 2.3 Language-Agnostic Interdigitation
The core operation maps $\mathbf{R} \times \mathbf{P} \to \mathbf{U}$ (abstract morphological units):
$$\text{interdigitate}(R, P) = [u_0, u_1, \dots, u_m]$$
where each $u_k$ is either a realized consonant-vowel unit $(c_i, v)$ or an unadorned fixed token $(\text{F}, s)$. This step performs zero language-specific rendering.

---

## 3. Language-Specific Parameterization

| Dimension | Classical Ge'ez | Classical Arabic | Biblical Hebrew |
| :--- | :--- | :--- | :--- |
| **Orthographic Encoding** | Syllabic fusion (Unicode block arithmetic: $c + v \to$ single fidel glyph) | Concatenative consonant letter + diacritic harakat | Consonant letter + niqqud pointings |
| **Vocalic Inventory** | 7 orders (ግዕዝ, ካዕብ, ሣልስ, ራብዕ, ኃምስ, ሳድስ, ሳብዕ) | 3 short vowels, 3 long vowels, 2 diphthongs | 7 full vowels, reduced vocalic shewa |
| **Stem Paradigms** | 5 core productive stems (ቀተለ, አቅተለ, ተቀተለ, ተቃተለ, አስተቃተለ) | 10–15 forms (*awzān* / *abwāb*) | 7 primary *binyanim* |
| **Verb Class Splits** | Type A (ቀተለ) vs. Type B (ባረከ) | Form I vocalism alternations | Weak and guttural verb classes |

---

## 4. Script Rendering Mechanics

In Ge'ez, each character in the Ethiopic syllabary fuses a consonant base with one of the 7 vocalic orders. The engine exploits Ethiopic Unicode geometry:

$$\text{codepoint}(c, v) = \text{base\_codepoint}(c) + \text{order\_offset}(v)$$

For scripts with explicit diacritics (Arabic/Hebrew), the renderer emits multi-codepoint sequences ($c_i + \text{diacritic}$). The underlying interdigitation engine remains completely unchanged.

---

## 5. Architectural Requirements for Scaling

To extend this engine into a comprehensive Semitic computational framework:
1. **Autosegmental Phonology Tier:** Incorporate two-level morphology (PC-KIMMO style) or finite-state rewrite transducers (FST) to handle post-interdigitation phonological assimilation (e.g., guttural vowel lowering, glide contraction).
2. **Weak-Radical Compensatory Rules:** Explicit mapping of elision and lengthening when radical $w$ or $y$ interacts with vowels.
3. **Multi-Task Neural Distillation:** Use the deterministic engine as an exact training data generator to train small, robust sequence-to-sequence neural models for out-of-vocabulary smoothing.
