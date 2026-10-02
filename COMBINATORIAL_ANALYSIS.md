# Combinatorial Analysis: Quantifying Ge'ez Surface Form Productivity

## 1. The Claim

Ge'ez root-and-pattern morphology generates an enormous number of surface
forms from a small inventory of roots. This document quantifies that claim.

## 2. Counting Roots

**Method**: Extract consonantal skeletons from verb headwords in the cleaned
dictionary via Unicode vowel-order arithmetic.

**Result**: 2,267 distinct verb roots (3-4 consonant skeletons) from 3,968
verb entries (out of 13,895 total dictionary entries).

## 3. Counting Patterns per Root

### 3.1 Subject Inflection

| Tense | Forms |
|-------|-------|
| Past | 10 |
| Present | 10 |
| Subjunctive | 10 |
| Jussive | 10 |
| Imperative | 4 |
| **Total per stem** | **44** |

### 3.2 Derived Stems

| Stem | Prefix | Example |
|------|--------|---------|
| Basic | - | ቀተለ |
| Causative | አ- | አቅተለ |
| Passive | ተ- | ተቀተለ |
| Reciprocal | ት- | ትቃተለ |
| Causative-Reciprocal | አስተ- | አስተቃተለ |

5 productive stems x 44 forms = **220 subject-inflected forms per root**.

### 3.3 Object Suffixes

10 pronominal object suffixes x 44 subject forms = 440 per stem.
5 stems = **2,200 inflected forms per root**.

## 4. The Total

| Scope | Total |
|-------|-------|
| Per root, subject only | 220 |
| Per root, with objects | 2,200 |
| All roots, subject only | 2,267 x 220 = **498,740** |
| All roots, with objects | 2,267 x 2,200 = **4,987,400** |

**~5 million theoretically generable verb forms** from the attested root
inventory, inflectional morphology only.

## 5. What About "Billions"?

Adding derivational morphology (verbal nouns, participles, adjectives),
nominal roots, and compounding pushes toward tens of millions. "Billions"
is rhetorical unless counting recursive derivation and unattested combinations.

**Honest assessment for a thesis**: The productive inflectional system
generates **millions** of forms from thousands of roots. "Millions from
thousands" is the defensible claim.

## 6. Attested vs Theoretical

| Metric | Count |
|--------|-------|
| Theoretical verb forms | ~4,987,400 |
| Dictionary verb entries | 3,968 |
| Morphology paradigm forms | ~200 |
| Coverage | <0.1% |

**The language's productive capacity vastly exceeds any dictionary.**
A model that memorizes surface forms will never cover the language.
A model that learns the generative process (root + pattern) can produce
any of the 5 million forms from 2,267 roots + ~64 patterns.

Representational compression: 5M forms -> (2,267 roots x 64 patterns),
nearly 100x reduction. This is why the root-and-pattern engine matters.
