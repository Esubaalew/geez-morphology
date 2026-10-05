"""
Ge'ez language configuration for the general root-and-pattern engine.

This is the ONLY Ge'ez-hardcoded part of the system. Everything else
(engine.py, pattern_learner.py) is language-agnostic.

Contents:
  1. Vowel inventory: the 7 Ethiopic vowel orders with abstract labels.
  2. Consonant inventory: abstract symbol -> base codepoint mapping.
  3. Renderer: (consonant, vowel) -> Ethiopic character via Unicode
     vowel-order arithmetic (the usual approach).
  4. Decomposer: character -> (consonant, vowel), inverse of renderer.
  5. Affix inventories: subject prefixes/suffixes per tense (for reference;
     the learner discovers how they combine, they are documented here).

Vowel order labels follow the traditional Ethiopic order:
  0: a (ሀ)   1: u (ሁ)   2: i (ሂ)   3: A (ሃ)
  4: e (ሄ)   5: I (ህ)   6: o (ሆ)
"""

# Abstract vowel labels -> Ethiopic vowel order (0-6)
VOWEL_ORDERS = {
    "a": 0,
    "u": 1,
    "i": 2,
    "A": 3,
    "e": 4,
    "I": 5,
    "o": 6,
}

ORDER_TO_VOWEL = {v: k for k, v in VOWEL_ORDERS.items()}

# Abstract consonant symbol -> base codepoint (the "a"-order character).
CONSONANTS = {
    "h1": 0x1200,
    "l":  0x1208,
    "h2": 0x1210,
    "m":  0x1218,
    "s2": 0x1220,
    "r":  0x1228,
    "s":  0x1230,
    "sh": 0x1238,
    "q":  0x1240,
    "qh": 0x1248,
    "b":  0x1260,
    "t":  0x1270,
    "ch": 0x1278,
    "kh": 0x1280,
    "n":  0x1290,
    "ny": 0x1298,
    "a":  0x12A0,
    "k":  0x12A8,
    "kh2": 0x12B0,
    "w":  0x12C8,
    "a2": 0x12D0,
    "z":  0x12D8,
    "zh": 0x12E0,
    "y":  0x12E8,
    "d":  0x12F0,
    "g":  0x1308,
    "T":  0x1320,
    "p":  0x1330,
    "S":  0x1338,
    "S2": 0x1340,
    "f":  0x1348,
    "p2": 0x1350,
}

BASE_TO_CONSONANT = {v: k for k, v in CONSONANTS.items()}


def render_unit(consonant: str, vowel: str) -> str:
    """
    Render an abstract (consonant, vowel) pair to an Ethiopic character
    using Unicode vowel-order arithmetic.
    """
    if consonant not in CONSONANTS:
        raise ValueError(f"Unknown consonant symbol: {consonant!r}")
    if vowel not in VOWEL_ORDERS:
        raise ValueError(f"Unknown vowel label: {vowel!r}")
    return chr(CONSONANTS[consonant] + VOWEL_ORDERS[vowel])


def decompose_char(ch: str):
    """
    Inverse of render_unit: Ethiopic character -> (consonant, vowel).
    Returns (consonant_symbol, vowel_label), or (None, None) if the
    character is not a recognized consonant+vowel syllable.
    Uses the usual arithmetic: base = (cp // 8) * 8, order = cp % 8.
    """
    if len(ch) != 1:
        return (None, None)
    cp = ord(ch)
    if not (0x1200 <= cp <= 0x137F):
        return (None, None)
    order = cp % 8
    if order > 6:
        return (None, None)
    base = (cp // 8) * 8
    consonant = BASE_TO_CONSONANT.get(base)
    if consonant is None:
        return (None, None)
    return (consonant, ORDER_TO_VOWEL[order])


def decompose_string(s: str):
    """Decompose each character; returns list of (consonant, vowel) or (None, None)."""
    return [decompose_char(ch) for ch in s]


# ---------------------------------------------------------------------------
# Reference affix inventories (documented; the learner discovers usage)
# ---------------------------------------------------------------------------

SUBJECT_PREFIXES = {
    "past": {"3ms": "", "3fs": "", "2ms": "", "2fs": "", "1cs": "",
             "3mp": "", "3fp": "", "2mp": "", "2fp": "", "1cp": ""},
    "present": {"3ms": "ይ", "3fs": "ት", "2ms": "ት", "2fs": "ት",
                "1cs": "እ", "3mp": "ይ", "3fp": "ይ",
                "2mp": "ት", "2fp": "ት", "1cp": "ን"},
    "subjunctive": {"3ms": "ይ", "3fs": "ት", "2ms": "ት", "2fs": "ት",
                    "1cs": "እ", "3mp": "ይ", "3fp": "ይ",
                    "2mp": "ት", "2fp": "ት", "1cp": "ን"},
    "jussive": {"3ms": "ይ", "3fs": "ት", "2ms": "ት", "2fs": "ት",
                "1cs": "እ", "3mp": "ይ", "3fp": "ይ",
                "2mp": "ት", "2fp": "ት", "1cp": "ን"},
    "imperative": {"2ms": "", "2fs": "", "2mp": "", "2fp": ""},
}

VERB_TYPES = {
    "strong_A": "Triliteral, e.g. ቀተለ (q-t-l). Full 7-order contrast.",
    "strong_B": "Triliteral with a-vowel in present stem, e.g. ባረከ.",
    "weak_initial": "First radical is weak (ኀ, አ), e.g. ኀደገ, አዘዘ.",
    "weak_middle": "Second radical is w/y, e.g. ሠይመ, ዐወቀ, ከወነ.",
    "weak_final": "Third radical is w/y, e.g. ሀለወ.",
    "irregular": "Suppletive or highly irregular, e.g. ብህለ (to say).",
    "quadriliteral": "Four radicals, e.g. መስከረ.",
}
