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
    "khw": 0x1288,  # ኈ xʷ (labiovelar; was missing)
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
    "gw": 0x1310,   # ጐ gʷ (labiovelar; was missing)
    "T":  0x1320,
    "p":  0x1330,
    "S":  0x1338,
    "S2": 0x1340,
    "f":  0x1348,
    "p2": 0x1350,
}

# Note: "qh" (ቈ) and "kh2" (ኰ) are labiovelars qʷ and kʷ, not aspirates; the
# names are kept because the dictionary roots use them. Labiovelar rows have no
# u or o order, so render_unit(qh/kh2/gw/khw, "u" or "o") gives an unassigned
# code point; the generator drops such forms.

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


# the maintainers (2026-10-05): after a guttural, 1st and 4th order are spelling
# variants. Generation writes the 1st order; comparison treats both as equal.
GUTTURAL_4TH_TO_1ST = str.maketrans("ኣዓሓኃሃ", "አዐሐኀሀ")
GUTTURAL_SYMBOLS = ("a", "a2", "h1", "h2", "kh")


def normalize_guttural_4th(s: str) -> str:
    """ኣ ዓ ሓ ኃ ሃ -> አ ዐ ሐ ኀ ሀ; nothing else changes."""
    return s.translate(GUTTURAL_4TH_TO_1ST) if s else s


def gutturals_equal(a: str, b: str) -> bool:
    return normalize_guttural_4th(a) == normalize_guttural_4th(b)


def decompose_string(s: str):
    """Decompose each character; returns list of (consonant, vowel) or (None, None)."""
    return [decompose_char(ch) for ch in s]


# Scholarly transliteration used in the Merho Ges gold roots -> Ethiopic letter.
TRANSLIT_TO_ETHIOPIC = {
    "ʾ": "አ", "ʼ": "አ", "'": "አ", "ʔ": "አ",
    "ʿ": "ዐ", "ʻ": "ዐ", "ʕ": "ዐ",
    "h": "ሀ", "ḥ": "ሐ", "ħ": "ሐ", "ḫ": "ኀ",
    "ś": "ሠ", "ṣ": "ጸ", "ḍ": "ፀ", "ṣ́": "ፀ", "ṭ": "ጠ",
    "qʷ": "ቈ", "qw": "ቈ", "ḳʷ": "ቈ", "kʷ": "ኰ", "gʷ": "ጐ", "ḫʷ": "ኈ",
}


def to_ethiopic_root(root_str: str) -> str:
    """Root string -> hyphenated Ethiopic first-order letters, e.g. ቀ-ተ-ለ.

    Accepts Ethiopic roots ("ሰ-ነ-አ-ወ", "m-r-ḥ (መ-ር-ሐ)"), consonant symbols
    ("q-t-l", "h2-S-b"; case matters: T = ጠ but t = ተ, S = ጸ but s = ሰ,
    S2 = ፀ but s2 = ሠ), and the scholarly transliteration of the gold roots
    ("ṭ-y-b", "m-m-ʿ"). Unknown parts are kept as written.
    """
    if not root_str:
        return ""
    if any(0x1200 <= ord(c) <= 0x137F for c in root_str):
        return "-".join(c for c in root_str if 0x1200 <= ord(c) <= 0x137F)
    out = []
    for p in root_str.split("(")[0].split("-"):
        p = p.strip()
        if not p:
            continue
        if p in CONSONANTS:                      # exact, case-sensitive first
            out.append(render_unit(p, "a"))
        elif p in TRANSLIT_TO_ETHIOPIC:
            out.append(TRANSLIT_TO_ETHIOPIC[p])
        elif p.lower() in CONSONANTS:
            out.append(render_unit(p.lower(), "a"))
        else:
            out.append(p)
    return "-".join(out)


def first_order_root(root_str: str) -> str:
    """Root -> every letter in the 1st order of its own series (a root label is
    consonants only), e.g. ን-ስ-ሕ-ለ-ል -> ነ-ሰ-ሐ-ለ-ለ, ር-እ-የ -> ረ-አ-የ, ሦ -> ሠ.

    Labialized series stay labialized: ቍ -> ቈ, ኵ -> ኰ, ጒ -> ጐ, ኍ -> ኈ. A 7th-order
    ቆ/ኮ/ጎ is taken as written, i.e. in the plain series (ቀ/ከ/ገ) (the maintainers
    rules). Latin roots go through to_ethiopic_root first. Letters that
    are not in the consonant table (e.g. ጘ) are kept as written.
    """
    out = []
    for part in to_ethiopic_root(root_str).split("-"):
        cons = decompose_char(part)[0] if len(part) == 1 else None
        out.append(render_unit(cons, "a") if cons else part)
    return "-".join(out)


def root_symbols(root_str: str) -> tuple:
    """Root in any script -> tuple of consonant symbols, e.g. ("q", "t", "l").

    Parts that are not a known letter are kept as written, so a bad root
    stays visibly bad instead of silently losing a radical.
    """
    out = []
    for part in to_ethiopic_root(root_str).split("-"):
        part = part.strip()
        if not part:
            continue
        cons = decompose_char(part)[0] if len(part) == 1 else None
        out.append(cons or part)
    return tuple(out)


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
    "strong_B": "Type B, doubled middle radical, e.g. ቀደሰ (imperfect ይቄድስ). Class comes from the lexicon, not the spelling.",
    "strong_C": "Type C, long a after the first radical, e.g. ባረከ (imperfect = subjunctive ይባርክ).",
    "weak_initial": "First radical is w/y, e.g. ወለደ (ይወልድ; subjunctive ይለድ only for W_DROP roots).",
    "weak_middle": "Second radical is w/y, e.g. ሠይመ, ዐወቀ, ከወነ.",
    "weak_final": "Third radical is w/y, e.g. ተለወ (ይተሉ), ሰትየ (ይሰቲ).",
    "irregular": "Suppletive or highly irregular, e.g. ብህለ (to say).",
    "quadriliteral": "Four radicals, e.g. መስከረ.",
}
