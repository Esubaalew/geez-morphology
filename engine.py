"""
Root-and-pattern interdigitation engine.

Puts the consonants of a root into the slots of a pattern template. The
engine does not render a script itself: rendering, the vowel inventory and
the affixes come from the caller (for Ge'ez, geez_config.render_unit).

The core operation:
    interdigitate(root, pattern) -> abstract_form
    render(abstract_form, config) -> surface string

A root is an ordered tuple of abstract consonant symbols, e.g. ("q", "t", "l").
A pattern is a list of slots. Each slot is one of:
    ("R", i, v)  -- root consonant i (0-indexed) realized with vowel v
    ("F", text)  -- fixed material (affix or pattern-internal segment)

Example (abstract):
    root    = ("q", "t", "l")
    pattern = [("R", 0, "a"), ("R", 1, "a"), ("R", 2, "a")]
    result  = [("q","a"), ("t","a"), ("l","a")]

Rendering is delegated to the renderer, which maps (consonant, vowel)
pairs to characters. For Ge'ez this uses Unicode vowel-order arithmetic on
the Ethiopic block.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Union, Dict, Any, Callable


# ---------------------------------------------------------------------------
# Abstract pattern representation
# ---------------------------------------------------------------------------

# Hollow-verb contraction: (vowel on C0, vowel on C1, middle glide) -> vowel on C0.
HOLLOW_CONTRACTION = {
    ("a", "a", "w"): "o", ("a", "a", "y"): "e",
    ("I", "I", "w"): "u", ("I", "I", "y"): "i",
}
# Roots with middle w/y that keep it (ፈወሰ, ሐወጸ). Extend from the lexicon.
NON_HOLLOW = {("f", "w", "s"), ("h2", "w", "S")}
# Middle-w roots with a final guttural whose jussive/imperative contracts to the 4th order (yəqāʿ, like ቦአ ፡ ይበውእ ፡ ይባእ)
# instead of the 2nd; filled by the generator from the roots whose own Kidane Wold triple writes it so (the maintainers,
# 2026-10-08, triple_check_exceptions.tsv, q-w-a2).
HOLLOW_SUBJ_4TH = set()

# A slot is either a root slot or fixed material.
#   ("R", index, vowel)  -- index into the root, vowel is an abstract label
#   ("F", text)          -- literal fixed string (affix, particle, etc.)
Slot = Tuple[str, ...]
Pattern = List[Slot]

# An abstract form is a list of (consonant, vowel) pairs and fixed strings.
#   ("C", consonant, vowel) -- a consonant+vowel unit
#   ("F", text)             -- fixed material passed through
AbstractUnit = Tuple[str, ...]


def make_root_slot(index: int, vowel: str) -> Slot:
    """A slot filled by root consonant `index` with vowel `vowel`."""
    return ("R", index, vowel)


def make_fixed(text: str) -> Slot:
    """Fixed material (prefix, suffix, infix, pattern-internal consonant)."""
    return ("F", text)


@dataclass
class InterdigitationEngine:
    """
    Semitic root-and-pattern interdigitation engine with phonological contraction.

    The engine performs abstract interdigitation of root consonants into pattern
    slots, applies language-specific phonological glide contractions for weak roots
    (such as Ge'ez medial glide monophthongization: aw -> o, ay -> e, əw -> u),
    and delegates script rendering to a configured renderer callable.
    """

    renderer: Callable[[str, str], str] = None
    fixed_renderer: Callable[[str], str] = None

    def interdigitate(self, root: Tuple[str, ...], pattern: Pattern) -> List[AbstractUnit]:
        """
        Fill a pattern template with root consonants.

        Args:
            root: ordered consonant symbols, e.g. ("q", "t", "l").
                  May be triliteral, quadriliteral, or any length.
            pattern: list of slots as defined above.

        Returns:
            List of abstract units, e.g.
            [("C","q","a"), ("C","t","a"), ("C","l","a")].

        Raises:
            ValueError: if a root slot indexes beyond the root length.
        """
        out: List[AbstractUnit] = []
        n = len(root)
        for slot in pattern:
            if slot[0] == "R":
                _, index, vowel = slot
                if not (0 <= index < n):
                    raise ValueError(
                        f"Root slot index {index} out of range for "
                        f"root of length {n}: {root}"
                    )
                out.append(("C", root[index], vowel))
            elif slot[0] == "F":
                _, text = slot
                out.append(("F", text))
            else:
                raise ValueError(f"Unknown slot type: {slot[0]}")
        return out

    def render(self, abstract: List[AbstractUnit]) -> str:
        """
        Convert abstract units to a surface string using the configured
        renderer. Fixed material passes through (optionally transformed
        by fixed_renderer).
        """
        if self.renderer is None:
            raise ValueError("No renderer configured for this engine")
        parts = []
        for unit in abstract:
            if unit[0] == "C":
                _, consonant, vowel = unit
                parts.append(self.renderer(consonant, vowel))
            elif unit[0] == "F":
                _, text = unit
                if self.fixed_renderer is not None:
                    parts.append(self.fixed_renderer(text))
                else:
                    parts.append(text)
        return "".join(parts)

    def contract_pattern(self, root: Tuple[str, ...], pattern: Pattern, stem: str = "basic") -> Pattern:
        """Merge root slots 0 and 1 of a hollow root (middle radical w or y).

        Only these vowel pairs contract, and only in the basic stem:
          C0a + C1a -> C0o (w) / C0e (y)   perfect: ቆመ, ሞተ, ሤመ
          C0I + C1I -> C0u (w) / C0i (y)   jussive and imperative: ይቁም, ይሙት, ይሢም
        C0a + C1I (imperfect) stays uncontracted: ይቀውም. Final-weak roots
        (ተለወ, ሀለወ) and roots in NON_HOLLOW (ፈወሰ) never contract. Derived
        stems of hollow verbs follow other rules (causative of ቆመ is አቀመ),
        so they are returned unchanged.
        """
        if stem != "basic" or len(root) < 3 or root[1] not in ("w", "y") or tuple(root) in NON_HOLLOW:
            return pattern
        out, i = [], 0
        while i < len(pattern):
            slot = pattern[i]
            nxt = pattern[i + 1] if i + 1 < len(pattern) else None
            if (nxt is not None and slot[0] == "R" and slot[1] == 0
                    and nxt[0] == "R" and nxt[1] == 1):
                merged = HOLLOW_CONTRACTION.get((slot[2], nxt[2], root[1]))
                if merged == "u" and tuple(root) in HOLLOW_SUBJ_4TH:
                    merged = "A"
                if merged:
                    out.append(("R", 0, merged))
                    i += 2
                    continue
            out.append(slot)
            i += 1
        return out

    def generate(self, root: Tuple[str, ...], pattern: Pattern, stem: str = "basic") -> str:
        """Contract (hollow roots only), interdigitate, render."""
        pattern = self.contract_pattern(tuple(root), pattern, stem)
        return self.render(self.interdigitate(root, pattern))

    # ------------------------------------------------------------------
    # Analysis (inverse) helpers
    # ------------------------------------------------------------------

    def pattern_signature(self, pattern: Pattern) -> str:
        """
        A canonical string signature for a pattern, for comparison and
        deduplication. Root slots render as C{i}{vowel}, fixed as-is.
        """
        parts = []
        for slot in pattern:
            if slot[0] == "R":
                parts.append(f"C{slot[1]}{slot[2]}")
            else:
                parts.append(f"[{slot[1]}]")
        return "-".join(parts)


# ---------------------------------------------------------------------------
# Pattern utilities (also language-agnostic)
# ---------------------------------------------------------------------------

def slots_from_spec(spec: str) -> Pattern:
    """
    Parse a compact pattern specification string into a Pattern.

    Spec mini-language:
      - "C0a" means root consonant 0 with vowel "a"
      - "C1u" means root consonant 1 with vowel "u"
      - "[text]" means fixed material "text"
      - Segments are separated by "-".

    Example:
        slots_from_spec("C0a-C1a-C2a")
        -> [("R",0,"a"), ("R",1,"a"), ("R",2,"a")]

        slots_from_spec("[ya]-C0a-C1a-C2a")
        -> [("F","ya"), ("R",0,"a"), ("R",1,"a"), ("R",2,"a")]
    """
    import re
    pattern: Pattern = []
    for seg in spec.split("-"):
        seg = seg.strip()
        m = re.fullmatch(r"C(\d+)([A-Za-z_]+)", seg)
        if m:
            pattern.append(("R", int(m.group(1)), m.group(2)))
            continue
        m = re.fullmatch(r"\[(.*)\]", seg, re.DOTALL)
        if m:
            pattern.append(("F", m.group(1)))
            continue
        raise ValueError(f"Cannot parse pattern segment: {seg!r}")
    return pattern


def spec_from_slots(pattern: Pattern) -> str:
    """Inverse of slots_from_spec: canonical compact string."""
    parts = []
    for slot in pattern:
        if slot[0] == "R":
            parts.append(f"C{slot[1]}{slot[2]}")
        else:
            parts.append(f"[{slot[1]}]")
    return "-".join(parts)
